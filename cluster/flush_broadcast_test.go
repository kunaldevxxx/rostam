// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/shard"
)

// keysCoveringEveryShard returns one key per shard group [0,numShards), each key
// hashing (via shardOf, the SAME router Node.Call uses) to its index. It is the
// fixture the flush-broadcast test needs: a keyspace spread across EVERY group, so
// that a flush wiping all of them can only be explained by the broadcast reaching
// every group — not just group 0, where a keyless op would otherwise route.
func keysCoveringEveryShard(t *testing.T, numShards int) [][]byte {
	t.Helper()
	keys := make([][]byte, numShards)
	filled := 0
	for i := 0; filled < numShards; i++ {
		k := []byte(fmt.Sprintf("flush-key-%d", i))
		g := shardOf(k, numShards)
		if keys[g] == nil {
			keys[g] = k
			filled++
		}
		if i > 100000 {
			t.Fatalf("could not cover all %d shards after 100k candidates (filled %d)", numShards, filled)
		}
	}
	return keys
}

// TestNodeFlushBroadcastWipesEveryShardGroup is the core proof of Stage 2: a single
// flush call, dispatched to a node hosting many independent shard groups, must empty
// EVERY group's keyspace — not only group 0, where a keyless op routes by default.
//
// It seeds one key per group (each hashing to a distinct group), confirms every key
// reads back, calls flush once, then asserts every key now misses. Because the keys
// live in different Raft groups with independent caches, an all-miss result can only
// come from the broadcast landing a flush in each group's log (broadcastFlush).
func TestNodeFlushBroadcastWipesEveryShardGroup(t *testing.T) {
	const numShards = 8
	n := newTestNode(t, numShards)

	keys := keysCoveringEveryShard(t, numShards)

	// Seed and confirm each key is stored in its group.
	for _, k := range keys {
		if _, err := n.Call("put", ops.EncodePutArgs(k, []byte("v"), 0)); err != nil {
			t.Fatalf("put %q: %v", k, err)
		}
	}
	for i, k := range keys {
		res, err := n.Call("get", ops.EncodeKeyArgs(k))
		if err != nil {
			t.Fatalf("get %q (shard %d) before flush: %v", k, i, err)
		}
		if string(res) != "v" {
			t.Fatalf("get %q (shard %d) before flush = %q, want v", k, i, res)
		}
	}

	// One flush call — the node must fan it out to every group.
	if _, err := n.Call("flush", nil); err != nil {
		t.Fatalf("flush: %v", err)
	}

	// Every group must now be empty. A single group left non-empty means the
	// broadcast missed it.
	for i, k := range keys {
		_, err := n.Call("get", ops.EncodeKeyArgs(k))
		if err == nil {
			t.Fatalf("key %q (shard %d) still present after flush — broadcast missed its group", k, i)
		}
		if !isNotFoundErr(err) {
			t.Fatalf("get %q (shard %d) after flush: unexpected err %v, want not-found", k, i, err)
		}
	}

	// The store is live, not poisoned: a post-flush write reads back.
	postKey := keys[numShards-1]
	if _, err := n.Call("put", ops.EncodePutArgs(postKey, []byte("w"), 0)); err != nil {
		t.Fatalf("put after flush: %v", err)
	}
	res, err := n.Call("get", ops.EncodeKeyArgs(postKey))
	if err != nil {
		t.Fatalf("get after post-flush put: %v", err)
	}
	if string(res) != "w" {
		t.Fatalf("post-flush get = %q, want w", res)
	}
}

// TestNodeFlushBroadcastAcrossPartitionedCluster proves the flush fan-out crosses
// NODE boundaries, not just group boundaries. With ReplicationFactor=1 over 3 nodes
// and 6 shard groups, each group lives on exactly ONE node, so a flush issued to a
// single node MUST forward a per-group leg (opFlushShardName) to the owner of every
// group it does not host — the proposeFlush leader-hop path. It seeds keys across
// all groups, flushes via one node, then asserts every key misses cluster-wide: a
// group whose remote owner never received the forwarded flush would still hold its
// keys.
func TestNodeFlushBroadcastAcrossPartitionedCluster(t *testing.T) {
	const nodes, numShards, rf = 3, 6, 1
	tc := newTestCluster(t, nodes, numShards, rf)
	ctx := context.Background()

	keys := keysCoveringEveryShard(t, numShards)

	// Seed via the routing client (each key reaches its group's owner) and wait for
	// every node to apply.
	for _, k := range keys {
		if _, err := tc.client.Call(ctx, "put", ops.EncodePutArgs(k, []byte("v"), 0)); err != nil {
			t.Fatalf("put %q: %v", k, err)
		}
	}
	for _, n := range tc.nodes {
		waitAllApplied(t, n)
	}
	for i, k := range keys {
		got, err := tc.client.Call(ctx, "get", ops.EncodeKeyArgs(k))
		if err != nil {
			t.Fatalf("get %q (shard %d) before flush: %v", k, i, err)
		}
		if string(got) != "v" {
			t.Fatalf("get %q (shard %d) before flush = %q, want v", k, i, got)
		}
	}

	// One flush issued to a SINGLE node — it must fan out to every group's owner,
	// hopping across nodes for the groups it does not host itself.
	if _, err := tc.nodes[0].Call("flush", nil); err != nil {
		t.Fatalf("flush via node 0: %v", err)
	}

	// Every key must now miss cluster-wide.
	for i, k := range keys {
		_, err := tc.client.Call(ctx, "get", ops.EncodeKeyArgs(k))
		if err == nil {
			t.Fatalf("key %q (shard %d) still present after flush — a remote group's owner was not reached", k, i)
		}
		if !isNotFoundErr(err) {
			t.Fatalf("get %q (shard %d) after flush: unexpected err %v, want not-found", k, i, err)
		}
	}
}

// TestNodeFlushBroadcastWipesBackupReplica proves the flush is applied on BACKUP
// replicas, not only on each group's leader. The two other broadcast tests run
// RF=1 (one owner per group), so they show fan-out reaches every group/node but do
// NOT prove a follower applies the wipe. Here RF=3 over 3 nodes puts every group on
// all three nodes; after a single flush, this reads each key from a FOLLOWER's LOCAL
// cache (getShard(idx).Get bypasses leader routing), so an all-miss result can only
// mean the backup applied the flush through its own Raft log — the property RF=1
// asserts only by construction. (Correctness rests on flush using the identical
// propose path as a normal write; this closes it by execution.)
func TestNodeFlushBroadcastWipesBackupReplica(t *testing.T) {
	const nodes, numShards, rf = 3, 6, 3
	tc := newTestCluster(t, nodes, numShards, rf)
	ctx := context.Background()

	keys := keysCoveringEveryShard(t, numShards)

	// Seed via the routing client, then wait for every replica to apply.
	for _, k := range keys {
		if _, err := tc.client.Call(ctx, "put", ops.EncodePutArgs(k, []byte("v"), 0)); err != nil {
			t.Fatalf("put %q: %v", k, err)
		}
	}
	for _, n := range tc.nodes {
		waitAllApplied(t, n)
	}

	// For each key, pin a FOLLOWER store hosting its group and confirm the key is
	// present there BEFORE the flush — reading a follower's local cache (not routing
	// to the leader) is what makes the post-flush miss prove the backup was wiped.
	type followerRead struct {
		key   []byte
		store *shard.Store
		node  string
	}
	var reads []followerRead
	for _, k := range keys {
		idx := shardOf(k, numShards)
		var fs *shard.Store
		var fnode string
		for _, n := range tc.nodes {
			if s := n.getShard(idx); s != nil && !s.IsLeader() {
				fs, fnode = s, n.cfg.NodeID
				break
			}
		}
		if fs == nil {
			t.Fatalf("shard %d (key %q): no follower store found; RF=%d over %d nodes should give one", idx, k, rf, nodes)
		}
		if got, err := fs.Get(k); err != nil || string(got) != "v" {
			t.Fatalf("key %q not present on follower %s (shard %d) before flush: v=%q err=%v", k, fnode, idx, got, err)
		}
		reads = append(reads, followerRead{key: k, store: fs, node: fnode})
	}

	// One flush via a single node → replicates through EVERY group's Raft log so
	// every replica (leader AND backups) applies it.
	if _, err := tc.nodes[0].Call("flush", nil); err != nil {
		t.Fatalf("flush via node 0: %v", err)
	}
	for _, n := range tc.nodes {
		waitAllApplied(t, n)
	}

	// The BACKUP replica's LOCAL cache must now miss.
	for _, r := range reads {
		_, err := r.store.Get(r.key)
		if err == nil {
			t.Fatalf("key %q still present on follower %s after flush — the backup replica did not apply the flush", r.key, r.node)
		}
		if !isNotFoundErr(err) {
			t.Fatalf("follower %s get %q after flush: unexpected err %v, want not-found", r.node, r.key, err)
		}
	}
}

// isNotFoundErr reports whether err is the cache key-miss, matched by message so it
// covers both the in-process sentinel and its stringified form across the Raft/apply
// boundary (the same approach httpapi.isKVNotFound uses).
func isNotFoundErr(err error) bool {
	return err != nil && strings.Contains(err.Error(), "not found")
}
