// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"encoding/binary"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/rostamlabs/rostam/shard"
)

// flushOpName is the KV op that wipes the ENTIRE keyspace. Node.Call intercepts it
// and broadcasts, because a keyless op would otherwise route to shard 0 alone.
const flushOpName = "flush"

// opFlushShardName is the INTERNAL, node-local wrapper op used to drive a flush
// into ONE named shard group on a peer. It dispatches off n.adminOps (Node.Call) —
// before op-registry routing — so the receiving node proposes the flush to exactly
// the requested group and does NOT re-broadcast.
//
// Sending flushOpName itself over the wire would re-enter Node.Call on the peer and
// fan out to every group IT hosts, which in a cluster whose nodes host
// overlapping-but-unequal shard subsets can bounce between nodes without
// terminating. The wrapper makes the remote step a leaf — the exact shape
// __register_wasm_shard__ gives the WASM-registration broadcast.
//
// It is not in authz's admin allowlist by NAME for its privilege to hold — an op
// that matches no allowlist and is absent from the ops registry already falls
// through to actionFor's deny-by-default "admin" bucket — but it IS enumerated in
// authz.adminOps regardless, so a future refactor that registers the wrapper in the
// ops registry cannot silently demote it from admin to write (the reasoning
// __register_wasm_shard__ records). Inter-node hops carry the internal service
// token, so peers pass; an external non-admin key does not.
const opFlushShardName = "__flush_shard__"

// encodeShardScopedFlush is the wire form of opFlushShardName: the target shard
// index as 4 big-endian bytes. A flush carries no other payload.
func encodeShardScopedFlush(shardIdx int) []byte {
	out := make([]byte, 4)
	binary.BigEndian.PutUint32(out, uint32(shardIdx)) //nolint:gosec // bounded by NumShards
	return out
}

// decodeShardScopedFlush reads the target shard index written by
// encodeShardScopedFlush.
func decodeShardScopedFlush(args []byte) (int, error) {
	if len(args) != 4 {
		return 0, fmt.Errorf("cluster: %s: want a 4-byte shard index, got %d bytes", opFlushShardName, len(args))
	}
	return int(binary.BigEndian.Uint32(args)), nil
}

// flushBroadcastGroupTimeout bounds ONE group's leg of the flush broadcast,
// mirroring wasmBroadcastGroupTimeout: without it a single
// unreachable-but-not-erroring peer stalls the whole flush — and the client
// connection behind it — indefinitely, because the loop is sequential over up to
// NumShards groups and forward() used an unbounded context. A group that times out
// is reported as a failure like any other, and the client retries.
const flushBroadcastGroupTimeout = 10 * time.Second

// broadcastFlush proposes a flush op into EVERY shard Raft group instead of only
// shard 0's, so one client-visible flush wipes the WHOLE keyspace rather than only
// the keys that happen to hash to group 0.
//
// WHY (a keyless op otherwise routes to shard 0 alone). flush has no routing key,
// so Node.shardIndexFor returns 0 for it and Call would propose it into group 0's
// log only, leaving groups 1..N intact. Each shard group is an INDEPENDENT Raft
// group with its OWN cache.Cache, so wiping the whole keyspace means landing a flush
// in every group's log. This is the same "replicate a mutation to every group"
// precedent broadcastWASMRegistration establishes; flush is simpler because it
// carries no payload and needs no propose-time refusals, size cap, or update gate.
//
// Each group applies the flush through its own FSM/Raft (and PB) path →
// ops.handleFlush → that group's cache.Flush(). No leader-stamping is needed: the
// cache layer captures each replica's LOCAL writeSeq floor at apply, so replicas
// converge on an empty keyspace without a coordinated clock (see cache.Cache.Flush's
// durability-ordering contract).
//
// REPLAY / RETRY. A single flush entry is idempotent WITHIN its own group — a Raft
// replay re-wipes to the same empty state and needs no dedup. Retrying the whole
// broadcast after a partial failure is how the groups that missed the first attempt
// are eventually wiped, but it is NOT a no-op on the groups that already succeeded:
// a write that landed in one of them since the first attempt is re-wiped by the
// retry. That is inherent to "wipe everything now" — not a bug — and is why the
// client op is nonReplayableOp, so Client.Call never blindly replays an ambiguous
// flush; a caller re-issues only when re-wiping intervening writes is acceptable.
//
// PARTIAL FAILURE. Every group is attempted even after one fails (a transient
// election on group 3 must not deny groups 4..N the flush); a non-empty failure set
// leaves the keyspace PARTIALLY wiped and is returned as an error so the caller can
// decide whether to complete it.
func (n *Node) broadcastFlush() ([]byte, error) {
	var failures []string
	for i := 0; i < n.cfg.NumShards; i++ {
		if err := n.proposeFlush(i); err != nil {
			failures = append(failures, fmt.Sprintf("shard %d: %v", i, err))
		}
	}
	if len(failures) > 0 {
		return nil, fmt.Errorf("cluster: flush: %d of %d shard groups failed — keyspace is now PARTIALLY wiped; re-issuing completes it but also re-wipes any writes made since to the groups that already succeeded: %s",
			len(failures), n.cfg.NumShards, strings.Join(failures, "; "))
	}
	return nil, nil
}

// proposeFlush lands a flush in shard idx's Raft log, wherever this node sits
// relative to that group (mirrors proposeWASMRegistration):
//
//   - hosted and led here: propose straight into the local store;
//   - hosted but led elsewhere (the normal case for all but one group in a
//     multi-shard cluster): shard.Store.Call answers NotLeaderError, so hop;
//   - not hosted at all (partitioned cluster): hop.
//
// The hop reuses forwardTimeout, whose per-owner client follows NotLeader to the
// group's leader — but carries the shard-scoped wrapper op so the peer handles this
// ONE group and does not re-broadcast (see opFlushShardName).
//
// A NotLeaderError must not escape to the client here: a broadcast spans groups with
// DIFFERENT leaders, so "retry at node X" is not a hint any single node can satisfy,
// and following it would bounce the client between nodes that each lead a different
// subset. The text is preserved for diagnostics; the type is not.
func (n *Node) proposeFlush(idx int) error {
	var hostedErr error
	if s := n.getShard(idx); s != nil {
		_, err := n.callHostedShard(s, flushOpName, nil)
		if err == nil {
			return nil
		}
		var nle *shard.NotLeaderError
		if !errors.As(err, &nle) {
			// A genuine propose/apply failure for this group (not a routing miss);
			// another owner would fail the same way.
			return err
		}
		hostedErr = err
	}
	if _, err := n.forwardTimeout(idx, opFlushShardName, encodeShardScopedFlush(idx), flushBroadcastGroupTimeout); err != nil {
		if hostedErr != nil {
			return fmt.Errorf("%v; leader hop: %v", hostedErr, err)
		}
		return err
	}
	return nil
}

// handleFlushShard is the node-local handler for opFlushShardName. It proposes the
// flush to the ONE group named in the payload — never to any other group, and never
// onward to another node — so the broadcast fan-out stays flat. Returning
// ErrNoShardOwner when this node does not host the group lets the sender's forward()
// loop move on to the next owner.
func (n *Node) handleFlushShard(args []byte) ([]byte, error) {
	idx, err := decodeShardScopedFlush(args)
	if err != nil {
		return nil, err
	}
	if idx < 0 || idx >= n.cfg.NumShards {
		return nil, fmt.Errorf("cluster: %s: shard %d out of range [0,%d)", opFlushShardName, idx, n.cfg.NumShards)
	}
	s := n.getShard(idx)
	if s == nil {
		return nil, ErrNoShardOwner
	}
	return n.callHostedShard(s, flushOpName, nil)
}
