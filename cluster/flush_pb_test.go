// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"fmt"
	"testing"
	"time"
)

// TestPBFlushBroadcastWipesBackupReplica proves the flush replicates through the
// PRIMARY-BACKUP apply path, not only Raft. The other flush tests use newTestCluster
// (the Raft harness); this uses newPBTestCluster (ReplicationMode="pb"), seeds keys
// via the primary, flushes, waits for apply, then reads a BACKUP's LOCAL cache
// (node.getShard(idx).Get) and asserts it misses — which can only mean the flush
// reached the backup through PB replication and its apply path wiped it.
//
// One shard keeps the topology unambiguous: a single primary and two backups, so the
// flush issued at the primary proposes locally and replicates to both backups.
func TestPBFlushBroadcastWipesBackupReplica(t *testing.T) {
	const nodes, numShards, minISR = 3, 1, 2
	tc := newPBTestCluster(t, nodes, numShards, minISR)

	// Seed a spread of keys (all route to shard 0's primary here).
	keys := make([][]byte, 0, 12)
	for i := 0; i < 12; i++ {
		k := []byte(fmt.Sprintf("pbflush-%03d", i))
		pbPutKey(t, tc, numShards, k, []byte("v"))
		keys = append(keys, k)
	}

	primaryIdx := pbPrimaryIdx(t, tc, 0)

	// Confirm every key is present on a BACKUP's local cache before the flush —
	// reading the backup directly (not routing to the primary) is what makes the
	// post-flush miss prove the backup itself applied the wipe.
	backupIdx := -1
	for i := range tc.nodes {
		if i != primaryIdx {
			backupIdx = i
			break
		}
	}
	if backupIdx < 0 {
		t.Fatalf("no backup node found (primary=%d of %d nodes)", primaryIdx, nodes)
	}
	backup := tc.nodes[backupIdx].getShard(0)
	if backup == nil {
		t.Fatalf("backup node %d does not host shard 0", backupIdx)
	}
	for _, k := range keys {
		if v, err := backup.Get(k); err != nil || string(v) != "v" {
			t.Fatalf("key %q not present on backup n%d before flush: v=%q err=%v", k, backupIdx+1, v, err)
		}
	}

	// One flush issued at the primary → proposes into shard 0 and replicates to the
	// PB backups.
	if _, err := tc.nodes[primaryIdx].Call("flush", nil); err != nil {
		t.Fatalf("flush via primary n%d: %v", primaryIdx+1, err)
	}

	// The BACKUP's LOCAL cache must now miss every key. Poll: PB apply on the backup
	// is asynchronous to the primary's flush return.
	deadline := time.Now().Add(10 * time.Second)
	for _, k := range keys {
		var missErr error
		gone := false
		for time.Now().Before(deadline) {
			_, err := backup.Get(k)
			if err != nil {
				missErr, gone = err, true
				break
			}
			// Still present: the backup has not applied the flush yet.
			time.Sleep(25 * time.Millisecond)
		}
		if !gone {
			t.Fatalf("key %q still present on backup n%d after flush — the PB backup did not apply the flush", k, backupIdx+1)
		}
		if !isNotFoundErr(missErr) {
			t.Fatalf("backup n%d get %q after flush: unexpected err %v, want not-found", backupIdx+1, k, missErr)
		}
	}

	// Post-flush write still works through the primary (store is live, not poisoned).
	pbPutKey(t, tc, numShards, []byte("pbflush-post"), []byte("w"))
}
