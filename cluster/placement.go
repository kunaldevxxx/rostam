// SPDX-License-Identifier: Apache-2.0

package cluster

import "sort"

// computePlacement deterministically assigns each of numShards data shards to a
// replica set of rf members — a sliding window over the sorted node ids — so
// shards distribute across the cluster (true storage partitioning) instead of
// every node hosting every shard.
//
// rf <= 0 or rf >= len(members) means full replication: every member hosts every
// shard. That is the default and preserves the original behavior exactly. The
// function is pure and order-independent (it sorts ids), so every node and the
// meta-Raft FSM derive identical placement from the same membership.
func computePlacement(members []Peer, numShards, rf int) [][]string {
	ids := make([]string, len(members))
	for i, p := range members {
		ids[i] = p.NodeID
	}
	sort.Strings(ids)

	out := make([][]string, numShards)
	n := len(ids)
	if n == 0 {
		return out
	}
	if rf <= 0 || rf >= n {
		for s := range out {
			out[s] = append([]string(nil), ids...)
		}
		return out
	}
	for s := 0; s < numShards; s++ {
		owners := make([]string, rf)
		for j := 0; j < rf; j++ {
			owners[j] = ids[(s+j)%n]
		}
		out[s] = owners
	}
	return out
}

// placementContains reports whether nodeID is in the owner set.
func placementContains(owners []string, nodeID string) bool {
	for _, o := range owners {
		if o == nodeID {
			return true
		}
	}
	return false
}

// peersForOwners returns the subset of peers whose NodeID is in owners,
// preserving the input order.
func peersForOwners(peers []Peer, owners []string) []Peer {
	out := make([]Peer, 0, len(owners))
	for _, p := range peers {
		if placementContains(owners, p.NodeID) {
			out = append(out, p)
		}
	}
	return out
}
