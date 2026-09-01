// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"os"
	"testing"
	"time"

	hraft "github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/raft/mux"
	"github.com/rostamlabs/rostam/shard"
)

// metaTransport builds the meta group's transport the way newMultiNode's default
// (mux) path does, so these tests exercise the same wiring.
func metaTransport(sl *mux.StreamLayer) hraft.Transport {
	return hraft.NewNetworkTransport(sl.For(metaGroupID), 3, 10*time.Second, os.Stderr)
}

func TestMetaRaftStartsAndElectsLeader(t *testing.T) {
	sl, err := mux.New("127.0.0.1:0", []uint32{metaGroupID}, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = sl.Close() })

	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	self := Peer{NodeID: "node1", RaftAddr: sl.Addr().String(), ServerAddr: "127.0.0.1:0"}
	cfg := Config{
		NodeID:    "node1",
		DataDir:   t.TempDir(),
		NumShards: 4,
		Bootstrap: true,
		ShardCfg: shard.Config{
			RaftHeartbeatMs: 50,
			RaftElectionMs:  100,
			NoSync:          true,
		},
		Ops:      reg,
		Peers:    []Peer{self},
		RaftAddr: sl.Addr().String(),
	}

	mr, err := startMetaRaft(cfg, metaTransport(sl))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = mr.Close() })

	if err := waitForAnyLeader(mr.Raft, 5*time.Second); err != nil {
		t.Fatal(err)
	}
	if err := mr.ApplySetMembersIfLeader(cfg.Peers, cfg.NumShards, 0, 0, 5*time.Second); err != nil {
		t.Fatal(err)
	}
	st := mr.FSM.State()
	if st.NumShards != 4 {
		t.Errorf("NumShards = %d, want 4", st.NumShards)
	}
	if len(st.Members) != 1 || st.Members[0].NodeID != "node1" {
		t.Errorf("Members = %+v, want [node1]", st.Members)
	}
	if len(st.Placement) != 4 {
		t.Errorf("Placement len = %d, want 4", len(st.Placement))
	}

	// Second apply should be a no-op (idempotent), no error.
	if err := mr.ApplySetMembersIfLeader(cfg.Peers, cfg.NumShards, 0, 0, 5*time.Second); err != nil {
		t.Errorf("idempotent re-apply: %v", err)
	}
}

func TestMetaRaftApplySetCatalogEntry(t *testing.T) {
	sl, err := mux.New("127.0.0.1:0", []uint32{metaGroupID}, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = sl.Close() })

	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	self := Peer{NodeID: "node1", RaftAddr: sl.Addr().String(), ServerAddr: "127.0.0.1:0"}
	cfg := Config{
		NodeID:    "node1",
		DataDir:   t.TempDir(),
		NumShards: 4,
		Bootstrap: true,
		ShardCfg: shard.Config{
			RaftHeartbeatMs: 50,
			RaftElectionMs:  100,
			NoSync:          true,
		},
		Ops:      reg,
		Peers:    []Peer{self},
		RaftAddr: sl.Addr().String(),
	}

	mr, err := startMetaRaft(cfg, metaTransport(sl))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = mr.Close() })

	if err := waitForAnyLeader(mr.Raft, 5*time.Second); err != nil {
		t.Fatal(err)
	}

	if err := mr.ApplySetCatalogEntry("default/docs", 8, 0, 5*time.Second); err != nil {
		t.Fatalf("ApplySetCatalogEntry: %v", err)
	}
	if p := mr.FSM.State().Catalog["default/docs"]; p != 8 {
		t.Fatalf("catalog docs = %d, want 8", p)
	}
	// Defensive: zero/invalid partition count is rejected without committing.
	if err := mr.ApplySetCatalogEntry("default/bad", 0, 0, 5*time.Second); err == nil {
		t.Fatal("ApplySetCatalogEntry with p=0 should error")
	}
	if _, ok := mr.FSM.State().Catalog["default/bad"]; ok {
		t.Fatal("p=0 must not be committed to the catalog")
	}
}


func TestApplySetMembersIfLeaderRFChangeIsNotIdempotent(t *testing.T) {
	sl, err := mux.New("127.0.0.1:0", []uint32{metaGroupID}, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = sl.Close() })

	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	self := Peer{NodeID: "node1", RaftAddr: sl.Addr().String(), ServerAddr: "127.0.0.1:0"}
	cfg := Config{
		NodeID:    "node1",
		DataDir:   t.TempDir(),
		NumShards: 4,
		Bootstrap: true,
		ShardCfg: shard.Config{
			RaftHeartbeatMs: 50,
			RaftElectionMs:  100,
			NoSync:          true,
		},
		Ops:      reg,
		Peers:    []Peer{self},
		RaftAddr: sl.Addr().String(),
	}

	mr, err := startMetaRaft(cfg, metaTransport(sl))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = mr.Close() })

	if err := waitForAnyLeader(mr.Raft, 5*time.Second); err != nil {
		t.Fatal(err)
	}

	// First apply: RF=0 (full replication).
	if err := mr.ApplySetMembersIfLeader(cfg.Peers, cfg.NumShards, 0, 0, 5*time.Second); err != nil {
		t.Fatalf("first apply (RF=0): %v", err)
	}
	st := mr.FSM.State()
	if !st.ReplicationFactorSet || st.ReplicationFactor != 0 {
		t.Fatalf("after RF=0: got RF=%d set=%v, want RF=0 set=true", st.ReplicationFactor, st.ReplicationFactorSet)
	}

	// Same members + same NumShards, only RF changes: the guard must fall through
	// and commit the new entry (regression for #71 — the old guard ignored RF).
	if err := mr.ApplySetMembersIfLeader(cfg.Peers, cfg.NumShards, 1, 0, 5*time.Second); err != nil {
		t.Fatalf("second apply (RF=1): %v", err)
	}
	st = mr.FSM.State()
	if st.ReplicationFactor != 1 || !st.ReplicationFactorSet {
		t.Fatalf("after RF=1: got RF=%d set=%v, want RF=1 set=true — guard may have short-circuited", st.ReplicationFactor, st.ReplicationFactorSet)
	}

	// Third apply with same RF=1: must be idempotent (no-op).
	if err := mr.ApplySetMembersIfLeader(cfg.Peers, cfg.NumShards, 1, 0, 5*time.Second); err != nil {
		t.Fatalf("idempotent re-apply (RF=1): %v", err)
	}
	if st2 := mr.FSM.State(); st2.ReplicationFactor != 1 {
		t.Fatalf("after idempotent re-apply: RF=%d, want 1", st2.ReplicationFactor)
	}
}
