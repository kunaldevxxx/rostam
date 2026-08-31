// SPDX-License-Identifier: Apache-2.0
//go:build linux || windows

package cache

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// Flush() tests — Stage 1 cache-layer core (global KV wipe by index swap + a
// per-shard durability watermark). Mmap tests need the platform mmap build (hence
// the build tag, matching warm_restart_test.go); the heap cases run there too.

// flushCfg is a persistent (mmap when dir != "") reject-writes cache with a small
// page budget so the handful of tiny keys these tests write keep occupancy well
// below mmapCompactMinOccupancy — cold compaction never runs, so what keeps flushed
// keys gone is the sidecar watermark, not the reclamation. A pinned NowFn keeps TTL
// out of the picture entirely.
func flushCfg(dir string, shards int) Config {
	cfg := DefaultConfig()
	cfg.NumShards = shards
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 8 << 20 // 8 pages
	cfg.AtCapPolicy = PolicyRejectWrites
	cfg.TTLSweepIntervalMs = 0
	cfg.DataDir = dir
	cfg.NowFn = func() uint64 { return 5_000_000 }
	return cfg
}

// liveCount returns the number of live entries visible through Iterate — the true
// logical keyset size, which a flush must drive to 0.
func liveCount(c *Cache) int {
	n := 0
	c.Iterate(func(_, _ []byte, _ uint64) bool { n++; return true })
	return n
}

func flushKey(i int) []byte   { return []byte(fmt.Sprintf("key-%06d", i)) }
func flushValue(i int) []byte { return []byte(fmt.Sprintf("val-%06d", i)) }

// TestFlushEmptiesCache: N keys across many internal shards, Flush, every Get
// misses and the live keyset (Iterate) is 0. Runs for both heap and mmap.
func TestFlushEmptiesCache(t *testing.T) {
	const n = 500
	for _, tc := range []struct {
		name string
		dir  string
	}{
		{"heap", ""},
		{"mmap", t.TempDir()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := flushCfg(tc.dir, 16)
			c, err := New(cfg)
			if err != nil {
				t.Fatal(err)
			}
			defer c.Close()
			for i := 0; i < n; i++ {
				if err := c.Put(flushKey(i), flushValue(i), 0); err != nil {
					t.Fatalf("Put %d: %v", i, err)
				}
			}
			if got := liveCount(c); got != n {
				t.Fatalf("pre-flush live count = %d, want %d", got, n)
			}
			if err := c.Flush(); err != nil {
				t.Fatalf("Flush: %v", err)
			}
			if got := liveCount(c); got != 0 {
				t.Fatalf("post-flush live count = %d, want 0", got)
			}
			for i := 0; i < n; i++ {
				if _, err := c.Get(flushKey(i)); err != ErrNotFound {
					t.Fatalf("Get %d after flush: err=%v, want ErrNotFound", i, err)
				}
			}
			// Cumulative counters are NOT reset by flush: Puts is a lifetime total.
			if st := c.Stats(); st.Puts != n {
				t.Fatalf("Stats.Puts = %d after flush, want %d (cumulative, not a live gauge)", st.Puts, n)
			}
		})
	}
}

// TestFlushPostFlushWritesWork: writes after a flush store and read back, for both
// modes.
func TestFlushPostFlushWritesWork(t *testing.T) {
	for _, tc := range []struct {
		name string
		dir  string
	}{
		{"heap", ""},
		{"mmap", t.TempDir()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, err := New(flushCfg(tc.dir, 4))
			if err != nil {
				t.Fatal(err)
			}
			defer c.Close()
			for i := 0; i < 50; i++ {
				if err := c.Put(flushKey(i), flushValue(i), 0); err != nil {
					t.Fatal(err)
				}
			}
			if err := c.Flush(); err != nil {
				t.Fatalf("Flush: %v", err)
			}
			for i := 100; i < 150; i++ {
				if err := c.Put(flushKey(i), flushValue(i), 0); err != nil {
					t.Fatalf("post-flush Put %d: %v", i, err)
				}
			}
			for i := 100; i < 150; i++ {
				v, err := c.Get(flushKey(i))
				if err != nil || string(v) != string(flushValue(i)) {
					t.Fatalf("post-flush Get %d: %q err=%v", i, v, err)
				}
			}
			if got := liveCount(c); got != 50 {
				t.Fatalf("live count = %d, want 50 (only post-flush keys)", got)
			}
		})
	}
}

// TestFlushMmapRestartNoResurrection: put, Flush, Close, reopen → all pre-flush
// keys are gone. This is the core durability property: the sidecar makes the wipe
// survive the warm-restart rebuild.
func TestFlushMmapRestartNoResurrection(t *testing.T) {
	dir := t.TempDir()
	cfg := flushCfg(dir, 1)
	const n = 100

	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		if err := c.Put(flushKey(i), flushValue(i), 0); err != nil {
			t.Fatal(err)
		}
	}
	if err := c.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}

	c2, err := New(cfg)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer c2.Close()
	if got := liveCount(c2); got != 0 {
		t.Fatalf("after restart live count = %d, want 0 (flushed data resurrected)", got)
	}
	for i := 0; i < n; i++ {
		if _, err := c2.Get(flushKey(i)); err != ErrNotFound {
			t.Fatalf("restart Get %d: err=%v, want ErrNotFound — flushed key came back", i, err)
		}
	}
}

// TestFlushThenWriteSurvivesRestart: put A, Flush, put B, Close, reopen → A gone,
// B present. Proves the watermark wipes only through the floor and post-flush
// writes (seq > floor) survive.
func TestFlushThenWriteSurvivesRestart(t *testing.T) {
	dir := t.TempDir()
	cfg := flushCfg(dir, 1)
	keyA, valA := []byte("A-before-flush"), []byte("value-A")
	keyB, valB := []byte("B-after-flush"), []byte("value-B")

	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Put(keyA, valA, 0); err != nil {
		t.Fatal(err)
	}
	if err := c.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if err := c.Put(keyB, valB, 0); err != nil {
		t.Fatal(err)
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}

	c2, err := New(cfg)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer c2.Close()
	if _, err := c2.Get(keyA); err != ErrNotFound {
		t.Fatalf("restart: A (pre-flush) came back (err=%v), want ErrNotFound", err)
	}
	if v, err := c2.Get(keyB); err != nil || string(v) != string(valB) {
		t.Fatalf("restart: B (post-flush) = %q err=%v, want %q", v, err, valB)
	}
}

// TestFlushWriteSeqRestoreHazard is the most important durability test. Without the
// writeSeq floor-restore in newShard, a post-flush write on an emptied shard would
// get a low sequence (1..floor) that the NEXT restart's rebuild wrongly classifies
// as flushed and skips — silently losing a committed write on the SECOND restart.
//
// put keys → Flush (shard now empty) → Close → reopen → put a NEW key → Close →
// reopen again → the new key MUST still be present.
func TestFlushWriteSeqRestoreHazard(t *testing.T) {
	dir := t.TempDir()
	cfg := flushCfg(dir, 1)
	const n = 100

	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		if err := c.Put(flushKey(i), flushValue(i), 0); err != nil {
			t.Fatal(err)
		}
	}
	if err := c.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}

	// First restart: shard rebuilds EMPTY (every entry skipped). writeSeq must be
	// lifted to the flush floor here, or the write below gets a flushed-range seq.
	c2, err := New(cfg)
	if err != nil {
		t.Fatalf("reopen 1: %v", err)
	}
	newKey, newVal := []byte("post-flush-new-key"), []byte("must-survive")
	if err := c2.Put(newKey, newVal, 0); err != nil {
		t.Fatal(err)
	}
	if err := c2.Close(); err != nil {
		t.Fatal(err)
	}

	// Second restart: the new key must NOT have been skipped as "flushed".
	c3, err := New(cfg)
	if err != nil {
		t.Fatalf("reopen 2: %v", err)
	}
	defer c3.Close()
	if v, err := c3.Get(newKey); err != nil || string(v) != string(newVal) {
		t.Fatalf("second restart: new post-flush key = %q err=%v, want %q — "+
			"writeSeq was not lifted to the flush floor, so the low-seq write was "+
			"wrongly skipped as flushed", v, err, newVal)
	}
	if got := liveCount(c3); got != 1 {
		t.Fatalf("second restart live count = %d, want 1 (only the new key)", got)
	}
}

// TestFlushHeapNeedsNoSidecar: heap flush empties the cache and writes no sidecar
// (heap has no rebuild path, so nothing to persist). No dir ⇒ nothing on disk.
func TestFlushHeapNeedsNoSidecar(t *testing.T) {
	c, err := New(flushCfg("", 4))
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	for i := 0; i < 50; i++ {
		if err := c.Put(flushKey(i), flushValue(i), 0); err != nil {
			t.Fatal(err)
		}
	}
	if err := c.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if got := liveCount(c); got != 0 {
		t.Fatalf("heap flush live count = %d, want 0", got)
	}
	// Heap shards keep no dataDir, so there is provably no sidecar to write.
	for _, s := range c.shards {
		if s.dataDir != "" {
			t.Fatalf("heap shard has a dataDir %q; expected none", s.dataDir)
		}
	}
}

// TestFlushSidecarCorruptionTolerance: a corrupt sidecar falls back to floor 0 —
// pre-flush data may resurrect, but the open must NOT crash or fail.
func TestFlushSidecarCorruptionTolerance(t *testing.T) {
	dir := t.TempDir()
	cfg := flushCfg(dir, 1)

	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Put([]byte("k"), []byte("v"), 0); err != nil {
		t.Fatal(err)
	}
	if err := c.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}

	sidecar := filepath.Join(dir, "shard-0000", flushSidecarName)
	b, err := os.ReadFile(sidecar)
	if err != nil {
		t.Fatalf("sidecar should exist after a flush: %v", err)
	}
	if len(b) != flushSidecarSize {
		t.Fatalf("sidecar size = %d, want %d", len(b), flushSidecarSize)
	}
	b[flushSidecarCRCOff] ^= 0xFF // corrupt the CRC
	if err := os.WriteFile(sidecar, b, 0o640); err != nil {
		t.Fatal(err)
	}

	// Must open cleanly (fail-open to floor 0), not panic or error.
	c2, err := New(cfg)
	if err != nil {
		t.Fatalf("reopen with corrupt sidecar must not fail: %v", err)
	}
	defer c2.Close()
	// floor 0 ⇒ the pre-flush key is re-indexed (resurrection is the documented,
	// accepted cost of a corrupt watermark). We only require no crash.
	if _, err := c2.Get([]byte("k")); err != nil && err != ErrNotFound {
		t.Fatalf("unexpected Get error after corrupt-sidecar reopen: %v", err)
	}
}

// TestFlushIdempotent: two flushes in a row are fine and leave the cache empty, for
// both modes.
func TestFlushIdempotent(t *testing.T) {
	for _, tc := range []struct {
		name string
		dir  string
	}{
		{"heap", ""},
		{"mmap", t.TempDir()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, err := New(flushCfg(tc.dir, 2))
			if err != nil {
				t.Fatal(err)
			}
			defer c.Close()
			for i := 0; i < 30; i++ {
				if err := c.Put(flushKey(i), flushValue(i), 0); err != nil {
					t.Fatal(err)
				}
			}
			if err := c.Flush(); err != nil {
				t.Fatalf("Flush 1: %v", err)
			}
			if err := c.Flush(); err != nil {
				t.Fatalf("Flush 2: %v", err)
			}
			if got := liveCount(c); got != 0 {
				t.Fatalf("after double flush live count = %d, want 0", got)
			}
		})
	}
}
