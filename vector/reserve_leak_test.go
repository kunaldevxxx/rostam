// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"math/rand"
	"testing"
)

// THE OWNERSHIP RULE, MADE TESTABLE.
//
// A slab reservation is OS memory the Go GC cannot see, and it carries no
// finalizer on purpose (see slabReservation's doc: a finalizer could fire while
// a slice derived from the region is still in flight). So it must be released
// explicitly, and a missed release is invisible to everything that normally
// catches bugs — the GC, -race, and every assertion about program output. It
// leaks silently, forever.
//
// liveSlabReservations exists to close exactly that hole. These tests build an
// index whose slabs really are reservation-backed, put it through the paths that
// discard or REPLACE those slabs, and require the live count to return to where
// it started. Before the fix, restoring into a pre-built index left the old
// arena's reservations mapped with nothing referencing them.

// reservationBalance runs fn and returns the net change in live reservations.
// Zero is the only acceptable answer for any path that ends with the index
// closed.
func reservationBalance(t *testing.T, fn func()) int64 {
	t.Helper()
	before := liveSlabReservations.Load()
	fn()
	return liveSlabReservations.Load() - before
}

// buildReserved fills h with n vectors starting at id `base` and fails the test
// unless both big slabs ended up on reservations — otherwise the leak test would
// pass vacuously. The id base lets a caller re-grow an index that already holds a
// restored id range without colliding with it.
func buildReserved(t *testing.T, h *hnsw, base uint64, n, dim int, seed int64) [][]float32 {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	out := make([][]float32, n)
	for i := 0; i < n; i++ {
		v := make([]float32, dim)
		for d := range v {
			v[d] = rng.Float32()
		}
		out[i] = v
		if _, _, err := h.Insert(base+uint64(i), v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d: %v", base+uint64(i), err)
		}
	}
	if h.arena.vecsRes == nil || h.graphRes == nil {
		t.Fatalf("slabs never became reservation-backed (vec=%v graph=%v) — leak test would be vacuous",
			h.arena.vecsRes != nil, h.graphRes != nil)
	}
	return out
}

// TestRestoreIntoPrebuiltIndexReleasesReservations is the regression. readSnapshot
// replaces h.arena wholesale; before the fix the outgoing arena's reservations
// were simply dropped on the floor. Restoring into a PRE-BUILT index is the shape
// NamedCollection.Restore and MultiVectorIndex.restore document, so this is one
// caller away from an unbounded per-restore leak.
func TestRestoreIntoPrebuiltIndexReleasesReservations(t *testing.T) {
	withSmallReservations(t, 64<<10, 64<<10, 64)
	const n, dim = 4000, 32
	cfg := Config{Dim: dim, Metric: L2, M: 8, EfConstruction: 32, EfSearch: 16, Seed: 5}

	src, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = src.Close() }()
	vecs := buildReserved(t, src, 0, n, dim, 5)
	var snap bytes.Buffer
	if err := src.Snapshot(&snap); err != nil {
		t.Fatal(err)
	}

	// The target is NOT fresh: it is already carrying its own reservation-backed
	// slabs, which the restore is about to replace.
	dst, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = dst.Close() }()
	buildReserved(t, dst, 0, n, dim, 99)

	oldArena, oldVecsRes := dst.arena, dst.arena.vecsRes
	delta := reservationBalance(t, func() {
		if rerr := dst.Restore(bytes.NewReader(snap.Bytes())); rerr != nil {
			t.Fatal(rerr)
		}
	})
	if dst.arena == oldArena {
		t.Fatal("restore did not replace the arena — this test is no longer covering the replacement path")
	}
	if oldVecsRes == nil {
		t.Fatal("pre-restore arena had no reservation to leak")
	}
	// The restored arena is heap-resident (readSnapshot reads the vecs block into
	// a fresh slice), so the net effect must be exactly "one reservation freed" —
	// never zero, which is what a leak looks like.
	if delta >= 0 {
		t.Errorf("restore into a pre-built index leaked: live reservations moved by %+d, want negative", delta)
	}
	if oldVecsRes.mapped() {
		t.Error("the replaced arena's vector reservation was never released")
	}

	// And the restore is still correct.
	for i := 0; i < n; i += 37 {
		got, _, _, _, _, ok := dst.Get(uint64(i))
		if !ok {
			t.Fatalf("id %d missing after restore", i)
		}
		for d := range got {
			if got[d] != vecs[i][d] {
				t.Fatalf("id %d dim %d after restore: got %v want %v", i, d, got[d], vecs[i][d])
			}
		}
	}
}

// TestRepeatedRestoreDoesNotAccumulateReservations is the same hazard stated the
// way it would actually bite: a long-lived index restored over and over (the
// cluster snapshot-install path) must not climb. One restore leaking is a bug;
// N restores leaking N is an outage.
func TestRepeatedRestoreDoesNotAccumulateReservations(t *testing.T) {
	withSmallReservations(t, 64<<10, 64<<10, 64)
	const n, dim = 3000, 32
	cfg := Config{Dim: dim, Metric: L2, M: 8, EfConstruction: 32, EfSearch: 16, Seed: 5}

	src, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = src.Close() }()
	buildReserved(t, src, 0, n, dim, 7)
	var snap bytes.Buffer
	if err := src.Snapshot(&snap); err != nil {
		t.Fatal(err)
	}

	dst, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = dst.Close() }()

	var peak int64
	before := liveSlabReservations.Load()
	for round := 0; round < 5; round++ {
		// Re-grow onto a reservation between restores, so each round really does
		// hand the restore something to release.
		// Fresh id range each round: the restore has just repopulated ids [0,n).
		buildReserved(t, dst, uint64(n)*uint64(round+1), n, dim, int64(round)+100)
		if rerr := dst.Restore(bytes.NewReader(snap.Bytes())); rerr != nil {
			t.Fatalf("round %d: %v", round, rerr)
		}
		if live := liveSlabReservations.Load() - before; live > peak {
			peak = live
		}
	}
	// Whatever a single round holds, five rounds must not hold five times it.
	if peak > 3 {
		t.Errorf("live reservations peaked at %+d above baseline across 5 restores — they are accumulating", peak)
	}
}

// TestCloseReleasesEveryReservation pins the ordinary lifecycle for all three
// reserved slabs at once, including the codes slab (which only a quantized index
// has) and the mmap-backed variants, where Close must release the reservation
// AND close the file.
func TestCloseReleasesEveryReservation(t *testing.T) {
	withSmallReservations(t, 64<<10, 64<<10, 64)
	const n, dim = 4000, 32

	for _, tc := range []struct {
		name string
		cfg  func(*testing.T) Config
		// needsFileReservations marks the case whose slabs are mmap-backed, and
		// so only reservation-backed where a file-backed slab can be (Linux).
		// Elsewhere those slabs never take a reservation, leaving nothing for
		// this test's balance to say anything about.
		needsFileReservations bool
	}{
		{name: "heap", cfg: func(*testing.T) Config {
			return Config{Dim: dim, Metric: L2, M: 8, EfConstruction: 32, EfSearch: 16, Seed: 5}
		}},
		// The codes slab is ALWAYS anonymous, so it takes a reservation
		// wherever anonymous ones exist - including where file-backed ones do
		// not. Without this case the mmap one below is the only quantized one,
		// and skipping it would leave Close's release of the codes reservation
		// unverified on exactly the platform this branch adds.
		{name: "heap+codes", cfg: func(*testing.T) Config {
			return Config{
				Dim: dim, Metric: L2, M: 8, EfConstruction: 32, EfSearch: 16, Seed: 5,
				Quant: QuantSQ8,
			}
		}},
		{name: "mmap+codes", cfg: mmapGrowConfig(dim, 5), needsFileReservations: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.needsFileReservations {
				requireFileBackedReservations(t)
			}
			delta := reservationBalance(t, func() {
				h, err := newHNSW(tc.cfg(t))
				if err != nil {
					t.Fatal(err)
				}
				buildReserved(t, h, 0, n, dim, 5)
				if err := h.Close(); err != nil {
					t.Fatalf("close: %v", err)
				}
				// Idempotent: a second Close must not double-release.
				if err := h.Close(); err != nil {
					t.Fatalf("second close: %v", err)
				}
			})
			if delta != 0 {
				t.Errorf("live reservations moved by %+d across build+close; want 0", delta)
			}
		})
	}
}

// TestDropVecsReleasesReservation covers the PQ-only path, where the float
// vectors are discarded outright rather than replaced. Nil-ing an off-heap
// header frees nothing, so dropVecs must hand the range back explicitly.
func TestDropVecsReleasesReservation(t *testing.T) {
	withSmallReservations(t, 64<<10, 64<<10, 64)
	a := newArena(32, 0)
	before := liveSlabReservations.Load()
	vec := make([]float32, 32)
	for i := 0; i < 4000; i++ {
		for d := range vec {
			vec[d] = float32(i + d)
		}
		if _, err := a.Insert(uint64(i), vec); err != nil {
			t.Fatal(err)
		}
	}
	if a.vecsRes == nil {
		t.Fatal("arena never took a reservation — test would be vacuous")
	}
	a.dropVecs()
	if a.vecsRes != nil {
		t.Error("dropVecs left vecsRes set")
	}
	if got := liveSlabReservations.Load() - before; got != 0 {
		t.Errorf("dropVecs left %+d reservations live; want 0", got)
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
}

// TestReinitializingCodesReleasesReservation pins the precondition documented on
// growCodes: the (re)initializers assign a.codes a fresh backing array directly,
// and must release any reservation first — otherwise the reservation is not just
// leaked but left as the destination the next growCodes would commit into, which
// is silent data loss rather than a crash.
func TestReinitializingCodesReleasesReservation(t *testing.T) {
	withSmallReservations(t, 64<<10, 64<<10, 64)
	const dim = 32
	a := newArena(dim, 0)
	a.setQuant(newQuantizer(QuantSQ8, dim, 0, 0, 0, 0, L2))

	vec := make([]float32, dim)
	for i := 0; i < 6000; i++ {
		for d := range vec {
			vec[d] = float32(i + d)
		}
		if _, err := a.Insert(uint64(i), vec); err != nil {
			t.Fatal(err)
		}
	}
	if a.codesRes == nil {
		t.Fatal("codes slab never took a reservation — test would be vacuous")
	}

	// Deltas across the re-init call itself, not against a pre-insert baseline:
	// the VECTOR slab is reservation-backed by now too and legitimately stays
	// live, so an absolute count would be measuring the wrong thing.
	reinit := func(name string, fn func()) {
		t.Helper()
		before := liveSlabReservations.Load()
		fn()
		if a.codesRes != nil {
			t.Errorf("%s left a stale codes reservation", name)
		}
		if got := liveSlabReservations.Load() - before; got != -1 {
			t.Errorf("%s moved live reservations by %+d; want -1 (the codes slab released)", name, got)
		}
	}
	reinit("setQuant", func() { a.setQuant(newQuantizer(QuantSQ8, dim, 0, 0, 0, 0, L2)) })

	// Re-grow so enableCodes has something of its own to release. setQuant left a
	// heap slice with a non-trivial cap, so the request has to clear BOTH that cap
	// and the reservation threshold to actually take a new reservation.
	a.growCodes(2*cap(a.codes) + int(slabReserveThreshold))
	if a.codesRes == nil {
		t.Fatal("codes slab did not re-take a reservation")
	}
	reinit("enableCodes", func() { a.enableCodes(dim) })

	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
}
