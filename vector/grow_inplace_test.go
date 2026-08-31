// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"math"
	"math/rand"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"unsafe"
)

// IN-PLACE SLAB GROWTH — the executable form of reserve.go's contract.
//
// grow_race_test.go pins the SAFETY invariant: growth is invisible to a
// concurrent reader because it happens under the write lock. These tests pin the
// COST property that replaced the stall: once a slab is backed by an
// address-space reservation, growing it neither copies nor moves it. The
// observable form of "neither copies nor moves" is the base address — assert the
// pointer, not a timing.
//
// Every test here shrinks slabReserveThreshold (and sometimes the reservation
// size itself) so a few thousand small vectors exercise the same code path a
// 1M x 768d index takes at 3 GB. Without that, the reservation path only engages
// past 32 MiB and no unit test could reach it.

// withSmallReservations shrinks the reservation knobs for the duration of a test
// and restores them after. threshold is the committed size at which a slab
// switches onto a reservation; factor/min size the reservation itself.
//
// It also SKIPS the test where the platform has no reservations at all (non-Linux
// or 32-bit Linux — see reserve_linux.go's build tag). Everything in this file
// and in reserve_leak_test.go asserts properties that only exist when a slab is
// reservation-backed, so on those platforms the honest outcome is "not
// applicable", not a failure: the fallback growth path they would exercise
// instead is already covered by grow_race_test.go.
func withSmallReservations(t *testing.T, threshold, minReserve, factor int64) {
	t.Helper()
	if !slabReservationsSupported {
		t.Skip("address-space reservations unavailable on this platform; slabs use the copy/remap growth path")
	}
	oldT, oldM, oldF := slabReserveThreshold, slabReserveMin, slabReserveFactor
	slabReserveThreshold, slabReserveMin, slabReserveFactor = threshold, minReserve, factor
	t.Cleanup(func() {
		slabReserveThreshold, slabReserveMin, slabReserveFactor = oldT, oldM, oldF
	})
}

// requireFileBackedReservations skips a test whose subject is an MMAP-BACKED
// slab growing in place. Where only anonymous reservations exist (Windows), an
// mmap-backed slab falls back to copy/remap, so a stable base — and everything
// downstream of it — is not a property the platform has. The heap variants of
// these same tests still run there and cover the reservation machinery itself.
func requireFileBackedReservations(t *testing.T) {
	t.Helper()
	if !fileBackedSlabReservationsSupported {
		t.Skip("file-backed slabs use the copy/remap growth path on this platform")
	}
}

func vecsBase(h *hnsw) unsafe.Pointer {
	if len(h.arena.vecs) == 0 {
		return nil
	}
	return unsafe.Pointer(&h.arena.vecs[0])
}

func level0Base(h *hnsw) unsafe.Pointer {
	if len(h.level0) == 0 {
		return nil
	}
	return unsafe.Pointer(&h.level0[0])
}

// growthWitness records, across a run of inserts, how many times each slab's
// capacity grew and how many DISTINCT base addresses it took while it was
// reservation-backed. The reservation contract is: many capacity growths, one
// base address.
type growthWitness struct {
	vecGrowths, vecBases                     int
	graphGrowths, graphBases                 int
	codeGrowths, codeBases                   int
	vecReserved, graphReserved, codeReserved bool
}

type growthCursor struct {
	vecCap, graphCap, codeCap    int
	vecBase, graphBase, codeBase unsafe.Pointer
}

func codesBase(h *hnsw) unsafe.Pointer {
	if len(h.arena.codes) == 0 {
		return nil
	}
	return unsafe.Pointer(&h.arena.codes[0])
}

func (w *growthWitness) observe(h *hnsw, c *growthCursor) {
	if n := cap(h.arena.vecs); n != c.vecCap {
		c.vecCap = n
		w.vecGrowths++
	}
	if n := cap(h.level0); n != c.graphCap {
		c.graphCap = n
		w.graphGrowths++
	}
	if n := cap(h.arena.codes); n != c.codeCap {
		c.codeCap = n
		w.codeGrowths++
	}
	// Base addresses are only counted once the slab is reservation-backed; before
	// that the legacy path is expected to move it.
	if h.arena.vecsRes != nil {
		w.vecReserved = true
		if b := vecsBase(h); b != c.vecBase {
			c.vecBase = b
			w.vecBases++
		}
	}
	if h.graphRes != nil {
		w.graphReserved = true
		if b := level0Base(h); b != c.graphBase {
			c.graphBase = b
			w.graphBases++
		}
	}
	if h.arena.codesRes != nil {
		w.codeReserved = true
		if b := codesBase(h); b != c.codeBase {
			c.codeBase = b
			w.codeBases++
		}
	}
}

// growAndWitness inserts n vectors one at a time, sampling both slabs after each
// insert. It is deliberately serial: the property under test is about the growth
// mechanism, and the concurrent variants live below.
func growAndWitness(t *testing.T, h *hnsw, n, dim int) growthWitness {
	t.Helper()
	rng := rand.New(rand.NewSource(int64(n)*7 + int64(dim)))
	v := make([]float32, dim)
	var w growthWitness
	cur := growthCursor{vecCap: -1, graphCap: -1, codeCap: -1}
	for i := 0; i < n; i++ {
		for d := range v {
			v[d] = rng.Float32()
		}
		if _, _, err := h.Insert(uint64(i), v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
		w.observe(h, &cur)
	}
	return w
}

// assertStableBases is THE assertion: many capacity growths, exactly one base
// address per slab. wantCodes is false for an unquantized index, which has no
// codes slab at all.
func assertStableBases(t *testing.T, w growthWitness, wantCodes bool) {
	t.Helper()
	if !w.vecReserved || !w.graphReserved || (wantCodes && !w.codeReserved) {
		t.Fatalf("slabs never became reservation-backed (vec=%v graph=%v codes=%v) — not exercising in-place growth",
			w.vecReserved, w.graphReserved, w.codeReserved)
	}
	check := func(name string, growths, bases int) {
		if growths < 3 {
			t.Fatalf("%s: only %d growth boundaries crossed — the test proves nothing about growing", name, growths)
		}
		if bases != 1 {
			t.Errorf("%s took %d distinct base addresses across %d growths; want exactly 1 (in-place)", name, bases, growths)
		}
	}
	check("vector slab", w.vecGrowths, w.vecBases)
	check("level-0 slab", w.graphGrowths, w.graphBases)
	if wantCodes {
		check("codes slab", w.codeGrowths, w.codeBases)
	}
	t.Logf("vec: %d growths / %d bases; graph: %d growths / %d bases; codes: %d growths / %d bases",
		w.vecGrowths, w.vecBases, w.graphGrowths, w.graphBases, w.codeGrowths, w.codeBases)
}

// TestGrowInPlaceStableBaseHeap is the heap-backed half: an anonymous
// reservation, committed forward with mprotect. The base address of arena.vecs
// and of the level-0 slab must not change across a dozen doublings.
func TestGrowInPlaceStableBaseHeap(t *testing.T) {
	withSmallReservations(t, 64<<10, 64<<10, 64)
	cfg := Config{Dim: 32, Metric: L2, M: 8, EfConstruction: 32, EfSearch: 16, Seed: 5}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = h.Close() }()
	assertStableBases(t, growAndWitness(t, h, 6000, cfg.Dim), false)
}

// TestGrowInPlaceStableBaseMmap is the mmap-backed half: a file-backed
// reservation, committed forward by ftruncate + a MAP_FIXED mapping of the new
// TAIL only. Same assertion — and this is the backing where a moving base was
// not merely slow but a SIGBUS risk for any reader still walking the old range.
func TestGrowInPlaceStableBaseMmap(t *testing.T) {
	withSmallReservations(t, 64<<10, 64<<10, 64)
	requireFileBackedReservations(t)
	dir := t.TempDir()
	cfg := Config{
		Dim: 32, Metric: L2, M: 8, EfConstruction: 32, EfSearch: 16, Seed: 5,
		Quant:         QuantSQ8,
		QuantStorage:  QuantMmap,
		MmapPath:      filepath.Join(dir, "vecs.dat"),
		GraphMmapPath: filepath.Join(dir, "graph.dat"),
	}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = h.Close() }()
	w := growAndWitness(t, h, 6000, cfg.Dim)
	assertStableBases(t, w, true)
	// The committed view the persist path msync's must track the reservation, not
	// the stale pre-reservation mapping.
	if len(h.arena.mmapRegion) == 0 || unsafe.Pointer(&h.arena.mmapRegion[0]) != vecsBase(h) {
		t.Error("arena.mmapRegion does not track the reservation's committed range")
	}
	if len(h.graphRegion) == 0 || unsafe.Pointer(&h.graphRegion[0]) != level0Base(h) {
		t.Error("graphRegion does not track the reservation's committed range")
	}
}

// TestGrowInPlaceEquivalentHeap / ...Mmap are the correctness half of the
// stable-base claim, and they are a DIFFERENTIAL rather than a recall bar: the
// same vectors, the same seed, inserted into one index whose slabs grow in place
// on a reservation and one whose slabs grow the old way, must produce identical
// stored vectors AND identical search results. HNSW insertion is deterministic
// under a fixed seed, so any divergence at all — a byte lost at a commit
// boundary, a stale prefix, an off-by-one page — shows up as a differing result
// list. A recall threshold could not do this: absolute self-recall depends on
// dim/M/quantization, not on whether growth was correct.
func TestGrowInPlaceEquivalentHeap(t *testing.T) {
	testGrowEquivalent(t, 5000, 64, func(t *testing.T) Config {
		return Config{Dim: 24, Metric: L2, M: 8, EfConstruction: 32, EfSearch: 16, Seed: 9}
	})
}

// TestGrowInPlaceEquivalentMmap is the file-backed half of that differential.
// It needs a platform where an mmap-backed slab can itself be
// reservation-backed, which is not every platform that maps files at all —
// see requireFileBackedReservations.
func TestGrowInPlaceEquivalentMmap(t *testing.T) {
	requireFileBackedReservations(t)
	testGrowEquivalent(t, 5000, 64, mmapGrowConfig(24, 9))
}

// TestGrowRelocationEquivalentHeap / ...Mmap drive the OTHER reservation path:
// factor 1 makes every reservation exactly as large as the slab that triggered
// it, so the very next grow exhausts it and relocates to a fresh, larger one.
// That is the rare fallback (a slab outgrowing a 64x reservation), and it is the
// only path that still copies — exactly where a lost or duplicated byte hides.
func TestGrowRelocationEquivalentHeap(t *testing.T) {
	testGrowEquivalent(t, 5000, 1, func(t *testing.T) Config {
		return Config{Dim: 24, Metric: L2, M: 8, EfConstruction: 32, EfSearch: 16, Seed: 4}
	})
}

// TestGrowRelocationEquivalentMmap drives the relocation path on file-backed
// slabs, where the copy is between two MAPPINGS rather than two heap
// allocations. Skipped where those slabs never take a reservation.
func TestGrowRelocationEquivalentMmap(t *testing.T) {
	requireFileBackedReservations(t)
	testGrowEquivalent(t, 5000, 1, mmapGrowConfig(24, 4))
}

// mmapGrowConfig returns a config factory that mmap-backs BOTH slabs in a fresh
// temp dir per index (the differential builds two, and they must not share
// files).
func mmapGrowConfig(dim int, seed int64) func(*testing.T) Config {
	return func(t *testing.T) Config {
		dir := t.TempDir()
		return Config{
			Dim: dim, Metric: L2, M: 8, EfConstruction: 32, EfSearch: 16, Seed: seed,
			Quant:         QuantSQ8,
			QuantStorage:  QuantMmap,
			MmapPath:      filepath.Join(dir, "vecs.dat"),
			GraphMmapPath: filepath.Join(dir, "graph.dat"),
		}
	}
}

func testGrowEquivalent(t *testing.T, n int, factor int64, cfgFor func(*testing.T) Config) {
	t.Helper()
	dim := cfgFor(t).Dim
	rng := rand.New(rand.NewSource(int64(n)))
	src := make([][]float32, n)
	for i := range src {
		v := make([]float32, dim)
		for d := range v {
			v[d] = rng.Float32()
		}
		src[i] = v
	}

	build := func(reserved bool) *hnsw {
		if reserved {
			withSmallReservations(t, 64<<10, 64<<10, factor)
		} else {
			// A threshold no slab can reach keeps this index entirely on the legacy
			// copy/remap growth path — the reference behavior.
			withSmallReservations(t, math.MaxInt64, 64<<20, 64)
		}
		h, err := newHNSW(cfgFor(t))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = h.Close() })
		for i, v := range src {
			if _, _, err := h.Insert(uint64(i), v, 0, nil, nil, nil, CASCond{}); err != nil {
				t.Fatalf("insert %d: %v", i, err)
			}
		}
		return h
	}

	ref := build(false)
	if ref.arena.vecsRes != nil || ref.graphRes != nil {
		t.Fatal("reference index unexpectedly took a reservation")
	}
	got := build(true)
	if got.arena.vecsRes == nil || got.graphRes == nil {
		t.Fatalf("slabs never became reservation-backed (vec=%v graph=%v) — not exercising in-place growth",
			got.arena.vecsRes != nil, got.graphRes != nil)
	}

	// 1. Every vector byte-identical to what was written, in both indices.
	for i := 0; i < n; i++ {
		v, _, _, _, _, ok := got.Get(uint64(i))
		if !ok {
			t.Fatalf("id %d missing after growth", i)
		}
		for d := range v {
			if v[d] != src[i][d] {
				t.Fatalf("id %d dim %d: got %v want %v", i, d, v[d], src[i][d])
			}
		}
	}
	// 2. Identical graphs, observed through identical search results.
	step := max(1, n/200)
	for i := 0; i < n; i += step {
		want, err := ref.Search(src[i], 10)
		if err != nil {
			t.Fatal(err)
		}
		have, err := got.Search(src[i], 10)
		if err != nil {
			t.Fatal(err)
		}
		if len(want) != len(have) {
			t.Fatalf("query %d: reference returned %d results, reserved index %d", i, len(want), len(have))
		}
		for j := range want {
			if want[j].ID != have[j].ID || want[j].Score != have[j].Score {
				t.Fatalf("query %d rank %d: reference %v, reserved index %v — in-place growth changed the graph",
					i, j, want[j], have[j])
			}
		}
	}
}

// TestGrowInPlaceConcurrentHeap / ...Mmap re-run grow_race_test.go's hammer with
// the reservation path engaged. The invariant is unchanged — growth under the
// write lock is invisible to readers — but the MECHANISM under it is different
// (commit-in-place rather than copy/remap), so it needs its own -race coverage.
func TestGrowInPlaceConcurrentHeap(t *testing.T) {
	withSmallReservations(t, 64<<10, 64<<10, 64)
	cfg := Config{Dim: 32, Metric: L2, M: 8, EfConstruction: 64, EfSearch: 32, Seed: 7}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = h.Close() }()
	growHammer(t, h, 6000, cfg.Dim, 3, 4)
	if h.arena.vecsRes == nil || h.graphRes == nil {
		t.Fatal("slabs never became reservation-backed — the hammer did not exercise in-place growth")
	}
}

// TestGrowInPlaceConcurrentMmap is the same race coverage for file-backed
// slabs, whose commit maps a new tail over the reservation instead of
// re-protecting anonymous pages. Skipped where that is unavailable.
func TestGrowInPlaceConcurrentMmap(t *testing.T) {
	requireFileBackedReservations(t)
	withSmallReservations(t, 64<<10, 64<<10, 64)
	dir := t.TempDir()
	cfg := Config{
		Dim: 32, Metric: L2, M: 8, EfConstruction: 64, EfSearch: 32, Seed: 7,
		Quant:         QuantSQ8,
		QuantStorage:  QuantMmap,
		MmapPath:      filepath.Join(dir, "vecs.dat"),
		GraphMmapPath: filepath.Join(dir, "graph.dat"),
	}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = h.Close() }()
	growHammer(t, h, 6000, cfg.Dim, 3, 4)
	if h.arena.vecsRes == nil || h.graphRes == nil {
		t.Fatal("slabs never became reservation-backed — the hammer did not exercise in-place growth")
	}
}

// TestGrowInPlaceRelocationConcurrent is the hammer against the RELOCATING
// variant (factor 1 ⇒ every grow exhausts its reservation), where the base
// address really does move and, for the heap backing, the prefix really is
// copied. It is the one that would SIGBUS or race if the relocation were not
// fully inside the write lock.
func TestGrowInPlaceRelocationConcurrent(t *testing.T) {
	withSmallReservations(t, 64<<10, 64<<10, 1)
	cfg := Config{Dim: 32, Metric: L2, M: 8, EfConstruction: 64, EfSearch: 32, Seed: 7}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = h.Close() }()
	growHammer(t, h, 6000, cfg.Dim, 3, 4)
}

// TestUpsertAcrossGrowHeap / ...Mmap answer the question a COPY-based growth
// scheme would have to answer and this one dissolves: committed vector bytes are
// NOT immutable — arena.Insert reuses a free slot and rewrites its vector in
// place — so any scheme that copied the prefix outside the lock would need dirty
// tracking to avoid losing a write that landed mid-copy.
//
// The differential: hammer delete+reinsert (Collection.Upsert's shape) on a
// churning id range while other writers push the slabs through many growth
// boundaries, then verify EVERY live id reads back the exact vector its last
// successful upsert wrote. A lost write shows up as a stale vector; a
// mis-targeted one as another id's vector.
func TestUpsertAcrossGrowHeap(t *testing.T) {
	withSmallReservations(t, 64<<10, 64<<10, 64)
	testUpsertAcrossGrow(t, Config{Dim: 32, Metric: L2, M: 8, EfConstruction: 32, EfSearch: 16, Seed: 21})
}

func TestUpsertAcrossGrowMmap(t *testing.T) {
	withSmallReservations(t, 64<<10, 64<<10, 64)
	dir := t.TempDir()
	testUpsertAcrossGrow(t, Config{
		Dim: 32, Metric: L2, M: 8, EfConstruction: 32, EfSearch: 16, Seed: 21,
		Quant:         QuantSQ8,
		QuantStorage:  QuantMmap,
		MmapPath:      filepath.Join(dir, "vecs.dat"),
		GraphMmapPath: filepath.Join(dir, "graph.dat"),
	})
}

// TestUpsertAcrossGrowRelocating runs the same differential against the
// relocating path, which IS a copy — the case where a concurrent rewrite could
// actually be lost if the copy were not exclusive of writers.
func TestUpsertAcrossGrowRelocating(t *testing.T) {
	withSmallReservations(t, 64<<10, 64<<10, 1)
	testUpsertAcrossGrow(t, Config{Dim: 32, Metric: L2, M: 8, EfConstruction: 32, EfSearch: 16, Seed: 21})
}

func testUpsertAcrossGrow(t *testing.T, cfg Config) {
	t.Helper()
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = h.Close() }()

	const (
		churnIDs = 300  // ids repeatedly deleted and reinserted (they free slots for reuse)
		freshIDs = 5000 // ids inserted once, driving the slabs through growth boundaries
		rounds   = 12
	)
	// Deterministic vector for (id, generation): the expected content is a pure
	// function of the pair, so the checker needs no shared state with the writers.
	vecFor := func(id uint64, gen int, dst []float32) []float32 {
		rng := rand.New(rand.NewSource(int64(id)*1_000_003 + int64(gen)))
		for d := range dst {
			dst[d] = rng.Float32()
		}
		return dst
	}

	buf := make([]float32, cfg.Dim)
	for id := 0; id < churnIDs; id++ {
		if _, _, err := h.Insert(uint64(id), vecFor(uint64(id), 0, buf), 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("seed insert %d: %v", id, err)
		}
	}

	var wg, rwg sync.WaitGroup
	var searches atomic.Int64
	var stop atomic.Bool

	// A reader, so the whole differential runs with queries in flight.
	rwg.Add(1)
	go func() {
		defer rwg.Done()
		qrng := rand.New(rand.NewSource(1234))
		q := make([]float32, cfg.Dim)
		var dst []Result
		for !stop.Load() {
			for d := range q {
				q[d] = qrng.Float32()
			}
			var serr error
			if dst, serr = h.SearchInto(dst[:0], q, 10, Filter{}); serr != nil {
				t.Errorf("search: %v", serr)
				return
			}
			searches.Add(1)
		}
	}()

	// Grower: fresh ids, so the slabs keep crossing boundaries.
	wg.Add(1)
	go func() {
		defer wg.Done()
		gbuf := make([]float32, cfg.Dim)
		for i := 0; i < freshIDs; i++ {
			id := uint64(churnIDs + i)
			if _, _, ierr := h.Insert(id, vecFor(id, 0, gbuf), 0, nil, nil, nil, CASCond{}); ierr != nil {
				t.Errorf("fresh insert %d: %v", id, ierr)
				return
			}
		}
	}()

	// Churner: delete + reinsert, which is what recycles slots and rewrites
	// already-committed vector bytes in place.
	wg.Add(1)
	go func() {
		defer wg.Done()
		cbuf := make([]float32, cfg.Dim)
		for gen := 1; gen <= rounds; gen++ {
			for id := 0; id < churnIDs; id++ {
				if _, derr := h.Delete(uint64(id), CASCond{}); derr != nil {
					t.Errorf("delete %d: %v", id, derr)
					return
				}
				if _, _, ierr := h.Insert(uint64(id), vecFor(uint64(id), gen, cbuf), 0, nil, nil, nil, CASCond{}); ierr != nil {
					t.Errorf("reinsert %d gen %d: %v", id, gen, ierr)
					return
				}
			}
		}
	}()

	wg.Wait() // both writers done
	stop.Store(true)
	rwg.Wait()

	if searches.Load() == 0 {
		t.Fatal("no searches ran concurrently — the differential is not exercising the invariant")
	}

	// Every churned id must hold its FINAL generation's vector, exactly.
	want := make([]float32, cfg.Dim)
	for id := 0; id < churnIDs; id++ {
		got, _, _, _, _, ok := h.Get(uint64(id))
		if !ok {
			t.Fatalf("churned id %d missing after growth", id)
		}
		vecFor(uint64(id), rounds, want)
		for d := range want {
			if got[d] != want[d] {
				t.Fatalf("churned id %d dim %d: got %v want %v — an upsert was lost or landed in the wrong slab",
					id, d, got[d], want[d])
			}
		}
	}
	// And every fresh id must hold its own, unclobbered by slot reuse.
	for i := 0; i < freshIDs; i++ {
		id := uint64(churnIDs + i)
		got, _, _, _, _, ok := h.Get(id)
		if !ok {
			t.Fatalf("fresh id %d missing after growth", id)
		}
		vecFor(id, 0, want)
		for d := range want {
			if got[d] != want[d] {
				t.Fatalf("fresh id %d dim %d: got %v want %v", id, d, got[d], want[d])
			}
		}
	}
}

// TestReserveSizePolicy pins slabReserveSize's arithmetic, which is easy to get
// subtly wrong (a reservation smaller than the need would fail a grow the legacy
// path would have served).
func TestReserveSizePolicy(t *testing.T) {
	oldMin, oldFactor := slabReserveMin, slabReserveFactor
	slabReserveMin, slabReserveFactor = 64<<20, 64
	t.Cleanup(func() { slabReserveMin, slabReserveFactor = oldMin, oldFactor })

	cases := []struct {
		name       string
		need, hint int64
		want       int64
	}{
		{"floor applies to a small need", 1 << 20, 0, 64 << 20},
		{"factor applies past the floor", 8 << 20, 0, 512 << 20},
		{"hint doubles and overrides the factor", 1 << 20, 1 << 30, 2 << 30},
		{"ceiling caps the factor", 4 << 30, 0, slabReserveMax},
		{"need always honored past the ceiling", 128 << 30, 0, 128 << 30},
		{"hint below the floor is floored", 1 << 20, 1 << 20, 64 << 20},
	}
	for _, tc := range cases {
		if got := slabReserveSize(tc.need, tc.hint); got != tc.want {
			t.Errorf("%s: slabReserveSize(%d, %d) = %d, want %d", tc.name, tc.need, tc.hint, got, tc.want)
		}
	}
}

// TestGrowInPlaceBulkBuildAndRestore covers the two presize paths, which now
// route through the same chokepoint as incremental growth: a concurrent bulk
// build (arena.Reserve + presizeGraphSlab) and a snapshot round-trip.
func TestGrowInPlaceBulkBuildAndRestore(t *testing.T) {
	const n, dim = 3000, 32
	cfg := Config{Dim: dim, Metric: L2, M: 8, EfConstruction: 32, EfSearch: 16, Seed: 3}

	rng := rand.New(rand.NewSource(77))
	ids := make([]uint64, n)
	vecs := make([][]float32, n)
	for i := range vecs {
		ids[i] = uint64(i)
		v := make([]float32, dim)
		for d := range v {
			v[d] = rng.Float32()
		}
		vecs[i] = v
	}
	withSmallReservations(t, 64<<10, 64<<10, 64)
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = h.Close() }()
	if err := h.BuildConcurrent(ids, vecs, 4); err != nil {
		t.Fatal(err)
	}
	if h.arena.vecsRes == nil || h.graphRes == nil {
		t.Fatal("bulk build did not put the slabs on a reservation")
	}
	// BuildConcurrent's link phase is parallel, so its graph is not reproducible
	// edge for edge and a differential against a legacy-path build would be
	// comparing two legitimately different graphs. A recall bar is the right shape
	// here: a slab that lost or misplaced bytes during presize collapses it, while
	// the occasional genuinely-unreachable point does not.
	var probes, hits int
	for i := 0; i < n; i += 7 {
		res, serr := h.Search(vecs[i], 1)
		if serr != nil {
			t.Fatal(serr)
		}
		probes++
		if len(res) > 0 && res[0].ID == ids[i] {
			hits++
		}
	}
	if got := float64(hits) / float64(probes); got < 0.95 {
		t.Fatalf("post-build self-recall@1 = %.3f (%d/%d), want >= 0.95", got, hits, probes)
	}

	var buf bytes.Buffer
	if err := h.Snapshot(&buf); err != nil {
		t.Fatal(err)
	}
	restored, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = restored.Close() }()
	if err := restored.Restore(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i += 97 {
		got, _, _, _, _, ok := restored.Get(ids[i])
		if !ok {
			t.Fatalf("id %d missing after restore", ids[i])
		}
		for d := range got {
			if got[d] != vecs[i][d] {
				t.Fatalf("id %d dim %d after restore: got %v want %v", ids[i], d, got[d], vecs[i][d])
			}
		}
	}
}
