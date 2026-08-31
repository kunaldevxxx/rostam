// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"sync/atomic"
)

// ADDRESS-SPACE OVER-RESERVATION — how the big slabs stopped moving.
//
// THE STALL. arena.vecs (dim*4 bytes per slot) and the level-0 graph slab
// (m0*4 bytes per slot) are flat, contiguous, base+slot*stride arrays — the
// batched distance kernels and the level-0 addressing both require exactly
// that, so neither can be chunked. They used to grow the only way a flat array
// can: allocate a bigger one and copy (heap), or munmap/ftruncate/mmap and let
// the kernel hand back a DIFFERENT address (mmap). Both run inside placement,
// under h.mu's write lock, with every reader excluded. At 1M x 768d the heap
// copy is a ~3 GB memcpy — a multi-hundred-millisecond window in which every
// query is blocked, and the last known reader stall in the engine.
//
// THE FIX. Separate RESERVING address space from COMMITTING memory. On 64-bit
// Linux virtual address space is free: a PROT_NONE MAP_NORESERVE mapping costs
// one VMA, no page tables, no physical pages, and nothing against the overcommit
// budget. So a slab reserves a large contiguous range ONCE and then commits it
// forward in place:
//
//	heap-backed slab  →  mprotect(base+committed, delta, PROT_READ|PROT_WRITE)
//	mmap-backed slab  →  ftruncate(fd, want) then
//	                     mmap(base+committed, delta, MAP_SHARED|MAP_FIXED, fd, committed)
//
// Both are O(1) single syscalls against the DELTA, never the whole slab. The
// base address is unchanged, the already-committed prefix is not touched at all
// — not copied, not remapped, not even TLB-shot — and cap() simply gets bigger.
//
// WHY THIS IS SAFE FOR A CONCURRENT UPSERT (the question a copy-based scheme has
// to answer and this one dissolves). Committed vector bytes are NOT immutable:
// arena.Insert reuses a free slot and rewrites its vector in place, so any
// scheme that copied the prefix unlocked would need dirty tracking plus a
// re-copy of slots written during the copy. Here there is no copy. Growth only
// makes address space beyond the current end usable; every byte of every
// already-placed slot stays at the same address in the same page for the whole
// life of the reservation. A write that lands during a grow lands in the one and
// only slab, because there is only ever one. The dirty-copy problem does not
// exist rather than being solved.
//
// WHAT DID NOT CHANGE. Growth still happens under h.mu's write lock, so the
// slab-growth invariant that grow_race_test.go pins holds verbatim; this is
// purely a cost reduction of that critical section, from O(bytes) to O(1). It
// also makes the slice header safer as a side effect: a grow now publishes a
// header with the SAME data pointer and a larger cap, so even a torn read of it
// yields a valid pointer into live memory.
//
// WHICH SLABS, AND WHAT IS LEFT. Three per-slot arrays are flat, large, and
// POINTER-FREE, which is the combination a reservation needs (off-heap memory is
// invisible to the GC, so anything holding pointers must stay on the Go heap):
// arena.vecs (dim*4 B/slot), the level-0 graph slab (m0*4 B/slot), and the
// quantization codes (dim B/slot under SQ8 — a quarter of the vectors, and the
// dominant stall on a quantized index once the other two stop copying). All
// three now grow in place.
//
// What still copies is every 8-byte-per-slot side array: expires, versions, ids
// (pointer-free but small), metadata, keyExpires, sparse and hnsw.nodes
// (pointer-bearing, so they CANNOT move off-heap). Measured on a 100k x 256d SQ8
// index at a doubling boundary, before vs after:
//
//	arena.vecs      100 MB copy/remap   →   ~50 µs   (in place)
//	level0           12 MB copy/remap   →   ~20 µs   (in place)
//	codes            25 MB copy        15.7 ms  →  ~7 µs   (in place)
//	six side arrays   ~5 MB copy         2.4 ms   (unchanged)
//	hnsw.nodes       0.8 MB copy         0.4 ms   (unchanged)
//
// So the write-lock hold at a grow goes from "proportional to the whole index"
// to a fixed few milliseconds of per-slot bookkeeping. Reserving the three
// pointer-free side arrays too would roughly halve that residual at the cost of
// three more VMAs per collection; it was measured, judged not worth it, and is
// the obvious next step if those milliseconds ever matter.
//
// END TO END, at the scale the stall was reported at — 500k x 768d, worst query
// latency observed by a searcher running across the boundary insert
// (TestGrowStallLatency):
//
//	heap  1.502 s  →  10.2 ms   (147x)
//	mmap  188.9 ms →  17.1 ms   (11x)
//
// ONE OPERATIONAL CONSEQUENCE. A reserved slab is off-heap, so it stops counting
// toward the Go heap and therefore toward GOGC's next-GC target and GOMEMLIMIT.
// RSS is unchanged; what changes is that the process's largest pointer-free
// allocation no longer inflates the GC's heap goal, which lowers peak footprint
// rather than raising it. This is not a new mode — QuantMmap already put the
// float32 vectors off-heap for exactly this reason — it now just applies to the
// heap-backed configuration too.
//
// FALLBACK, EVERYWHERE. A reservation is an optimization, never a requirement.
// Where one cannot be made it simply is not made, and a reservation that runs
// out of reserved range relocates once to a bigger one — paying the old
// copy/remap exactly there.
//
// Which is available is now two questions rather than one, because Windows
// answers them differently. slabReservationsSupported says whether ANONYMOUS
// slabs can be reserved: true on 64-bit Linux and 64-bit Windows, where
// VirtualAlloc splits reserve from commit natively. fileBackedSlabReservations-
// Supported says whether an MMAP-BACKED slab can be, and that is 64-bit Linux
// alone: committing one means mapping the new tail over its slice of the
// reservation, which Linux does with MAP_FIXED and Windows has no equivalent
// for outside the placeholder API (see reserve_windows.go). So an mmap-backed
// slab on Windows keeps the copy/remap growth path while its anonymous
// neighbours — arena.vecs and level-0 in heap mode, and the codes slab always —
// grow in place. The platforms in mmap_other.go have neither. Every one of those
// paths lands back on the pre-existing grow code, so behavior is identical and
// only latency differs.

var (
	// errSlabReserveUnsupported means this platform has no way to reserve
	// address space separately from committing it; the caller keeps the
	// copy/remap growth path.
	errSlabReserveUnsupported = errors.New("vector: address-space reservation not supported on this platform")

	// errSlabReserveExhausted means the requested commit is larger than the
	// reserved range. The caller relocates to a fresh, larger reservation.
	errSlabReserveExhausted = errors.New("vector: address-space reservation exhausted")
)

// liveSlabReservations counts reservations currently mapped: incremented by a
// successful newSlabReservation, decremented by release(). It exists because the
// ownership rule (release is EXPLICIT — see slabReservation's doc, there is no
// finalizer) is otherwise untestable: a leaked reservation is invisible to the
// GC, to -race, and to every assertion about program output — it is not a wrong
// answer, it is memory that never comes back. A test that replaces or discards
// an index watches this return to its starting value (reserve_leak_test.go).
// Free in production: two atomic adds per reservation, and reservations are rare
// by construction.
var liveSlabReservations atomic.Int64

// slabReserveMax bounds the reservation: a process with many collections must
// not spend its 128 TiB of user address space on a handful of them.
const slabReserveMax = int64(64) << 30

// slabReserveFactor is how much address space to reserve per byte the slab
// currently needs, when no explicit cap is configured. 64x means a slab
// crossing the threshold below reserves ~2 GiB and can then grow ~64-fold
// without a single relocation; the whole growth curve to slabReserveMax costs at
// most two relocations.
//
// slabReserveMin floors it, so a just-over-threshold slab does not reserve a
// range it would exhaust almost immediately.
//
// Both are vars, not consts, purely so the growth tests can shrink the
// reservation enough to drive the exhaustion/relocation path in a unit test
// instead of only at multi-gigabyte scale.
var (
	slabReserveFactor = int64(64)
	slabReserveMin    = int64(64) << 20
)

// slabReserveThreshold is the committed size a slab must reach before it is
// worth backing with a reservation. Below it the slab is small enough that a
// doubling copy is microseconds, and paying 2-4 VMAs plus a page-aligned
// allocation per index would be a bad trade for the many tiny indices a process
// holds (every test index, every empty collection). Above it the index is large
// enough that its VMAs are bounded by physical memory, which is what keeps the
// process clear of vm.max_map_count even if some index is never Closed.
//
// A var, not a const, so growth tests can lower it and exercise the reservation
// path without building a 32 MiB slab.
var slabReserveThreshold = int64(32) << 20

// slabReserveSize picks the reservation for a slab that needs `need` bytes now.
// `hint` is the configured upper bound on the slab's eventual size in bytes (0 =
// unknown): a declared cap is authoritative, but it is DOUBLED because slot
// count is not vector count — tombstoned slots stay allocated until Reclaim, so
// Capacity can run well past a MaxVectors that only gates live points.
func slabReserveSize(need, hint int64) int64 {
	want := need * slabReserveFactor
	if want/slabReserveFactor != need { // overflow on an absurd need
		want = slabReserveMax
	}
	if hint > 0 {
		want = hint * 2
	}
	if want < slabReserveMin {
		want = slabReserveMin
	}
	if want > slabReserveMax {
		want = slabReserveMax
	}
	// A reservation smaller than what the caller needs right now is useless;
	// honor the need even if it is past the ceiling (the alternative is failing
	// a grow that the old path would have served).
	if want < need {
		want = need
	}
	return want
}

// slabHintBytes converts a configured MaxVectors cap into a byte hint for a slab
// with the given per-slot stride. Returns 0 (unknown) when uncapped.
func slabHintBytes(maxVectors int64, bytesPerSlot int) int64 {
	if maxVectors <= 0 || bytesPerSlot <= 0 {
		return 0
	}
	return maxVectors * int64(bytesPerSlot)
}
