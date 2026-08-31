// SPDX-License-Identifier: Apache-2.0
//go:build windows && (amd64 || arm64)

// The 64-bit premise is in the build tag for the same reason reserve_linux.go
// puts it there: the scheme spends address space freely because there is a lot
// of it, and on a 32-bit target (windows/386) a 64x over-reservation would
// exhaust a 2 GiB user address space almost immediately. windows/386 keeps
// reserve_other.go's stubs and the copy/remap growth path — mmap STORAGE still
// works there, since mmap_windows.go's tag is plain `windows`.

package vector

import (
	"fmt"
	"os"
	"unsafe"

	"golang.org/x/sys/windows"
)

// slabReservationsSupported reports that this build can reserve address space
// and commit it forward in place.
const slabReservationsSupported = true

// fileBackedSlabReservationsSupported is false: newSlabReservation refuses a
// non-nil file, so an mmap-backed slab keeps the copy/remap growth path. See
// newSlabReservation for what it would take to lift this.
const fileBackedSlabReservationsSupported = false

// slabReservation owns a contiguous range of reserved virtual address space plus
// the prefix of it that is currently committed. See reserve.go for why the split
// exists; this is the Win32 half of it.
//
// Reserve/commit is Windows' native model, so the anonymous case maps onto it
// exactly: VirtualAlloc(MEM_RESERVE) takes address space without charging
// memory, and VirtualAlloc(MEM_COMMIT) over a sub-range of that reservation
// makes just that range usable, leaving the already-committed prefix — the
// bytes readers are walking — untouched, unmoved and un-reprotected.
//
// OWNERSHIP — THERE IS NO FINALIZER, for the reason reserve_linux.go gives: a
// finalizer could fire while a []float32 derived from the region is still in
// flight, turning a missed Close from a leak into an access violation. Release
// is explicit, from arena.Close / closeGraphMmap or the per-slab release paths.
type slabReservation struct {
	base   unsafe.Pointer // start of the reserved range; stable for the life of the reservation
	resLen uintptr        // reserved bytes (address space)
	commit uintptr        // committed bytes, always a prefix [base, base+commit)
	f      *os.File       // always nil here; see newSlabReservation
}

// mapped reports whether this slab is reservation-backed at all. A nil
// receiver is the ordinary "no reservation" case, not a bug: newSlabReservation
// returns one whenever the reservation could not be made.
func (r *slabReservation) mapped() bool { return r != nil && r.base != nil }

// pageAlign rounds n up to a whole number of pages. Commits are page-granular
// on Windows (the 64 KiB allocation granularity applies to where a reservation
// STARTS, and VirtualAlloc picks that itself when the base is NULL), and
// committing in page units is what lets the next commit start exactly where the
// previous one ended.
func pageAlign(n int64) int64 {
	p := int64(os.Getpagesize())
	if n <= 0 {
		return 0
	}
	return (n + p - 1) / p * p
}

// newSlabReservation reserves reserveBytes of address space and commits the
// first commitBytes of it.
//
// FILE-BACKED SLABS ARE NOT SUPPORTED HERE, and that is a platform limit rather
// than an omission. Linux commits a file-backed slab by mapping the new tail
// over its slice of the reservation with MAP_FIXED; Windows will not map a view
// into a range it considers reserved — MapViewOfFileEx demands the target be
// FREE, and releasing the reservation first opens a window in which another
// thread can take the address. The supported way is the placeholder API
// (VirtualAlloc2 with MEM_RESERVE_PLACEHOLDER, split with
// MEM_PRESERVE_PLACEHOLDER, filled by MapViewOfFile3 with
// MEM_REPLACE_PLACEHOLDER), which needs Windows 10 1803 or newer, hand-written
// bindings — x/sys/windows binds none of the three — and per-view bookkeeping
// that the single-munmap Linux release does not.
//
// So an mmap-backed slab gets errSlabReserveUnsupported and keeps the
// copy/remap growth path it already had, which is exactly the contract
// reserve.go states: a reservation is an optimization, never a requirement.
// The heap-backed slabs — arena.vecs and the level-0 graph in heap mode, and
// the quantization codes always — are where the larger stall was measured
// anyway (1.502 s -> 10.2 ms, against 188.9 ms -> 17.1 ms for mmap).
func newSlabReservation(f *os.File, reserveBytes, commitBytes int64) (*slabReservation, error) {
	if f != nil {
		return nil, errSlabReserveUnsupported
	}
	res := pageAlign(reserveBytes)
	if com := pageAlign(commitBytes); res < com {
		res = com
	}
	if res <= 0 {
		return nil, errSlabReserveUnsupported
	}
	base, err := windows.VirtualAlloc(0, uintptr(res), windows.MEM_RESERVE, windows.PAGE_NOACCESS)
	if err != nil {
		return nil, fmt.Errorf("vector: reserve %d bytes of address space: %w", res, err)
	}
	//nolint:govet,gosec // unsafeptr: an OS reservation address, not a Go pointer
	r := &slabReservation{base: unsafe.Pointer(base), resLen: uintptr(res)}
	liveSlabReservations.Add(1)
	if err := r.commitTo(commitBytes); err != nil {
		_ = r.release()
		return nil, err
	}
	return r, nil
}

// commitTo makes [0, n) of the reservation readable/writable, growing the
// committed prefix in place. Shrinking is not supported (a smaller n is a
// no-op). Returns errSlabReserveExhausted when n is past the reserved range, so
// the caller can relocate rather than fail the insert.
//
// The call is issued against the DELTA only — this is the whole point: growing
// a 3 GB slab by one page costs the same as growing an empty one. Committed
// pages arrive zero-filled, as they do on Linux.
//
// One honest difference from Linux: MEM_COMMIT charges the system commit limit
// (RAM + pagefile) for the delta immediately, where mprotect over a
// MAP_NORESERVE range charges nothing until a page is touched. The RESERVATION
// itself is free on both, so the over-reserve premise is unaffected; what
// differs is that a committed-but-untouched tail counts against this machine's
// commit budget. Commits track the slab's actual length, so that tail is at
// most one growth step.
func (r *slabReservation) commitTo(n int64) error {
	want := uintptr(pageAlign(n))
	if want <= r.commit {
		return nil
	}
	if want > r.resLen {
		return errSlabReserveExhausted
	}
	off, delta := r.commit, want-r.commit
	addr := uintptr(r.base) + off
	got, err := windows.VirtualAlloc(addr, delta, windows.MEM_COMMIT, windows.PAGE_READWRITE)
	if err != nil {
		return fmt.Errorf("vector: commit %d bytes at +%d: %w", delta, off, err)
	}
	if got != addr {
		// Cannot happen for a commit inside an existing reservation, but a
		// silently relocated base would hand every reader a stale pointer, so
		// it is checked rather than assumed.
		return fmt.Errorf("vector: commit relocated the range: got %#x, want %#x", got, addr)
	}
	r.commit = want
	return nil
}

// region returns the committed prefix as a byte slice aliasing the reservation.
func (r *slabReservation) region() []byte {
	if r == nil || r.base == nil || r.commit == 0 {
		return nil
	}
	// Audited: [base, base+commit) is committed readable/writable by
	// construction, byte has no pointers so the GC never scans it, and the
	// reservation outlives every slice derived from it (it is released only from
	// Close, by which point the owning arena/index has dropped its headers).
	//nolint:gosec // G103: reviewed unsafe view of the committed reservation.
	return unsafe.Slice((*byte)(r.base), r.commit)
}

// sync is a no-op: only a file-backed reservation has anything to flush, and
// newSlabReservation refuses those on this platform.
func (r *slabReservation) sync() error { return nil }

// release frees the whole reservation — committed prefix and reserved tail in
// one call. Idempotent.
func (r *slabReservation) release() error {
	if r == nil || r.base == nil {
		return nil
	}
	// MEM_RELEASE requires a zero size and frees the entire region that was
	// reserved at base, committed sub-ranges included.
	//
	// The bookkeeping is cleared only AFTER the free succeeds. Doing it first
	// would say "released" about a range still mapped: mapped() would report
	// false, liveSlabReservations would under-count the very leak it exists to
	// detect, and a second release would return early instead of retrying.
	if err := windows.VirtualFree(uintptr(r.base), 0, windows.MEM_RELEASE); err != nil {
		return fmt.Errorf("vector: release reservation: %w", err)
	}

	r.base, r.commit, r.resLen = nil, 0, 0
	liveSlabReservations.Add(-1)

	return nil
}
