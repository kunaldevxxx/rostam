// SPDX-License-Identifier: Apache-2.0
//go:build linux && (amd64 || arm64)

// The 64-bit premise is in the build tag on purpose. The whole scheme spends
// address space freely because there is 128 TiB of it; on a 32-bit target
// (linux/386, linux/arm) uintptr(res) would silently truncate any reservation
// past 4 GiB, and even below that a 64x over-reservation would exhaust a 3 GiB
// user address space almost immediately. Both outcomes are memory-safe but
// mis-sized, which is the worst kind of bug to leave to runtime. 32-bit Linux
// gets reserve_other.go's stubs and keeps the copy/remap growth path — mmap
// STORAGE still works there, since mmap_linux.go's tag is still plain `linux`.

package vector

import (
	"fmt"
	"os"
	"unsafe"

	"golang.org/x/sys/unix"
)

// slabReservation owns a contiguous range of reserved virtual address space plus
// the prefix of it that is currently committed (mapped readable/writable). See
// reserve.go for why the split exists.
//
// The range is reserved as one PROT_NONE MAP_NORESERVE anonymous mapping.
// Committing overwrites a prefix of it with the real mapping via MAP_FIXED
// (file-backed) or turns it readable/writable via mprotect (anonymous). Only the
// UNCOMMITTED tail is ever touched by a commit, so the committed prefix — the
// bytes readers are walking — is never unmapped, moved, or re-protected.
//
// OWNERSHIP — THERE IS NO FINALIZER. A reservation is OS memory the Go GC cannot
// see, and deliberately carries no runtime.SetFinalizer: a finalizer could fire
// while a []float32 derived from the region is still in flight (the derived
// slice does not keep the reservation reachable), turning a missed Close from a
// leak into a SIGSEGV. The consequence is that it must be released EXPLICITLY,
// via arena.Close / closeGraphMmap for a whole index, or via
// arena.releaseVecsBacking / releaseCodesBacking for a single slab being dropped
// or replaced. Any path that discards or overwrites a slab header without one of
// those leaks the mapping for the life of the process. What keeps that bounded
// in the worst case is slabReserveThreshold, not the GC: a slab only takes a
// reservation once it is large enough that live ones are capped by physical
// memory (see reserve.go).
//
// Not safe for concurrent use; the owning index's write lock serializes it.
type slabReservation struct {
	base   unsafe.Pointer // start of the reserved range; stable for the life of the reservation
	resLen uintptr        // reserved bytes (address space)
	commit uintptr        // committed bytes, always a prefix [base, base+commit)
	f      *os.File       // backing file; nil = anonymous (heap-mode slab)
}

// slabReservationsSupported reports whether this build can reserve address space
// separately from committing it. Lets tests that are ONLY meaningful with
// reservations skip rather than fail on a platform that has none.
const slabReservationsSupported = true

// fileBackedSlabReservationsSupported reports whether an mmap-backed slab can
// live on a reservation too, not just an anonymous one. Linux commits a
// file-backed slab by mapping the new tail over its slice of the reservation
// with MAP_FIXED, so both backings get the in-place growth; Windows has no
// equivalent that is safe to use (see reserve_windows.go), which is why the two
// capabilities are separate constants rather than one.
const fileBackedSlabReservationsSupported = true

// mapped reports whether the reservation still holds its range. Portable
// counterpart of the base pointer, so tests can assert "this was released"
// without reaching into a platform-specific field.
func (r *slabReservation) mapped() bool { return r != nil && r.base != nil }

// pageAlign rounds n up to a whole number of pages. Both mprotect and a
// MAP_FIXED file mapping require page-aligned addresses and file offsets, and
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
// first commitBytes of it. A nil f gives an anonymous (heap-mode) slab; a
// non-nil f backs the committed prefix with that file from offset 0, and the
// file is truncated up as the commit grows.
//
// The caller owns f: release() unmaps but never closes it.
func newSlabReservation(f *os.File, reserveBytes, commitBytes int64) (*slabReservation, error) {
	res := pageAlign(reserveBytes)
	if com := pageAlign(commitBytes); res < com {
		res = com
	}
	if res <= 0 {
		return nil, errSlabReserveUnsupported
	}
	base, err := unix.MmapPtr(-1, 0, nil, uintptr(res), unix.PROT_NONE,
		unix.MAP_PRIVATE|unix.MAP_ANONYMOUS|unix.MAP_NORESERVE)
	if err != nil {
		return nil, fmt.Errorf("vector: reserve %d bytes of address space: %w", res, err)
	}
	r := &slabReservation{base: base, resLen: uintptr(res), f: f}
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
// The syscall is issued against the DELTA only — this is the whole point:
// growing a 3 GB slab by one page costs the same as growing an empty one.
func (r *slabReservation) commitTo(n int64) error {
	want := uintptr(pageAlign(n))
	if want <= r.commit {
		return nil
	}
	if want > r.resLen {
		return errSlabReserveExhausted
	}
	off, delta := r.commit, want-r.commit
	if r.f != nil {
		// File-backed: the file must cover the range before it is mapped, then
		// the new tail is mapped over its slice of the reservation. The already
		// committed prefix keeps its own (older) mappings of the same file at the
		// same offsets, so its pages are literally untouched.
		if err := r.f.Truncate(int64(want)); err != nil {
			return fmt.Errorf("vector: truncate (commit) to %d: %w", want, err)
		}
		fd := int(r.f.Fd()) //nolint:gosec // uintptr->int: fd values are small and positive on Linux
		if _, err := unix.MmapPtr(fd, int64(off), unsafe.Add(r.base, off), delta,
			unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED|unix.MAP_FIXED); err != nil {
			return fmt.Errorf("vector: commit file bytes %d..%d in place: %w", off, want, err)
		}
	} else {
		// Anonymous: the pages are already reserved, just inaccessible. mprotect
		// makes them usable; they fault in zero-filled on first touch, which is
		// exactly the guarantee make([]T, n) gives.
		//
		// Audited: the range is inside the reservation by the bound above, and
		// the []byte is used only as mprotect's address/length argument.
		//nolint:gosec // G103: reviewed unsafe view of reserved address space.
		tail := unsafe.Slice((*byte)(unsafe.Add(r.base, off)), delta)
		if err := unix.Mprotect(tail, unix.PROT_READ|unix.PROT_WRITE); err != nil {
			return fmt.Errorf("vector: commit anon bytes %d..%d in place: %w", off, want, err)
		}
	}
	r.commit = want
	return nil
}

// region returns the committed prefix as a byte slice. The slice aliases the
// reservation; it is valid until release() and its data pointer is STABLE across
// every commitTo (only its length grows).
func (r *slabReservation) region() []byte {
	if r == nil || r.base == nil || r.commit == 0 {
		return nil
	}
	// Audited: [base, base+commit) is mapped readable/writable by construction,
	// byte has no pointers so the GC never scans it, and the reservation outlives
	// every slice derived from it (it is released only from Close, by which point
	// the owning arena/index has dropped its headers).
	//nolint:gosec // G103: reviewed unsafe view of the committed reservation.
	return unsafe.Slice((*byte)(r.base), r.commit)
}

// sync flushes the committed prefix of a file-backed reservation. A no-op for an
// anonymous one (there is nothing to write back).
func (r *slabReservation) sync() error {
	if r == nil || r.f == nil || r.commit == 0 {
		return nil
	}
	if err := unix.Msync(r.region(), unix.MS_SYNC); err != nil {
		return fmt.Errorf("vector: msync reservation: %w", err)
	}
	return nil
}

// release unmaps the whole reservation — reserved tail and every committed
// sub-mapping in one munmap. Idempotent. Does not close the backing file.
func (r *slabReservation) release() error {
	if r == nil || r.base == nil {
		return nil
	}
	// The bookkeeping is cleared only AFTER the unmap succeeds. Doing it
	// first would say "released" about a range that is still mapped:
	// mapped() would report false, liveSlabReservations would under-count
	// the very leak it exists to detect - so reserve_leak_test.go would
	// pass while the range leaked - and a second release would return
	// early instead of trying again.
	//
	// munmap of a range this package mapped itself is close to
	// unfailable, which is why the order went unnoticed; that makes it
	// cheap to get right rather than a reason to leave it.
	if err := unix.MunmapPtr(r.base, r.resLen); err != nil {
		return fmt.Errorf("vector: munmap reservation: %w", err)
	}

	r.base, r.commit, r.resLen = nil, 0, 0
	liveSlabReservations.Add(-1)
	return nil
}
