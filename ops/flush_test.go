// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"testing"

	"github.com/rostamlabs/rostam/cache"
)

// TestBuiltinFlushWipesKeyspace dispatches the flush op through a TxContext-backed
// store and asserts every prior key misses afterward while a post-flush write reads
// back — the single-cache contract every shard group's FSM relies on.
func TestBuiltinFlushWipesKeyspace(t *testing.T) {
	r, tx := newTestSetup(t)

	putH, _, _, _ := r.Lookup("put")
	getH, _, _, _ := r.Lookup("get")

	// Seed several keys.
	for _, k := range [][]byte{[]byte("a"), []byte("b"), []byte("c")} {
		if _, err := putH(tx, EncodePutArgs(k, []byte("v"), 0)); err != nil {
			t.Fatalf("put %q: %v", k, err)
		}
	}

	flushH, kind, _, ok := r.Lookup("flush")
	if !ok {
		t.Fatal("flush not registered")
	}
	if kind != OpReadWrite {
		t.Fatalf("flush kind = %v, want OpReadWrite", kind)
	}
	// Flush ignores its args (nil here) and returns an empty ack.
	res, err := flushH(tx, nil)
	if err != nil {
		t.Fatalf("flush handler: %v", err)
	}
	if len(res) != 0 {
		t.Fatalf("flush result = %q, want empty", res)
	}

	// Every seeded key must now miss.
	for _, k := range [][]byte{[]byte("a"), []byte("b"), []byte("c")} {
		if _, err := getH(tx, EncodeKeyArgs(k)); err != cache.ErrNotFound {
			t.Fatalf("get %q after flush: err = %v, want ErrNotFound", k, err)
		}
	}

	// A post-flush write reads back — the store is live, not poisoned.
	if _, err := putH(tx, EncodePutArgs([]byte("d"), []byte("w"), 0)); err != nil {
		t.Fatalf("put after flush: %v", err)
	}
	got, err := getH(tx, EncodeKeyArgs([]byte("d")))
	if err != nil {
		t.Fatalf("get after post-flush put: %v", err)
	}
	if !bytes.Equal(got, []byte("w")) {
		t.Fatalf("post-flush get = %q, want w", got)
	}
}
