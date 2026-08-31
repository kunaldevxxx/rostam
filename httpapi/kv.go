// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"encoding/base64"
	"math"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/rostamlabs/rostam/ops"
)

// maxKVKeyLen caps the KV key length the REST surface accepts. Keys are arbitrary
// bytes on the wire, but v1 addresses them as a URL-encoded UTF-8 path segment,
// and the get/put/del wire codecs length-prefix the key with a u16 — so an
// over-long key is a client mistake with an obvious remedy (shorten it), rejected
// 400 in-handler. The /v1/kv/ prefix is NOT covered by nameLenGuard (which only
// guards the collection/alias routes), so this cap is enforced here.
const maxKVKeyLen = 512

// maxTTLMs is the largest ttl_ms that survives the *time.Millisecond
// conversion to time.Duration (an int64 count of nanoseconds) without
// overflowing: math.MaxInt64 / int64(time.Millisecond). A value above this
// wraps to a negative or truncated duration and would silently store the
// wrong TTL instead of failing loud, so kvPut rejects it as a 400.
const maxTTLMs = int64(math.MaxInt64) / int64(time.Millisecond)

// kvKey decodes the {key} path segment. net/http's ServeMux already unescapes
// a {wildcard} match before PathValue returns it, so raw is the literal key —
// unescaping it again would merge distinct keys (a literal "a/b" and the
// encoded "a%2Fb" both decode to "a/b") and reject a valid key that merely
// contains a bare '%' (e.g. "%ZZ"). Only the length is bounded here, at
// maxKVKeyLen. Returns ok=false (after writing the 400) on an over-long key.
func kvKey(w http.ResponseWriter, r *http.Request) ([]byte, bool) {
	key := []byte(r.PathValue("key"))
	if len(key) > maxKVKeyLen {
		writeError(w, http.StatusBadRequest, "key too long")
		return nil, false
	}
	return key, true
}

// kvGetResponse is the GET /v1/kv/{key} body. ValueB64 carries the raw bytes
// (always present when found, including an empty-but-present value); ValueUTF8
// is the string form, emitted ONLY when the bytes are valid UTF-8 (omitted
// otherwise so a client never mistakes lossy bytes for text). Both are pointers
// so an empty string ("") still marshals rather than being dropped by
// omitempty — omitempty only elides a nil pointer, which is how a miss keeps
// both fields absent. TTLMs is present only when ?with_ttl=1 was requested.
type kvGetResponse struct {
	Found     bool    `json:"found"`
	ValueB64  *string `json:"value_b64,omitempty"`
	ValueUTF8 *string `json:"value_utf8,omitempty"`
	TTLMs     *int64  `json:"ttl_ms,omitempty"`
}

// isKVNotFound reports whether a dispatch error is the cache's key-miss sentinel
// (cache.ErrNotFound, "cache: not found"). Matched by message so it covers both
// the in-process path (the sentinel verbatim) and the clustered path where it is
// stringified across the Raft boundary — the KV analog of statusForError's
// string fallbacks. A miss is rendered as {"found":false}, NOT a 404.
func isKVNotFound(err error) bool {
	return strings.Contains(err.Error(), "cache: not found")
}

// kvGet reads a KV key → {found, value_b64, value_utf8?}. A key miss is
// found=false (200), not a 404, so a client distinguishes "absent" from a real
// error without inspecting status codes. With ?with_ttl=1 it also dispatches the
// ttl op and includes ttl_ms (-1 = no expiry, -2 = absent, per the ttl op's
// convention).
func (a *api) kvGet(w http.ResponseWriter, r *http.Request) {
	key, ok := kvKey(w, r)
	if !ok {
		return
	}
	args := ops.EncodeKeyArgs(key)
	// Authorize then dispatch directly (not via a.call) so a key MISS can be
	// intercepted and rendered found=false rather than written out as an error.
	if !a.authorize(w, r, "get", args) {
		return
	}
	body, err := a.disp.Call("get", args)
	var resp kvGetResponse
	switch {
	case err == nil:
		resp.Found = true
		b64 := base64.StdEncoding.EncodeToString(body)
		resp.ValueB64 = &b64
		if utf8.Valid(body) {
			utf8Val := string(body)
			resp.ValueUTF8 = &utf8Val
		}
	case isKVNotFound(err):
		resp.Found = false
	default:
		writeDispatchError(w, "get", err)
		return
	}
	if r.URL.Query().Get("with_ttl") == "1" {
		ttlBody, ok := a.call(w, r, "ttl", ops.EncodeKeyArgs(key))
		if !ok {
			return
		}
		ms, err := ops.DecodeTTLResult(ttlBody)
		if err != nil {
			writeInternalError(w, "ttl decode", err)
			return
		}
		resp.TTLMs = &ms
	}
	writeJSON(w, http.StatusOK, resp)
}

// kvPutReq is the PUT /v1/kv/{key} body. Exactly one of ValueB64 / Value must be
// set (base64-encoded raw bytes, or a UTF-8 string); TTLMs is optional (0 or
// absent = no TTL). ValueB64 and Value are pointers so "" is distinguishable from
// absent — an explicit empty value is a valid (empty) write, and the both/neither
// guard needs to know which fields were actually present.
type kvPutReq struct {
	ValueB64 *string `json:"value_b64"`
	Value    *string `json:"value"`
	TTLMs    int64   `json:"ttl_ms"`
}

// kvPut writes a KV key. The value arrives as EITHER value_b64 (raw bytes) OR
// value (a UTF-8 string) — exactly one; both or neither is a 400. An optional
// ttl_ms sets a relative expiry (0/absent = permanent). Dispatched via callWrite
// with the default (no write-consistency) path.
func (a *api) kvPut(w http.ResponseWriter, r *http.Request) {
	key, ok := kvKey(w, r)
	if !ok {
		return
	}
	var req kvPutReq
	if !decodeBody(w, r, &req) {
		return
	}
	if (req.ValueB64 == nil) == (req.Value == nil) {
		writeError(w, http.StatusBadRequest, "exactly one of value_b64 or value must be set")
		return
	}
	var val []byte
	if req.ValueB64 != nil {
		dec, err := base64.StdEncoding.DecodeString(*req.ValueB64)
		if err != nil {
			writeError(w, http.StatusBadRequest, "invalid value_b64: "+err.Error())
			return
		}
		val = dec
	} else {
		val = []byte(*req.Value)
	}
	if req.TTLMs < 0 {
		writeError(w, http.StatusBadRequest, "ttl_ms must be non-negative")
		return
	}
	if req.TTLMs > maxTTLMs {
		writeError(w, http.StatusBadRequest, "ttl_ms too large")
		return
	}
	ttl := time.Duration(req.TTLMs) * time.Millisecond
	if _, ok := a.callWrite(w, r, "put", ops.EncodePutArgs(key, val, ttl), 0, true); !ok {
		return
	}
	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}

// kvDelete removes a KV key → {"deleted":bool} (false when the key was already
// absent). Dispatched via callWrite with the default (no write-consistency) path.
func (a *api) kvDelete(w http.ResponseWriter, r *http.Request) {
	key, ok := kvKey(w, r)
	if !ok {
		return
	}
	body, ok := a.callWrite(w, r, "del", ops.EncodeKeyArgs(key), 0, true)
	if !ok {
		return
	}
	deleted := len(body) > 0 && body[0] == 1
	writeJSON(w, http.StatusOK, map[string]bool{"deleted": deleted})
}

// kvFlush wipes the ENTIRE KV keyspace → {"flushed":true}. Dispatched via callWrite
// with the default (no write-consistency) path and empty args; the flush op is
// keyless and takes no body, and in cluster mode the node fans it out to every shard
// group (cluster.broadcastFlush). It requires GLOBAL write authority (a read-only
// key gets 401), exactly like kvDelete's classification but against the empty
// (cluster) resource — see the flush routing row in sdk/wire.
func (a *api) kvFlush(w http.ResponseWriter, r *http.Request) {
	if _, ok := a.callWrite(w, r, "flush", nil, 0, true); !ok {
		return
	}
	writeJSON(w, http.StatusOK, map[string]bool{"flushed": true})
}
