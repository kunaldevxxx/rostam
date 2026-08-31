// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/rostamlabs/rostam/authz"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// TestHTTPKVFlush covers POST /v1/kv/flush: it returns 200 {"flushed":true} and
// wipes the keyspace — a key written before the flush reads back found=false after.
func TestHTTPKVFlush(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	if rec := do(t, h, "PUT", "/v1/kv/gone", `{"value":"x"}`, nil); rec.Code != http.StatusOK {
		t.Fatalf("put = %d, want 200 (%s)", rec.Code, rec.Body)
	}

	var flushed struct {
		Flushed bool `json:"flushed"`
	}
	rec := do(t, h, "POST", "/v1/kv/flush", "", &flushed)
	if rec.Code != http.StatusOK {
		t.Fatalf("flush = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if !flushed.Flushed {
		t.Fatalf("flush body = %s, want flushed:true", rec.Body)
	}

	var out kvGetResponse
	do(t, h, "GET", "/v1/kv/gone", "", &out)
	if out.Found {
		t.Fatalf("key still present after flush; want found=false")
	}
}

// TestHTTPKVFlushRBAC pins the authz bar on the REST surface: flush is a
// global-WRITE op, so a read-only key is denied 401 while a cluster-wide write:* key
// succeeds 200. This mirrors the del classification against the empty resource.
func TestHTTPKVFlushRBAC(t *testing.T) {
	keyReg, err := vector.OpenKeyRegistry(filepath.Join(t.TempDir(), "keys.json"))
	if err != nil {
		t.Fatal(err)
	}
	mustAddKey(t, keyReg, vector.APIKey{Token: "k_read", Tenant: "acme", Scopes: []string{"read:*"}})
	mustAddKey(t, keyReg, vector.APIKey{Token: "k_write", Tenant: "acme", Scopes: []string{"write:*"}})

	opsReg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(opsReg); err != nil {
		t.Fatal(err)
	}
	h, cleanup := newTestAPIOpts(t, Options{Authenticator: authz.NewRBACAuthenticator(keyReg, opsReg, "")})
	defer cleanup()

	flushReq := func(token string) int {
		r := httptest.NewRequest("POST", "/v1/kv/flush", nil)
		if token != "" {
			r.Header.Set("Authorization", "Bearer "+token)
		}
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, r)
		return rec.Code
	}

	if code := flushReq("k_read"); code != http.StatusUnauthorized {
		t.Fatalf("flush by read-only key = %d, want 401", code)
	}
	if code := flushReq(""); code != http.StatusUnauthorized {
		t.Fatalf("flush by anonymous caller = %d, want 401", code)
	}
	if code := flushReq("k_write"); code != http.StatusOK {
		t.Fatalf("flush by write:* key = %d, want 200", code)
	}
}
