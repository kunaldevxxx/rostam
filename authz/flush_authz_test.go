// SPDX-License-Identifier: Apache-2.0

package authz

import (
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// flush wipes the ENTIRE KV keyspace. It is registered OpReadWrite (no special
// admin entry), so it inherits del's classification: ActionWrite against the EMPTY
// (cluster) resource. That deliberately requires GLOBAL write authority — write:* /
// *:* — because write authority already implies deleting any key and flush is only
// the O(1) form. These tests pin that: a read-only key is denied, a
// collection-scoped write:docs is denied (resource mismatch), and only a
// cluster-wide write grants it. No authz code change backs this; it is pure default
// classification.

func newOpsRegForFlush(t *testing.T) *ops.Registry {
	t.Helper()
	r := ops.NewRegistry()
	if err := ops.RegisterBuiltins(r); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	return r
}

func TestActionForFlushIsWrite(t *testing.T) {
	reg := newOpsRegForFlush(t)
	// Precondition: flush is a registered OpReadWrite op...
	if _, kind, _, ok := reg.Lookup("flush"); !ok || kind != ops.OpReadWrite {
		t.Fatalf("precondition: flush must be registered OpReadWrite (ok=%v kind=%v)", ok, kind)
	}
	// ...and is NOT in the admin allowlist, so it classifies as write (unlike
	// __register_wasm__, which overrides its OpReadWrite registration to admin).
	if got := actionFor("flush", reg); got != ActionWrite {
		t.Fatalf("actionFor(flush)=%q want %q", got, ActionWrite)
	}
}

func TestResourceForFlushIsEmpty(t *testing.T) {
	// flush is keyless / not collection-scoped, so it targets the empty cluster
	// resource — which only a "*" pattern (write:* / *:*) can cover.
	if got := resourceFor("flush", nil); got != "" {
		t.Fatalf("resourceFor(flush)=%q want empty (cluster resource)", got)
	}
}

func TestRBACFlushRequiresGlobalWrite(t *testing.T) {
	opsReg := newOpsRegForFlush(t)
	reg := newRegWithKeys(t,
		vector.APIKey{Token: "reader", Tenant: "acme", Scopes: []string{"read:*"}},
		vector.APIKey{Token: "coll-writer", Tenant: "acme", Scopes: []string{"write:docs"}},
		vector.APIKey{Token: "coll-writer-ns", Tenant: "acme", Scopes: []string{"write:acme/docs"}},
		vector.APIKey{Token: "writer", Tenant: "acme", Scopes: []string{"write:*"}},
		vector.APIKey{Token: "super", Tenant: "acme", Scopes: []string{"*:*"}},
	)
	auth := NewRBACAuthenticator(reg, opsReg, "INTERNAL-SVC-TOKEN")

	cases := []struct {
		token string
		want  bool
		desc  string
	}{
		{"reader", false, "a read-only key is denied a write op"},
		{"coll-writer", false, "collection-scoped write:docs does not cover the empty cluster resource"},
		{"coll-writer-ns", false, "namespace-scoped write:acme/docs does not cover the empty cluster resource"},
		{"writer", true, "cluster-wide write:* covers the empty resource"},
		{"super", true, "*:* covers everything"},
	}
	for _, c := range cases {
		got := auth(AuthRequest{Token: c.token, Op: "flush", Args: nil})
		if got != c.want {
			t.Errorf("authorize(%s, flush)=%v want %v — %s", c.token, got, c.want, c.desc)
		}
	}
}
