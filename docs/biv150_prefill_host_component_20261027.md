# Explicit CPU-metadata host-routing component

This executable prototype adds only tools/docs. It installs no service hook,
imports no Torch/Triton, performs no JIT/SSH/device work and creates no GPU queue.
Only the old unconditional reverse kernel and this CPU protocol have evidence;
the new host component's device numerical/performance and model gates remain
**unverified**. Device work waits for instance identity verification and root's
explicit scheduling. The two missing target custom-op/context sources in the
[feasibility study](biv150_prefill_host_route_feasibility_20261027.md) remain missing.

## Boundary and complete contract

`tools/biv150_prefill_host_component.py` consumes explicit CPU Qmax/U(Kmax), CPU
cu_q, query identity, a fresh invocation stamp and an explicit component scope.
It never infers CPU metadata or q_start from a GPU tensor shape. Scope must state
plain **unencoded BF16** page16/KV-heads2/head128 packing, NONE runtime, empty hooks,
no async compiler mode and a coherent immutable CPU/GPU metadata snapshot for the
launch. A dtype/shape-compatible encoded V cache is not this format. Missing or
unknown scope/packing does not enable the route. These are caller obligations,
not an automatic verifier of a live framework or GPU payload.

The component boundary is **already-parent-selected Q32**, B256/Q32/w16/K64/s1.
The SDK initially calls the main M2/Q32 parent policy with BM128/Q16/w8. Sending
canonical B256 arguments back into that top-level policy can bypass its eligibility
and lose the Q-hoist control. This prototype does not do that. Its fallback is
the exact caller-owned `Call` at the selected Q32 boundary. The main M2 outer
object is retained as an opaque parent-chain identity. Multi-sequence/raw-SDK
calls remain their original calls; the prototype neither replaces nor installs
anything in that parent chain.

After checking the exact original descriptor guard and grid, select reversal only
when CPU cu_q is exactly(0,Q), Q>=1024 equals the current query's entire effective
single-sequence row count, and Q<=U<=5Q. CPU integers must be real Python integers,
with int32 metadata representability. Query identity, call stamp and parent-chain
identity must match. Unsupported padding/nonzero starts, bad types, absent
provenance, capture, multiple/zero sequences, ir_override and geometry changes
all return the **original Call object identity**, with the same entry, args,
grid, kwargs dictionary and Tensor references. No dictionary contents are edited.
Python's eventual `**kwargs` call creates the callee's keyword dictionary normally;
the original record and every value reference remain intact.

For Q2048/U16384 the ratio fails before the descriptor guard or reverse entry is
touched. Running that original Call invokes the original Q32 path/binary. There
is no new GPU predicate, metadata load kernel, warmup, compile or reverse launch
on that path. This is a structural guarantee of the router, **not zero host cost**.
An explicit full-batch upper bound can miss a short prefill: own K4096/Q2048 with
U16384 keeps the original call. No benchmark label is consulted; for example,
Q4096/U16384 and fresh Q16384/U16384 both pass the same ratio rule.

`cpu_metadata(...)` is an explicit one-shot context, not a service metadata hook.
Missing/stale/used/wrong-call metadata fails closed. A nested context blocks the
inner route and restores the outer context. A selection stored beyond its context
falls back before reverse entry access. `Router.invalidate()` revokes qualification,
including a selected but not yet executed reverse call. An executed selection
cannot be retried. Parent/reverse launch exceptions propagate unchanged with no
automatic alternate launch. CPU snapshot coherence and actual GPU contents still
need a device falsifier; a caller falsely declaring fresh data cannot be detected
by this CPU protocol.

## Setup, source reuse and execution interface

`prepare_source(sdk_text,helper_text,source_root)` uses the unchanged140 factory's
`mode='always'` to reconstruct the previously measured fca source, SHA256
`38198704c37479e4b5c26778d7dd1da2a894409d1c81ff2626e1d988facca104`.
It guards SDK54c495/helper59d9d3, normalization957720, fragment38e44e and the
unchanged factory. Exact byte/AST inverse restores the original normalized kernel.
Only the original CTA-ID assignment changes; no new device kernel algorithm,
device predicate, floating math, K/Q casts, softmax, reducer, geometry or warps are
introduced. The source contains zero added route-metadata loads.

Preparation and CPU checks may read/hash/parse sources. The hot selector does
not inspect/bind Python signatures, read files, run Git/hash, call GPU `.item()`
or compare Tensor values. It reuses the exact parent descriptor-only guard.
Launchers must be supplied explicitly by a future qualified component harness:

```python
# parent_entry is the original already-selected Q32 launcher;
# parent_chain is the unchanged outer M2/Q32 chain, retained as identity.
scope = ComponentScope(BOUNDARY, PACKING, "NONE", True, True, True, True,
                       parent_chain, parent_entry, reverse_entry)
router = Router(prepared_source, scope)
original = Call(parent_entry, original_args, original_grid, original_kwargs,
                parent_chain)
cpu = CpuMetadata(original.stamp, original_kwargs["query_ptr"], Qmax, U,
                  explicit_cpu_cu_q, True, PROVENANCE, False)
with cpu_metadata(cpu):
    result = router.run(original)
```

The explicit flags in this example require real component qualification; they
are not permission to assert unknown framework state. No real launcher binding
or service integration is supplied here, and no new device-run CLI is exposed.

## Reproducible CPU evidence

```bash
python3 -B tools/check_biv150_prefill_host_component_cpu.py \
  --source "$PWD" \
  --sdk-source /path/to/pinned/triton_unified_attention.py \
  --helpers-source /path/to/pinned/triton_attention_helpers.py \
  --out /tmp/cjj-prefill-host-component-new-output
```

The output directory must be new; old evidence is never overwritten. The
[committed CPU evidence](evidence/prefill-reverse-20261027/host-component-v1/cpu.json)
contains13 boundary cases and45 rejection cases, plus stale/missing/nested scope,
revocation/expired selection, unchanged payloads, Tensor scalar/equality traps,
exception propagation and no hot I/O/hash/Git/signature binding checks. It executes
the actual main56f Q32 and M2 class AST with CPU launchers: raw BM128 is upgraded
by the original parent, short context chooses reverse, long context chooses its
original Q32, and513+513 still reaches M2's full grid34x2. It does not import or
install the framework. [Manifest](evidence/prefill-reverse-20261027/host-component-v1/manifest.json)
records source/output hashes. Generated source is the old kernel; CPU success
does not establish GPU typed ABI, Q-hoist retention, resources, performance,
service dispatch correctness or full105 precision for this host component.
