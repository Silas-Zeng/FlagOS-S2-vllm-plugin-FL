# Read-only feasibility: host metadata chooses CTA ordering

No host hook, kernel, production installer, new PR or device queue is implemented
by this study. The existing140cecc device-predicate component remains frozen.
The proposed distinction is general context/query ratio; it uses no benchmark
name, request label or change to official service/benchmark parameters.

The existing high-level API **does expose CPU integers** suitable for a
conservative decision. This is a possible route to keep the useful short-context
observation while sending long context to the original Q32 binary. It is not a
measured zero-cost dispatcher. In particular, the metadata upper bound can reduce
coverage, and the final target custom-op/context source link is still missing.

## Source provenance and arguments

A fresh fetch/API/CLI check found team main
`56f1838192abc940b3fb9accef490022ac125000`. The single-sequence Q32, normalization
and Q-fragment files are byte-identical between main56f and measured parent2412:

| File under vllm_fl/patches | SHA256 |
| --- | --- |
| biv150_prefill_q32.py | 3906f34e89b814cafa94ef04a0f268ffc3cf237b99ecb011a62c9abfef56878b |
| biv150_prefill_normalized.py | 957720a31e63fd994e096e1bf5cfce9197541506cb06d66cb71ba5614bd323a1 |
| biv150_q_fragment_service.py | 38e44e6b0272116a1c76561119e9742337caed4dfaf5ab891fc57df4f2c16a82 |

The [source manifest](evidence/prefill-reverse-20261027/refined-first-falsifier/host-route-source-manifest.json)
and [source archive](evidence/prefill-reverse-20261027/refined-first-falsifier/host-route-sources-v1.tar.gz)
retain exact main56f caller/C3 code and pinned SDK/helper/backend/dispatcher/wrapper
sources. Paths and line numbers below refer to those files. Sources from a local
official0.24 package are used only where their bytes match an existing audited
target pin; its unverified attention/context files are excluded.

| Actual module/source | SHA256 prefix | Relevant lines |
| --- | --- | --- |
| main56f vllm_fl/worker/model_runner.py | 0161c595ff2f | 792–794,1995–1997,2081–2098,2340,2428–2438,2473–2535,4232–4244,4371–4385,4420–4437 |
| main56f biv150_compute_order.py | e60909d88325 | 249–297,299–320,335–424 |
| main56f biv150_mixed_decode.py | a129361f1ef1 | 86–111,172–205,220–226 |
| SDK triton_attn.py | 1219c4d1820b | 183–224,628–650 |
| SDK triton_unified_attention.py | 54c495dcea40 | 780–815 |
| SDK cudagraph_dispatcher.py | 51f632753af7 | 132–155,235–324 |
| SDK breakable_cudagraph.py | 722683b44b44 | 310–323 |

SDK54c495 `unified_attention` receives `q,k,v,out,cu_seqlens_q,max_seqlen_q,
seqused_k,max_seqlen_k,...`. Its raw kernel launch omits the two maximum-length
arguments, so a raw `_Policy` cannot recover them without another interface.
The existing high-level function is the natural metadata-visible seam.
Backend1219c4 `forward` assigns these maximum arguments from
`attn_metadata.max_query_len/max_seq_len` at628–631 and passes them at648/650.
The builder copies the common metadata Python values at191/193 and218/220.

The actual runner computes Qmax from the current scheduler's CPU list/NumPy array
at4232–4234, before C3's order permutation. Kmax is the maximum of
`optimistic_seq_lens_cpu.numpy()[:num_reqs]` at2340; the buffer is created without
a device argument at792–794, and its `.numpy()` CPU view is the one used here.
The final `.item()` is a NumPy scalar conversion, **not GPU `.item()`**.
The CPU buffer is refreshed with current computed-token plus scheduled-token
counts at2093–2098. Capture intentionally uses `max_model_len` instead at2334–2338;
capture metadata must not be mistaken for a current ordinary prefill.

## What the two maxima prove

Retain every existing single-sequence Q32/BF16/causal/geometry/source eligibility
check. In the trusted normal service path, one remaining prefill sequence has
exact query length Qmax. Removing an initial decode prefix removes only lengths1,
so it cannot remove the maximum when Qmax>=1024. C3 merely permutes requests,
leaving both maxima invariant. Its plan uses its actual CPU scheduled order for
the query cumulative sums; it also provides the corresponding CPU seq metadata.

Kmax is **a batch maximum/upper bound**, not necessarily this suffix's KV length.
`mixed_decode.py`200–204 slices seq_lens/block_table and rebases cu_q, but does not
replace max_seq_len. For example, a prefill with query2048/own KV4096 can inherit
Kmax16384 from a long decode in the same batch. A host decision must then keep the
original binary. It must not call16384 the prefill's exact context or claim the
same hit range as the device predicate.

As a conditional rule, if the exact single query length is Q>=1024, its actual
KV length is K, and the already-provided CPU bound U satisfies K<=U, then
`Q<=U<=5*Q` implies `K-Q<=4*Q`. The existing valid-sequence contract supplies
K>=Q. Thus this rule admits only the intended short-context side; an overestimate
can produce a conservative miss. No actual KV tensor value is read on the CPU.
For long context K>5*Q, U>=K necessarily rejects reversal and can call the
unchanged parent Q32 Manager/binary directly. Host-call overhead and any coverage
miss still need measurement; using the original device binary alone does not
establish unchanged whole-service latency.

The zero query start is a **source provenance fact**, not an inference from
q.shape or cu_q.shape. Runner2081 writes CPU cu_q[0]=0;2082 copies current CPU
cumulative scheduled lengths, then2086 transfers this metadata. For the mixed
suffix, `_prefix` checks CPU offsets[0]=0 and exact positive lengths, and every
removed prefix length is1 (mixed86–111). Therefore cu_q[prefix]=prefix, and the
actual assignment `cu_q[prefix:count+1]-prefix` at194 yields zero. Q/output are
sliced at the same prefix boundary at224–225. C3's stable partition and
`compute_order.scheduled` feed the same cumulative construction (runner1995–1997).
Arbitrary raw SDK callers, unsupported nonzero starts, padded queries or a changed
caller chain are not admitted by this service-source argument. A future hook must
preserve/verify that provenance or fall back; shape checks alone are insufficient.

## Graph and per-call metadata boundary

For the fixed service `FULL_DECODE_ONLY`, the existing C3 cold audit requires
only uniform=True FULL keys, an empty PIECEWISE key set and uniform decode length1
(compute_order.py176–208). Runner3899–3916 returns `uniform_decode=False` for
ordinary Qmax>1, and3964–3975 passes that value to the exact pinned dispatcher.
Dispatcher132–155 preserves uniform=False in the descriptor;302–324 cannot
match a uniform=True FULL key or an absent PIECEWISE key, so returns NONE.
This uses actual initialized key invariants, not a guess from a batch tensor shape.

The actual breakable wrapper returns `self.runnable(*args,**kwargs)` for NONE
at322–323. Runner4371–4385 builds metadata on each execute call; its
`cached_attn_metadata` dictionary at2473 is local to that call and only reuses
same-step metadata across compatible groups. Runner4420–4437 installs the current
metadata/context and calls the model. C3 `check_dispatch` rejects a changed mode,
padding, microbatching or DP coordination after preparing the private compute order.

The remaining gap is the **target** custom-op/context implementation that bridges
a compiled model back to `self.impl.forward` with the current per-call metadata:

- `vllm/model_executor/layers/attention/attention.py`, especially
  `get_attention_context` and `unified_attention_with_output`.
- `vllm/forward_context.py`, especially `get_forward_context` and
  `set_forward_context`.

Local official0.24 copies suggest the expected mechanism, but their presence or
package version is not evidence that target source bytes/bindings match. Obtain
the actual files and SHA256 before closing this part of the feasibility proof.
Until then, fresh metadata at the runner and NONE wrapper is proven; the full
compiled/custom-op per-call chain is deliberately not declared complete.

Both identity and unconditional reversal are whole-grid bijections with the same
logical work/math/addresses. For a fixed valid grid/ABI, capturing either ordering
and then changing valid metadata cannot by itself change numerical results; a
stale host choice can still violate the intended performance selection. That
mathematical statement does not replace actual service Graph/binding validation.
The existing13-state device-predicate replay screen does not test a new host hook.

Any later integration needs a separately reviewed source identity/cache-fingerprint
update and preservation of existing normalization/Q32/C3 audits. A new top-level
wrapper cannot silently bypass their strict expected bindings. The required next
deliverable here is source feasibility only: no host route is installed, no
kernel changes are made, and no new service/GPU work is queued.
