# PR53 reopened: isolate residual JIT before rejecting or claiming a gain

The original full pair remains an observed negative result: runtime97d5d479bdf03567859085a787a408e7bd16eeea, candidate102/105 with real mixed/component checks passed,4K5253.616573 and16K4051.084796; same-source linear4K5626.933347 and16K4103.124533. Its precision is sufficient, but the scored4K kernel additions differ:24 new MM kernels for candidate versus0 for control (four index additions each). Historical PR28 replay had3 MM+4 index, and gate-only has17 MM+7 index+2 normalization. Apply the same limitation to high and low scores.

These counts invalidate a claim of isolated steady-state performance; they do not prove JIT explains the entire difference. The original negative report/archive is retained. PR53 was reopened as Draft for a controlled reassessment, not as a positive optimization.

## Compiler observer prototype

`tools/biv150_compile_observer.py` is import-inert and opt-in. Install via a private future service bootstrap before JIT binders capture the compile callable. It pins the installed runtime/jit.py SHA6e9ce85c877b84807d97191211e53fbc7547328dd22dc6b8d9185c7bafc67210 and compiler/compiler.py SHAf9646f23963160af7856e6f725df7af01c2bf0fc0ee98afeb7987f552507e5fd. The inspected binder imports triton.compiler.compile into self.compile, so the observer validates and wraps its package/implementation aliases before binding. It records host start/end intervals and kernel names without tensors, CUDA calls or synchronization. Original returns/arguments/exceptions are preserved; record failures produce a bounded stderr marker, and installation failure restores aliases.

CPU tests pass return identity, kwargs, original exceptions, late binding, the explicit already-bound limitation, aliases, failure rollback, overlapping interval unions and corrupt evidence rejection. These are not real service or GPU evidence. Existing bound instances are not retrofitted, and restoring aliases does not rewrite already-bound references. Use a fresh unobserved service for final acceptance.

Summary consumes explicit monotonic windows and reports clipped interval unions plus sums/by-kernel counts. A host compile call can include a binary-cache load; it is not necessarily code generation. Client-process windows are wider than the official benchmark timer. Concurrent compilation may overlap other work; never subtract measured seconds to manufacture a hypothetical TPS. Validate record coverage against kernel additions and reject logger-error markers. Diagnostic logging benchmarks are not formal scores.

## Next controlled rerun

Preserve original runtime97d/source, library5cbb, service/official workload arguments and model. Both arms need documented mature caches, matching warmup criteria, and actual compile/autotune evidence; do not choose the highest repeat. Current SDK source shows MM uses exact M/N/K keys by default while linear uses align32 (power-of-two below32); this is a plausible source of different cache coverage, not proven performance causality. SDK TRITON_PRINT_AUTOTUNING can report its existing miss bench_time during diagnostics. The new observer has not been installed in any running service yet. PR57 currently owns GPU and PR58 is already queued afterward; no PR53 GPU job is queued.
