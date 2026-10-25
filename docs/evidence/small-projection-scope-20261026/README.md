# Projection scope ablation after historical PR28 replay

Historical PR28 4K was reproduced at 5739.28 total tok/s, but its startup mixed-request diagnostic failed. The current correction wraps QKV, O, down and uncompressed gate projections, including rows above 8 inside an opaque operation. This experiment isolates which projections need that protection and lets the others retain the original compiled route.

Base main: `34942f7e28da5c954ab05878dec75807da9d7a93`. Default remains all. `VLLM_FL_BIV150_STABLE_SMALL_O_PROJECTIONS` accepts all, none, or a canonical subset of qkv,o,down,gate; it depends only on model projection identity, never benchmark names, requested lengths or scoring inputs. The actual canonical scope enters compile-cache identity. The BF16 kernel and CUDA implementation are unchanged.

First candidate: gate only. Real service M2 selection must equal the exact selected geometry set; unselected geometries must not enter the custom op. The owned controller keeps fixed contest commands, permits complete105 accuracy and exploratory throughput at99/105, preserves extra semantic failures as diagnostics, supports one scored length for cost control, and can repeat the mixed fixture after all warmup/scoring so cold and warm behavior are not conflated.

CPU passed:140 routing cases across five scopes and rows1/2/4/8/16/64/2048,100 fallback cases, six invalid scopes, attempted live scope changes, canonical cache separation, unchanged CUDA-math AST, unchanged owned-process cleanup AST. These are not GPU or model qualification. Existing BF16 kernel validation remains evidence in PR51/PR53, not a newly executed component result here.

GPU state: not yet launched. Plan exclusive personal-instance service diagnostics, then full105 and sufficiently warmed one-round4K/16K measurements against matched controls only for viable scopes. Keep historical and default-all results. No production change is ready to merge, and PR55 GPU work remains deferred.
