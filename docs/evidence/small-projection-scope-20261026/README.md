# Projection scope ablation after historical PR28 replay

Historical PR28 4K was reproduced at 5739.28 total tok/s, but its startup mixed-request diagnostic failed. The current correction wraps QKV, O, down and uncompressed gate projections, including rows above 8 inside an opaque operation. This experiment isolates which projections need that protection and lets the others retain the original compiled route.

Base main: `34942f7e28da5c954ab05878dec75807da9d7a93`. Default remains all. `VLLM_FL_BIV150_STABLE_SMALL_O_PROJECTIONS` accepts all, none, or a canonical subset of qkv,o,down,gate; it depends only on model projection identity, never benchmark names, requested lengths or scoring inputs. The actual canonical scope enters compile-cache identity. The BF16 kernel and CUDA implementation are unchanged.

First candidate: gate only. Real service M2 selection must equal the exact selected geometry set; unselected geometries must not enter the custom op. The owned controller keeps fixed contest commands, permits complete105 accuracy and exploratory throughput at99/105, preserves extra semantic failures as diagnostics, supports one scored length for cost control, and can repeat the mixed fixture after all warmup/scoring so cold and warm behavior are not conflated.

CPU passed:140 routing cases across five scopes and rows1/2/4/8/16/64/2048,100 fallback cases, six invalid scopes, attempted live scope changes, canonical cache separation, unchanged CUDA-math AST, unchanged owned-process cleanup AST. These are not GPU or model qualification. Existing BF16 kernel validation remains evidence in PR51/PR53, not a newly executed component result here.

GPU state: not yet launched. Plan exclusive personal-instance service diagnostics, then full105 and sufficiently warmed one-round4K/16K measurements against matched controls only for viable scopes. Keep historical and default-all results. No production change is ready to merge, and PR55 GPU work remains deferred.

## First live service observation

Runtime remains frozen at47e1405; this documentation commit does not hot-edit it. Worker install and M2 selection confirm gate-only. Generated AST Call nodes show nine extern MM sites and two custom-op sites, excluding comment/reference strings; these are static sites, not runtime counts or timings. The initial mixed fixture still fails the long request while counting succeeds. Complete105 is active (33 prediction rows at04:47 Shanghai); no final precision or throughput result is available. The current candidate is diagnostic until its outstanding gates are resolved. Post-score mixed checking is enabled.

## Partial105 and4K update

Runtime47e1405:99/105=.9429,105 unique predictions and metric count, zero failed_predictions. Precision gate fails. Diagnostic4K:5311.820237520383 total tok/s, TTFT3705.551362952974ms,256 successes/zero failures. Three short warmups plus full warm231.8356s preceded scoring, but26 new Triton kernel metadata JSONs appeared during the score (including MM/normalization/index variants). This is not a compilation-free steady-state comparison or an isolated regression estimate.16K and the post-score mixed check remain in progress; do not close the performance question or merge based on this partial run.
