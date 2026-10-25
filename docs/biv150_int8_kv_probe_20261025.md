# INT8 token/head KV + fused decode consumer — diagnostic draft

Team main0915f01 was rechecked before this independent branch. No service hook, model cache allocation, backend dtype change, serving flag or benchmark change is installed here. This is an original quantize/write kernel and an original INT8-consuming extension of the shipping GQA kernel, not enabling an installed FP8/INT8 path.

The completed main profile makes attention worth targeting: the16K sampled Graph's GQA producer is44.10ms out of54.51ms including observation overhead; whole-run nonGraph prefill is209.50s and nonGraph GQA42.86s. These are measured windows, not quantities that can all be added or extrapolated into a speedup. The invalid physical-bandwidth counter interpretation is not used.

## Representation and operator

Packed cache is `[pages,2,16,2,128]`, INT8 with a separate contiguous FP32 scale tensor `[pages,2,16,2]`. Each token/head/key-or-value128-vector is encoded with its own symmetric maxabs scale and nearest-away-from-zero integer rounding, clipped to[-127,127]. Zero vectors have scale1; tiny scales use the FP32 min-normal floor. Inputs are finite BF16. Total encoded storage is132/256=0.515625 of BF16, a theoretical48.4375% saving before allocator effects; this does not establish reclaimed model KV capacity.

`append_token_kv` fuses per-vector quantization with writes at physical slots; negative/out-of-range padded slots return without writes. The consumer directly loads INT8 and FP32 scales, reconstructs BF16 operands in registers and retains the shipping online-softmax/segment layout. There is no paged gather or materialized BF16 dequantized cache. It is approximate quantization and requires the full105-question precision gate after actual model integration.

## Bounded probe

The probe covers actual B64/4K and B32/16K shapes plus B8/ragged257 including empty, single-token, page-boundary and partial-page cases. It preserves packed NHD element strides, shuffled unique page maps,8192-wide table stride and narrowed16-wide builder scratch, with the baseline4K compiled hash required to match641478e6. Both BF16 and INT8 caches are deliberately retained for comparison; no capacity gain is claimed.

CPU FP64 oracles are separate: original BF16 and independently reconstructed quantized operands. The latter gates consumer correctness; the former reports intended quantization error without calling a component tolerance model accuracy. A second changed Q/K/V pattern must reproduce eager from poisoned Graph output. Ragged checks cover invalid-slot no-write and pointwise quantization error bounds. Timing includes the experimental writer plus producer and unchanged reducer in three alternating rounds; attention-only timing is separate. The BF16 writer control is also experimental and is not claimed to measure vLLM's writer performance.

Current status: Python syntax and inert CPU planning path pass. No GPU compilation, numeric result, throughput result, full accuracy or actual model-cache integration has run. GPU remains reserved for the Q32 paired full-workload-warmup pipeline. This direction must not be merged as a serving improvement on component evidence alone.

Future integration must implement a real cache specification/allocation including scales, a compatible self-developed prefill consumer and normal model-wide fallback ownership. A decode-only INT8 cache cannot serve prefill or be declared a memory saving while retaining the BF16 copy. Keep fixed competition serving parameters; do not switch an existing quantization hyperparameter.

## Initial GPU result, sourcef339109

All three cases compile and complete. Quantized-operand CPU FP64 gates, changed-input poisoned Graph checks, positive finite scales,[-127,127] representation, invalid-slot no-write and ragged pointwise error bounds pass. This confirms the intended approximate representation/consumer, not equality to BF16 or model accuracy; ragged BF16-reference error reaches0.03125 and exceeds the diagnostic0.005+0.005r tolerance at1109 values in one pattern. Full model accuracy is untested.

The actual4K baseline compiled hash is641478e6. Writer+producer+reducer median speedups are0.85776x for ragged,0.70993x for B64/4K,0.70514x for B32/16K. Registers increase from93 to162, shared remains8192; use_sme changes14→206 with the new input types/arguments. n_spills=0 is not proof of no SGPR spills. No serving integration or performance gain is claimed. Raw reports: docs/evidence/int8-kv-probe-20261025/.

One bounded follow-up moves V-scale load and INT8→FP32→BF16 V reconstruction after QK/softmax, preserving the arithmetic and representation. The aim is to shorten simultaneous K/V temporary lifetimes; register/timing effects must be measured, not assumed. It has syntax-only validation at this commit. Related baseline correctness work PR48 now reports native-on mixed failure; this direct GQA component does not load that gate library, and future model integration must use a validated baseline.
