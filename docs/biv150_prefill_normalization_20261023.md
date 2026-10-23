# BI-V150 prefill normalization candidate

Integration base: team main9a43eb1, checked through GitHub API after a direct
Git fetch timed out. No newer official framework or FlagGems version is merged.
This branch changes only an audited BF16 attention operator and its registration.
It is a correctness fix with full precision and warmed throughput acceptance.
It is not a +20% performance result; measured throughput differences are small.

## Observed correctness discrepancy

An independent dense FP32 reference revealed a short causal prefill discrepancy
in the pinned kernel: qlen16/KV128, max output difference0.30078125. A private
instrumented clone reproduced the original output exactly; score masks match,
score error1.43e-6 and row maximum error7.15e-7, but the denominator differs by
up to1.0000114. One row reports7.0556555 versus reference6.0556555.

Initializing cumulative mass to zero changes that row to6.0556555 and reduces
the denominator error to1.14e-5. The exact compiler/lowering cause is not claimed:
a standalone first-update helper, both plain and MMA-backed, behaves correctly.
The evidence is a full-kernel behavior discrepancy, not proof of a particular
vendor compiler defect. Zero initial probability mass is appropriate when there
are no sinks; sink attention retains the original implementation and initial mass.

## Implementation boundary

The patch checks the exact existing attention/helper source hashes and clones
the pinned kernel in memory, changing only cumulative-sum initialization (plus
an identifying kernel name). Python linecache exposes the trusted generated
source to Triton without editing the installed vLLM files or upgrading packages.
The existing Python launch wrapper is cloned with only its kernel reference
changed. Registration retains the existing mixed-attention and GQA decode stack.
Pure decode and unsupported geometry/features fall back to that existing stack.

Eligible scope: Iluvatar BF16 Q16/KV2/head128/page16 causal prefill; no sinks,
ALiBi, QQ-bias, PrefixLM, KV quantization, softcap, sliding window or tensor
descriptors. No scheduling policy, model weights, generation settings, benchmark
load or official service argument is changed.

## Validation

Eight independent dense FP32 comparisons cover single/multiple sequences,
empty rows, mixed query lengths1/16/40, partial pages, qlen2048 and KV up to16384.
All pass combined atol=.005/rtol=.005 after conversion to the BF16 output dtype.
The maximum absolute difference can exceed .005 at large BF16 magnitudes; the
combined relative-plus-absolute criterion is used. The original kernel has
large short-context discrepancies; the zero-mass clone removes them.

The standalone operator time remains essentially unchanged (approximately±1%).
These checks do not prove model accuracy or end-to-end performance. Full105-item
Level3 evaluation and real-serving selection evidence are required before any
model-accuracy claim. The registered-stack checker passed all eight cases with the existing decode
launch/GQA hooks active. Sink fallback remained bit-exact and did not select
the normalized kernel. Full-model Level3 evaluation completed on code commit4f00fcb: accuracy0.9714
(102/105), all105 questions unique, no failed predictions, and actual-serving
normalization selection confirmed. This single stochastic evaluation passes the
0.95 gate; it is not evidence of a statistically established accuracy improvement.
Formal throughput/TTFT acceptance remains pending.

## Reproduce

Use the competition environment and this checkout via PYTHONPATH; no native
build or dependency changes are needed. Run only when GPU and port9031 are free.

```bash
PYTHONPATH="$PWD" VLLM_PLUGINS=fl python tools/check_biv150_prefill_normalization.py \
  --source vllm_fl/patches/biv150_prefill_normalized.py --out /tmp/prefill-check-new.json
python tools/run_biv150_normalization_accuracy.py \
  --source "$PWD" --out /workspace/experiments/minicpm-s2/normalization-accuracy-new
```

The accuracy runner starts the exact official FULL_DECODE_ONLY service, verifies
installation, runs the full Level3 protocol (temperature1/top_p.95/max32768,
concurrency8), requires105 unique questions and no failed predictions, and checks
that normalization was selected for actual evaluation shapes. It cleans only
its own service process group. The default mode produces no throughput scores. Use --mode throughput for
two to four warmups per case (adjacent TPOT change below3%), followed by exactly
one scored full run each:4K/1024/concurrency64/256 requests and
16K/1024/concurrency64/128 requests, seed0.

Formal4K/16K acceptance still requires sufficient warmup followed by one complete
scored round per case, as explicitly authorized by the user. Throughput and TTFT
must satisfy the competition gates; the overall optimization goal remains+20%
in both scenarios. Current evidence does not establish that goal.


## Formal serving acceptance

Production source is unchanged from the precision-evaluated4f00fcb. The
throughput runner used541850b with the exact official service and benchmark
parameters. Both full cases ran once after independent seeds in warmup; all
requests succeeded. The service owned by the runner was cleaned after completion.

| Case | Total tok/s | TTFT ms | TPOT ms | Requests | Recent baseline tok/s | Difference |
|---|---:|---:|---:|---:|---:|---:|
|4K|4873.33126|4655.34015|60.78266|256/256|4835.56124|+0.78%|
|16K|3368.31020|155275.46155|133.15901|128/128|3366.81795|+0.044%|

Warmup TPOT sequences:4K146.37362/123.10459/123.10483ms;
16K364.15547/364.23437ms. Final adjacent changes are below3%.
Recent baseline TTFT was4730.537ms /155342.488ms, so neither case violates
the +1% TTFT gate. Precision is102/105 (0.9714), with105 unique predictions
and no errors. All individual acceptance gates pass against the recorded baseline;
the overall +20% target remains unachieved. One score per case does not establish
a statistically significant throughput gain. This patch is retained for the
controlled-reference correctness repair, not advertised as a performance win.

Raw scores, warmup records and service/source manifest are in docs/evidence/
throughput-{scores,warmups,manifest}.json. Full logs remain on the owned instance
at /workspace/experiments/minicpm-s2/prefill-normalization-20261023/.

Final review60de355: Ruff0.14 check/format passes on the three new Python files.
Production AST is identical to the scored version apart from standard-library
import ordering; arithmetic, gates and generated kernel source are unchanged.
The checker binds its input tensors explicitly. All eight independent FP32 cases
and bit-exact sink fallback passed again on the final source.
Fork CI jobs are all skipped by repository conditions; these are not counted as
passed CI. Actual hardware/model acceptance is documented above.
