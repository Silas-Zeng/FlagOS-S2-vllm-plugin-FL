# BI-V150 GQA bounded software exp2 component experiment

Status: Draft; CPU falsifier only. `precision_gate=false`, `official_score=false`.
No GPU, service, SSH, dependency, SDK, `vllm_fl`, or `csrc` change has been made.
Production baseline is team main `241220723a09061c182764f8f738e5ec01190367`.

The experiment asks whether replacing the probability exp2 work of the existing
page16 GQA decode loop with five FP32 FMAs is beneficial on BI-V150. This has not
been demonstrated. The factory has `probability` and `both` scopes (the latter
also replaces the two online rescale calls), and preserves the original while
loop, Q hoist, segment layout, FP32 sums, BF16 probability-to-V conversion, and
original reducer. It is a falsifiable component candidate, not a service gain.

## Source and numerical audit

KDA's software-exp suggestion was traced to primary FlashAttention-4 material:
[Tri Dao's FA4 article](https://tridao.me/blog/2026/flash4/) describes mixing native
MUFU with polynomial work. The official
[softmax implementation](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/softmax.py)
uses compile-time fragment selection; it does not evaluate both paths behind a
runtime `where`. Its
[utility implementation](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/utils.py)
uses Sollya polynomials on [0,1] and NVIDIA-specific rounding/bit operations.
Those hardware mechanisms and B200 performance conclusions are not assumed for
BI-V150. No official upstream code or SDK is merged into the competition branch.

The independent candidate uses `x <= 0`, `safe=max(x,-126)` **before** converting
to signed int32 with round toward zero, and `r=safe-int(safe)` in (-1,0]. A degree5
Chebyshev-node interpolant of 2^r, constrained to c0=1 and rounded to binary32,
is evaluated using five explicit `tl.fma` calls. The scale is constructed as
`((n+127)<<23)` and bitcast to FP32. Thus no floor, ldexp, or native exp is needed
inside the helper. Values below -126 explicitly return zero, with absolute error
strictly below 2^-126 (relative error is 1). Zero returns exactly one and -inf
returns zero. Positive inputs and NaN return NaN to expose a domain violation.

The candidate **must not be combined with conditional-rescale threshold 8**:
that changes the probability input domain to include positive numbers. The
factory pins the complete original GQA SHA256 and requires a new audit if it
changes. Exact reverse substitution recovers the original GQA function bytes.

The CPU checker re-derives coefficients at 90 Decimal digits using radical
forms for six Chebyshev nodes; the reference uses 65-digit Decimal exp, and
`fmaf` supplies actual single-rounding FP32 FMA. It checks dense reduced and
actual-domain samples, deterministic random FP32 bit patterns, every integer
boundary and its FP32 neighbors, huge negatives and -inf. Evidence is in
[evidence/gqa-software-exp-20261027/cpu-result.json](evidence/gqa-software-exp-20261027/cpu-result.json).
The observed maximum relative error is approximately 3.304e-7 on 98,688 normal
inputs. A conservative interpolation/constraint/coefficient/Horner rounding
bound is approximately 1.468e-6 relative on [-126,0], under the recorded FP32 FMA
assumptions. This bound does not substitute for device numerical validation.

A Taylor4 polynomial on the candidate reduced interval [-1,0] has approximately
0.0023867 maximum relative error; the independent calculation does not support
an unqualified 22-bit-accuracy claim for that Taylor approximation.

## Reproduce CPU checks

```bash
python3 tools/biv150_software_exp2.py --source . \
  --out /tmp/gqa-software-exp-cpu-result.json
```

Device primitive checks, packed model-shape GQA correctness/resources/timing,
CUDA graph and side-stream behavior are pending. Root coordinates all GPU runs
serially. Even if component tests pass, BF16 math changes still require complete
105-question model precision and unchanged official service throughput/TTFT
acceptance before any production integration or merge.
