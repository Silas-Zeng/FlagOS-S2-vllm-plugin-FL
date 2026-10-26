# BI-V150 GQA bounded software exp2 component experiment

Status: closed after no observed component gain for this exact degree5,
five-FMA **probability-only** candidate. `precision_gate=false`,
`official_score=false`. Frozen tested commit:
`40e14f37081edaa88527b7ef6ac6a31c68c8e940`; team production baseline:
`241220723a09061c182764f8f738e5ec01190367`.

The completed BI-V150 component run produced median baseline/candidate speedups
below one in both shapes. The candidate used 94 registers versus 93 for baseline,
with zero reported spills and 8192B shared memory for both. These small slowdowns
are descriptive observations, without an A/A significance claim.

| Actual packed component | Segments | Median speedup | Implied candidate latency change |
| --- | --- | --- | --- |
| 4K, batch64 | 2 | 0.9952833856780836x | +0.4739% |
| 16K, batch32 | 4 | 0.9936934017515131x | +0.6347% |

The primitive passed 33,159 inputs with maximum relative error
`3.303807943390251e-7`, including exact integer powers. The two raw GQA records
retain `passed=true`, but their device-FP64 RMS diagnostics are inconsistent:
amplitude1 comparisons report relative RMS zero while max absolute differences
are nonzero (4K: 0.00048828125 / ragged 0.000244140625; 16K: 0.000244140625 /
ragged 0.0001220703125). Therefore the original pass flags do **not** establish
full-output numerical equivalence. The code performs those reductions on GPU
`.double()` tensors; no obvious baseline/clone alias was found, and the precise
statistics failure remains unproved. Changed-input graph RMS uses the same
unvalidated path and cannot independently qualify graph numerical equivalence.

The primitive's host reference, 24 sampled CPU-double attention-reference checks,
uniform/side-stream flags and original graph records remain archived unchanged.
No full output tensors were saved, so the historical full-output RMS cannot be
recomputed on CPU from these artifacts. Nothing was rerun to repair old metrics.
Future diagnostics must copy tensors to CPU before float64 RMS and enforce that
RMS is zero exactly when host max_abs is zero.

No `both`-scope GPU run, full 105-question model precision, official service
throughput or TTFT acceptance was performed. This closure stops only the tested
five-FMA probability replacement, without drawing conclusions about other
native/software mixtures, other SFU mechanisms or FA4 scheduling/pipelining.
Production kernels, SDKs, dependencies and official launch/benchmark parameters
remain unchanged. Archiving used only reads and CPU packaging of the terminated
run; it started no GPU work and did not touch another experiment's source/cache.

The original hypothesis was to replace probability exp2 work in the existing
page16 GQA decode loop with five FP32 FMAs. The factory also contains an untested
`both` scope, but it was not part of the executed result. The original while loop,
Q hoist, segment layout, FP32 sums, BF16 probability-to-V conversion and reducer
are preserved. Component timing is not a service score.

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

## Historical probability component command

Use the designated Python environment and the frozen PR commit on an idle
BI-V150. The harness refuses a live port9031 service, a non-idle ixsmi report,
a changed SDK JIT hash, dirty tracked source, production-source differences, or
a baseline kernel hash unlike the current model's packed allocation. It does
not stop anything. The executed scope was probability only. Its frozen command was:

```bash
python3 tools/bench_biv150_gqa_software_exp.py --source "$PWD" \
  --commit 40e14f37081edaa88527b7ef6ac6a31c68c8e940 --exp-scope probability \
  --out /tmp/gqa-software-exp-probability
```

The primitive checks over 33,000 actual-domain inputs precede attention timing;
normal-domain relative error must be <=2e-6, integers exact, tails zero and invalid
positive/NaN inputs visibly NaN. The GQA cases are packed 4K/B64 and 16K/B32,
with physical block-table stride8192 and narrowed segment scratch. The original
model baseline hashes are asserted for both. The harness records code artifacts,
registers, spills, launch metadata and three alternating graph-timed main+reduce
pairs per shape. Numerical checks include uniform exactness, random Q at scales
1 and 8, ragged/zero lengths, a CPU-double dense reference, changed-input CUDA
graph replay and side-stream determinism. Failed numerical checks skip timing.
Approximate random-output comparisons use reported relative RMSE thresholds;
they are component screening criteria, not the model precision gate.

`probability` retains native rescale exp2 and replaces probability exp2 only;
`both` replaces all four audited full/tail call sites. No runtime select computes
native and software exponentials together. The final helper `where` only handles
the declared numerical domain and flush tail. Compiler codegen and BI-V150
resources must still establish whether the FMA path is beneficial.

The raw result is retained as historical evidence with the numerical
qualifications above. This closed candidate is not being promoted to a service
run or merge.

## Terminal evidence archive

[Final scoped verdict](evidence/gqa-software-exp-20261027/terminal/final-verdict.json),
[unaltered raw result](evidence/gqa-software-exp-20261027/terminal/result-original.json)
and [download/member verification](evidence/gqa-software-exp-20261027/terminal/verification.json)
are directly reviewable. The terminal folder also contains the complete gzip tar,
its per-file manifest and archive index. The 301,578B single package is below
2MiB and contains 41 members: 40 payload files plus its manifest. It preserves
all 30 actual primitive/baseline/candidate source/IR/ASM/cubin resources,
generated source, source proof, raw result, owner, log, input Git bundle and four
frozen source files; `__pycache__` and `.pyc` files are excluded.

Archive SHA256:
`865564eba5859b0073963e7b04adaf89e88f3eaaa8a6aeec53b319d38e0cfd8b`.
The archive, external manifest and index each matched their remote SHA256 after
download. All 40 payload hashes, all 30 original result resource hashes, four
frozen source blobs and the original input-bundle hash were verified locally.
Owner PID602351/start340484724 was observed in terminal state Z. See the terminal
[README](evidence/gqa-software-exp-20261027/terminal/README.md) for verification.
