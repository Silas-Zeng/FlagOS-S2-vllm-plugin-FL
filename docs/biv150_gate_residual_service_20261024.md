# Small-M residual activation service — 16K performance rejected

Base: team main `3283672`, gate-only candidate `5b88eac`, residual operator `01bdf10`.
The gate-only parent's Level-3 score was 99/105, below 0.95. This candidate tests
whether small-M activation error contributes to that failure. The exact 20c991a
candidate now passes one complete Level-3 run: 102/105 (reported 0.9714), 105
unique questions and zero failed predictions. The completed warmed runs improve 4K but regress 16K throughput and exceed
the 16K working TTFT limit. Do not merge this candidate.

`gate_silu_auto` selects the already micro-tested residual operator for compressed
weights and M=1..16. Other M uses the unchanged native gate/SwiGLU implementation;
real original BF16 weights retain their original fallback. The model invokes the
single opaque native op, so shape dispatch occurs in C++, before the existing
GEMM and fused epilogue. No scheduler or official service/benchmark parameter changes.
Only interior gate/up layers are quantized, as in the parent. QKV/O/down and edge
layers retain BF16. This is distinct from the historical `beb3518` configuration.

The residual **primitive** previously passed 24 numerical cases, CPU INT64 sampled
references, graph changed-input replay, and fullgraph/Fake Tensor tests. The new
`gate_silu_auto` dispatch was originally unvalidated at a0287ef. On 2026-10-24,
20c991a was independently built and passed all nine M=0/1/8/16/17/32/40/64/2048
dispatch cases, including sampled CPU references, M8/M32 fullgraph and
changed-input graph replay, and unchanged original-weight BF16 fallback. Do not
transfer these results to a changed source/library or another configuration.

## Resume validation

First inspect current processes, GPU occupancy, ports, worktrees and team main.
Do not restart earlier mixed-attention acceptance or the BF16 fusion queue merely
because SSH observation failed. Retrieve their terminal state and results first.
Build in this candidate's own ignored `build-native-w8a8` directory, using the
existing `tools/build_biv150_native_w8a8.py`; never reuse the parent's old SO for
the new op. Run `tools/check_biv150_gate_auto.py --library ... --out ...`.
This checks M=0/1/8/16/17/32/40/64/2048, pitched input, exact selected-route output,
independent sampled CPU references for small M, graph replay with changed input,
fullgraph at M8/M32, and original-weight BF16 fallback.

After the operator checks pass, use `tools/run_biv150_native_gate_only_acceptance.py`
with `--mode all` and a new output directory. Its install metadata gate requires
`gate_silu_auto` and residual row interval [1,16], records source/SO hashes, tests
all 105 unique Level-3 questions with zero failed predictions, and only proceeds
to stabilized warmup and one scored round of each official workload if precision
is at least 0.95. Evaluate both throughput and TTFT before any merge.

## Current evidence

Personal-instance source:
/workspace/experiments/minicpm-s2/code/gate-residual-service-20261024,
exact commit 20c991ac06f2228a5e9aae2cfdb0d265c769d075, clean worktree.
Outputs: /workspace/experiments/minicpm-s2/gate-residual-service-20261024.
Library SHA256: cf966eda1802da08bfdc07cc0b9be1e2688f80518fbbd519a6ce10a16949c562.
The manifest verifies actual gate_silu_auto installation, [1,16] residual rows,
interior 34 gate/up layers only, with unchanged QKV/O/down/edge-layer BF16.
Reported GPU KV capacity is 547632 tokens. Official serving parameters are
unchanged. Complete accuracy passed at approximately 08:58 CST.

The final two 4K warmups have TPOT 108.8195 / 108.8643 ms, meeting the existing
stability gate. The scored workloads are still running as of 09:03 CST. No
throughput number is accepted from these warmups. The gate-only parent's 99/105
and this candidate's 102/105 are distinct one-run observations, not proof of
statistical stability or of a general accuracy gain.

Raw evidence is stored under docs/evidence/biv150-gate-residual-service-20261024.

## 4K scored result recovered at 09:07 CST

The stable-warmed single 4K run completed 256/256 requests with zero failures,
1048576 input and 262144 output tokens, duration 249.264267 seconds:
5258.354991 total tokens/s, TTFT 4244.359266 ms, TPOT 56.390478 ms.
This is +9.3359% against the original 4809.36 throughput reference, below +20%.
The working TTFT limit is 4770.3209 ms. The previously accepted integration
measurement was 4873.331261; the current single observation is +7.9006% versus
that earlier measurement, not a same-session paired comparison.

These scores belong to 20c991a, not the historical beb3518 combination.


## Completed 16K rejection, 09:26 CST

The stable-warmed single 16K run completed 128/128 with zero failures,
2097152 input and 131072 output tokens, duration 715.636150 seconds.
Throughput 3113.626945 total tokens/s is -7.4435% versus original 3364.03,
below its 99% floor 3330.3897. Mean TTFT 161138.989054 ms exceeds the
working 156938.85 ms ceiling. Mean TPOT is 154.943382 ms. Two preceding
warmup TPOTs were 340.076810 and 339.921964 ms (difference about 0.046%).

The runner's acceptance.json records completed measurements and a precision
gate; it does not enforce throughput/TTFT limits. Its completed=true is not
full acceptance. Both workload outputs and the manifest are archived here.
The owned server was cleaned by its own runner; the next independent probe
started serially. This branch remains experimental, excluded from main.

Small-M residual compensation passes precision but has not solved the
long-context performance regression. Do not assume its cause from these
whole-run measurements alone, or copy this accuracy to a changed operator.
