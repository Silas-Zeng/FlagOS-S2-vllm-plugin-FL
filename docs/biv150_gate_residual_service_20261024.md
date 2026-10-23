# Small-M residual activation service candidate — unvalidated

Base: team main `3283672`, gate-only candidate `5b88eac`, residual operator `01bdf10`.
The gate-only parent's Level-3 score was 99/105, below 0.95. This candidate is
intended to test whether small-M activation error contributes to that failure.
It has no model precision or throughput result yet and must not be merged.

`gate_silu_auto` selects the already micro-tested residual operator for compressed
weights and M=1..16. Other M uses the unchanged native gate/SwiGLU implementation;
real original BF16 weights retain their original fallback. The model invokes the
single opaque native op, so shape dispatch occurs in C++, before the existing
GEMM and fused epilogue. No scheduler or official service/benchmark parameter changes.
Only interior gate/up layers are quantized, as in the parent. QKV/O/down and edge
layers retain BF16. This is distinct from the historical `beb3518` configuration.

The residual **primitive** previously passed 24 numerical cases, CPU INT64 sampled
references, graph changed-input replay, and fullgraph/Fake Tensor tests. The new
`gate_silu_auto` dispatch and model integration have only passed local Python lint,
syntax and whitespace checks; C++ build and GPU/model checks remain unexecuted.
Do not transfer primitive results or a different configuration's accuracy to this one.

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

At the latest local check, both the original SSH hostname and previously resolved
IP rejected the existing instance key with `Permission denied (publickey)`.
Remote outcomes are unknown. No new remote test has been started for this candidate.
