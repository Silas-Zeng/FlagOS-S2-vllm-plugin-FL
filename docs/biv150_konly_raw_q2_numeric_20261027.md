# Private raw3D max-q2 extension

This separate frontend leaves a432 and all kernel/codec/gate bytes unchanged.
It clones only the pinned host function, with exact byte inverse: replace the
all-q1 workspace check by a token-indexed query/segment check and save a source
sidecar. A private copy of the descriptor validator changes only max_q<=1 to
max_q<=2, with exact inverse; the original validator still rejects q2. Every
other dtype/stride/format/alias guard remains. No global monkeypatch is installed.

Two cases keep64 sequences, mixed qlen1/2, totalQ96,16 segments and the original
raw3D source directory/reducer. They include8/24 entered-but-causal-empty
token/segment rows respectively. Independent pinned prefix/mask enumeration
proves token coverage, query-indexed workspace addresses and causal key coverage.
This is private reader coverage, not a formal batchTokens or SDK dispatch change.

In this BM16/BQ2/max-q2 geometry each sequence fits one query block and its last
query sees the whole KV sequence. Every physically entered segment therefore
runs at least one tile. For an earlier query whose last tile is entirely masked,
the actual59d9 softmax_step replaces m_j=-inf by0, produces P=0 and L=0, and
the accumulator remains0. The scoped check requires M=0/L=0/O=0 for this row;
it does not incorrectly require L>0. Nonempty rows require finite workspace and
positive sums. Physically inactive segments stay NaN. A zero-iteration M=-inf,
initial-L1 region is excluded by this particular query/loop geometry; it is not
generally rejected as mathematically invalid or silently licensed here.

The same actual-source/ABI/options/IR/ASM, full encoder bytes, independent
global FP64 original/quantized references, full finite outputs, canaries,
read-only guards and strict baseline/implementation gates from a432 still run.
Unknown new specialization is exported before stopping. Original BF16 proxy
failure may continue only with explicit numeric diagnostic. All96 output rows
are referenced on both patterns because each per-sequence qlen<=2.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_biv150_konly_raw_q2_numeric_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/bench_biv150_konly_raw_q2_numeric.py
# Root only, fresh output after exclusive device admission:
PYTHONDONTWRITEBYTECODE=1 python3 tools/bench_biv150_konly_raw_q2_numeric.py \
  --run-gpu --numeric-diagnostic --commit EXACT_NEW_FRONTEND_COMMIT --out FRESH_OUT \
  --source-root /workspace/experiments/minicpm-s2/cjj-konly-raw3d-compile-v1-20261027/sources
```

Default imports no Torch/Triton/NumPy. No local SSH/JIT/GPU, queue, Graph,
timing, deployment or actual q2 numerical pass is claimed. The explicit GPU
mode retains whole-job native-host-PID/source admission and no foreign fallback.
