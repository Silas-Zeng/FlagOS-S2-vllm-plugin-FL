# BI-V150 MiniCPM native linear and low-shared GQA candidate

Team integration base: `9a43eb1e883a1ba986a4ae8052f1d00d8fd7aa5f`.
No newer official framework branch is merged. Framework scheduling and the
official serving arguments are unchanged. This change is scoped to BI-V150,
BF16 MiniCPM5-2B, TP=1, with audited model dimensions and attention sources.

## Build and run

Use the competition Python/COREX environment. Build the native operator once:

```bash
VLLM_VENDOR= python -m pip install --no-deps --no-build-isolation -e .
MAX_JOBS=2 python tools/build_biv150_native_w8a8.py
```

The library is written to `build-native-w8a8/biv150_native_w8a8.so` and loaded from
that path by default. Do not reuse an unrelated binary after changing the C++/CUDA
source. A missing library produces an actionable build error. No quantization
enablement flag or new serving argument is needed:

```bash
export VLLM_PLUGINS=fl
vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

Install/use this plugin checkout in the existing competition environment before
launching, and verify the worker's imported plugin path. Development worktrees
do not automatically replace an already installed package.

## Operator changes

The native linear operator quantizes activations per row, performs vendor INT8
GEMM, and converts INT32 accumulators to BF16. The row quantizer uses 64-lane
shuffles; the epilogue uses a 2D grid instead of per-element integer division.
Small batches are zero-padded to 32 rows. The fused gate/up epilogue reproduces
BF16 projection rounding before FP32 SiLU/multiply and avoids materializing the
full intermediate activation.

The first/last four layers and every down projection remain BF16. Middle-layer
QKV/O/gate-up projections retain INT8 weights; a scalar broadcast marker preserves
the public BF16 weight shape without retaining a duplicate full weight allocation.
Other geometries retain the original linear method. The fused MLP entry point
is guarded, retains down-projection dispatch, and requires TP=1.

The GQA kernel delays V loading until after score/softmax work, reducing shared
memory from 8 KiB to 4 KiB. A 384-program threshold selects power-of-two segment
widths near 512 actual programs. Batches below 16 retain the original kernel;
the stride-aware reduction and empty-row handling remain in place. Changing
segment widths changes FP32 reduction order, so full model accuracy is required.

## Validation status

The preceding native-linear configuration (before the low-shared GQA change)
completed the official Level-3 protocol at **0.9524 (100/105)**, with 105 unique
questions and no failed requests. After sufficient warmup, its single full-size
4K run measured **5318.13 total tok/s**, mean TTFT **3978.19 ms**. The fresh
same-machine baseline measured **4835.56 / 3366.82 total tok/s** for 4K/16K.
The original reference was **4809.36 / 3364.03**.

Thus the preceding 4K result is **+10.6% versus the original reference**, not the
requested +20%. Its 16K score was not completed before moving the GPU to further
operator experiments. No incomplete case is treated as a valid score.

The final default configuration, including low-shared GQA, **failed** its new
complete Level-3 evaluation at **0.9429 (99/105)**, with 105 unique questions and
no failed requests. Its throughput stage was not run. This candidate remains a
draft and must not be merged or treated as competition-ready.

The official sampling settings are temperature 1.0 and top_p 0.95. The preceding
native configuration passed once by only one question; this second observation
shows that accuracy stability requires further investigation. It does not prove
the GQA change caused the difference: the new GQA policy keeps the old kernel
for batches below 16, while the accuracy evaluation has concurrency 8.

The exact `beb3518` configuration completed its missing warmed 16K run at
05:56 CST: **3094.24 total tok/s**, TTFT **160217.19 ms**, 128/128 successful
requests. This is **−8.02% throughput** and **+3.11% TTFT** versus the original
16K reference (3364.03 tok/s, 155385.04 ms), failing the performance and TTFT
gates. Its two warmups took 226.73/198.53 s; adjacent TPOT changed 2.56%, within
the declared 3% threshold. No extra scored rounds were run.

Thus neither candidate is ready for integration, and the requested +20% in
both cases was not reached. The earlier 4K gain does not establish an overall
improvement. The failed default-candidate accuracy and the preceding candidate's
16K regression are preserved separately. Saved manifests, verdicts and score
summaries are under `docs/evidence/biv150-20261023/`.

Kernel checks cover batch boundaries 1/8/16/24/31/32/40/48/56/64, shuffled physical
pages, unequal lengths, empty graph rows and partial pages. Short rows also use
an independent FP32 attention reference. The low-shared policy improves uniform
primary shapes approximately 4–5%; mixed/padded shapes can improve more.

The initial 2.36x gate/up microbenchmark compared against the FlagGems dispatch
API. The actual serving graph uses an Inductor-generated activation kernel, so
that number is not a compiled-serving speedup. End-to-end results are authoritative.

## Reproduction tools

```bash
python tools/bench_biv150_native_w8a8.py \
  --source vllm_fl/patches/biv150_native_w8a8.py \
  --library build-native-w8a8/biv150_native_w8a8.so --compressed --min-rows 17
python tools/bench_biv150_native_silu.py \
  --library build-native-w8a8/biv150_native_w8a8.so
```

`tools/run_biv150_operator_arm.py` starts and cleans only its own service process
group, refuses an occupied port, records source/binary hashes, and validates the
full 105-question report and failed-request count. Its warmup uses different
prompt seeds from scoring; at least two runs must reach TPOT stability within 3%.
Each scored case retains the official input/output lengths, concurrency and
request count. The single-round protocol follows the user's explicit instruction
and is not the original four-run/last-three average.

To reproduce the default candidate acceptance on an idle GPU and free port 9031:

```bash
python tools/run_biv150_operator_arm.py \
  --source "$PWD" --out /workspace/experiments/minicpm-s2/operator-acceptance-new \
  --library "$PWD/build-native-w8a8/biv150_native_w8a8.so" \
  --shipping-default --mode full --accuracy-first
```

Use a new output directory. The dataset must already be available at
`/workspace/evalscope-datasets/math_500`. The runner refuses an occupied service
port and does not terminate a pre-existing service. Check the saved manifest,
`accuracy-verdict.json`, warmup JSON and `scores.json` together.

Raw evidence and rejected experiments remain on the separate
`codex/20261023-operator-fastpaths` branch and the owner's instance under
`/workspace/experiments/minicpm-s2/operator-fastpaths-20261023/`.
