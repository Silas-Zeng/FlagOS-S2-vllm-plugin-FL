# Isolated untied LM-head service candidate — unvalidated

Base is freshly fetched team main 3283672, not the rejected residual gate
candidate. The audited native source/loader/build helper is copied from
20c991a without calling its hidden-layer install function. Only the separate
UnquantizedEmbeddingMethod hook for a single ParallelLMHead is installed.
All hidden layers, input embedding, scheduler, official serving parameters
and benchmark workloads retain the team baseline behavior.

The full 130560 x 2048 output weight is compressed to per-output-column INT8
weights and scales. An expanded scalar preserves public weight shape/dtype,
with no retained full BF16 output weight. Native linear handles all sampling
row counts; no request-length or benchmark-case selection occurs. Require the
exact audited MiniCPM geometry, untied embeddings, TP1, BF16 inputs and no bias.
Unsupported inputs fail closed because no original BF16 head remains.

The synthetic full-vocabulary primitive probe 7c1ba7d passes seven cases and
changed-input/zero graph replay. Median speedups are 1.59–1.69x, but dense
relative RMSE is about 1.2% instead of BF16's 0.17%. Actual hidden-state outliers
and model accuracy remain unknown. These measurements do not validate this
hook or predict end-to-end speedup.

The isolated hook checker uses actual SDK method/classes with minimal temporary
modules and a mocked config only. It checks unchanged input embedding, exactly
two bytes of marker storage, eight M cases including zero and pitched inputs,
CPU INT64 sampled references, fullgraph at M8/M32 and changed-input/zero graph
replay. It does not instantiate or validate a complete model.

The serial driver privately builds the SO, requires all eight hook checks,
then runs a new full 105-question Level-3 evaluation before warmed single-round
4K/16K measurements. It records exact code/SO hashes, actual head installation,
exactly one weight conversion and post-ready serving selection evidence, and
enforces the working request/throughput/TTFT gates. Quantization accuracy cannot
be copied from another branch. Record actual KV capacity from the service log.

Local syntax, Ruff and diff checks pass. No GPU hook or model result is available
yet. The predecessor-aware queue will run only after the current combination
acceptance and the independent prefill diagnosis, with GPU/port idle checks.
