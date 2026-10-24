# Latest-team compressed residual gate: complete acceptance passed

Runtime source e3a0497adbd6570dd522877131bc47058350da83, based on team main
1ea082f79db65a66f78fdd7b1d958335450fa9f4 (PR22+PR23). Personal BI-V150 only.

| Workload | Total tok/s | vs original reference | Mean TTFT ms | Requests |
|---|---:|---:|---:|---:|
|4K|5750.700805|+19.5731% vs4809.36|3649.061884|256/256,0 failed|
|16K|4002.302967|+18.9735% vs3364.03|125347.042290|128/128,0 failed|

Math500 Level3:102/105 (0.9714),105 unique,0 failed predictions. Correct token
totals:4K1048576/262144,16K2097152/131072. Adjacent warmup TPOT stabilized:
4K113.802706/101.390309/101.393715ms;16K285.638091/280.142193ms. Each formal
workload ran once after warmup, per the user's research protocol. These are not
four-round competition submission averages. Full JSON and commands are archived.

Both workload request, accuracy, throughput-floor and working TTFT gates passed.
Neither reached the separate+20% target5771.232/4036.836. Against the teammate's
reported PR23 numbers5313.59/3764.67, absolute differences are about+8.23%/+6.31%
but involve different-time runs and protocols; do not claim isolated causality.

Actual service evidence includes compressed gate-only middle34 layers,
small-M1..16 activation residual, native SO SHA256
0c03563cc9bed9c06791372a20c5a0037cffa58760d2d330533331b85675dc50,
KV547536 tokens; compute-order selected after readiness with N62/M2048,
Dprefix2/Dall61, original sampler row order; Q-fragment selected override
8ae22585… with zero recorded fallback and loop Q reads0. All fixed serve options
and benchmark workload parameters are in manifest.json. Scheduling algorithm,
limits and framework dependencies remain unchanged by this feature.

## Reproduce on the audited personal environment

Use Python3.12, PyTorch2.10/Corex4.5, FlagGems5.3.5, pinned vLLM0.24 competition
stack. Build once before using the existing official serve command:

    MAX_JOBS=2 python tools/build_biv150_native_w8a8.py

The private output is build-native-w8a8/biv150_native_w8a8.so, ignored by Git.
The installer resolves it relative to the imported source root; PYTHONPATH must
select that same checkout. No new serve options are required. On an eligible
BI-V150 MiniCPM model the build is required; a missing SO fails with a build hint.
Other hardware/model combinations retain the existing eligibility behavior.
No binary artifact or global package installation is committed.

Owned full research acceptance entry:

    python tools/run_biv150_gate_team_queue.py --source "$PWD" --out /absolute/new/output

It refuses busy GPU/port/dirty source, builds privately, checks9 numerical/Graph
cases, evaluates105 questions, then warms and runs the two fixed workloads.
Do not run concurrently with another GPU acceptance. Keep output paths unique.
The report/evidence commit adds no runtime changes to the measured source.
