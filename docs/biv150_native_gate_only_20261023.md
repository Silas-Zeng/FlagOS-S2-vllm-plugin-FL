# BI-V150 native gate/up-only fusion candidate

Base: verified team main3283672. Codea638560, local build-ignore ruleffd9977.
Status: compiled successfully; full precision/throughput acceptance running.

The previous W8A8 QKV/O/gate-up candidate improved4K but regressed16K. New
shape probes show that native M32 QKV/O can be modestly slower than FlagGems
BF16, while fused gate/up remains faster. This candidate quantizes only interior
gate/up weights and fuses the native INT8 GEMM with BF16 rounding and SwiGLU.
QKV/O/down and first/last4 layers remain BF16. The C++/CUDA kernels are unchanged
from the prior audited native candidate; the new experiment isolates layer scope.
This does not prove that QKV/O alone caused the prior8% regression.

Build using `python tools/build_biv150_native_w8a8.py`; artifacts stay in ignored
build-native-w8a8/. No site-packages or dependencies are modified. The installer
checks the pinned model geometry and the worker restricts installation to Iluvatar.
Default selected layers are fixed to gate_up_proj in code, not enabled by serving
quantization flags. Official serve/benchmark/scheduler limits remain unchanged.

Acceptance runs from clean runtime commitffd9977 with manifest hashes for source
and the locally compiled library. Installed metadata confirms gate_up_proj only,
compressed weights, fused activation, skip_edge_layers4. Startup reports547536 KV
tokens: this differs from both BF16 baseline522960 and prior native545616. Do not
assume reducing quantized layers made the cache capacity equal to the baseline;
workspace/profile allocation also needs consideration. Record actual score/logs.

Driver97292, engine97438 are historical startup identifiers. Check liveness before
any new GPU action. Root:/workspace/experiments/minicpm-s2/native-gate-only-20261023/full;
driver log:/workspace/experiments/minicpm-s2/native-gate-only-acceptance-driver.log.
The runner requires full105-question Level3 accuracy>=0.95, then stable warmup
and one official scored round per4K/16K case. No model throughput gain or recovery
of the prior10.6% is claimed before the complete scores and TTFT gates pass.

## Full precision failed: do not integrate

All105 unique questions completed with zero failed requests, but only99 correct:
reported score0.9429, below0.95. The runner raised the accuracy gate, skipped
throughput, and cleaned its service. There are no valid4K/16K throughput scores
for this gate-only version. Do not reuse the old all-projection candidate's4K
score as evidence for this configuration. Keep the PR a draft, not merged.
Raw verdict:docs/evidence/native-gate-only-20261023/full-model-accuracy.json.
