# Visible decode diagnosis (not official scores)

Source baseline9a43eb1 / nativebeb3518; official service defaults unchanged in
production. Diagnostic alone uses CUDA graph NONE and torch profiler (delay300,
8 iterations, recorded shapes). Direct32 parallel completions:15360 token IDs,
512 output tokens, ignore_eos, temperature1/top_p.95. No preliminary test prompt.
The initial 16K version refused insufficient KV capacity and cleaned up without
profiling; this version fits full workload in the measured KV capacity.

Both traces confirm 8×42=336 attention calls with actual Q shape[32,16,128] and
no prefill, 32 requests running. Requests complete32/32 and service groups exit.
Raw files and manifests remain at:
/workspace/experiments/minicpm-s2/decode-visible-15k-20261023.

Baseline GPU kernel cumulative durations422.581ms, GQA326.962ms (77.4%),
BF16 GEMM68.135ms (16.1%). Native414.858ms, GQA322.863ms, remaining BF16
GEMM28.853ms, INT8 GEMM27.237ms, row quantization5.628ms, plus epilogues.
Sampled total difference is only1.83%, not a new full-model throughput result.
Graph-NONE CPU overhead and this shortened diagnostic input do not reproduce
full official graph-mode latency; no acceptance inference is made.

Baseline serving GEMMs appear as mm_kernel whereas native vendor GEMM symbols
are different. Earlier isolated F.linear microbenchmarks must not be treated as
identical serving dispatch without verifying their actual kernels. The earlier
activation API-versus-compiled mismatch was already recorded separately.

Loop-pipeline probe: 36 checks over batch64/32/31/40/8/1, uniform/mixed
lengths, shuffled pages, empty rows and partial pages. Every output bit-exact;
short rows additionally use independent FP32 attention. Replacing while with a
for loop allows different pipelining, but increases shared memory from8192 to
10240 bytes (stages2/3 add registers). No large scored-shape benefit; only
batch8/length1024 shows about6.6% improvement. No production patch is adopted.
