# Bounded INT32 workspace for the full-vocabulary head

Parent source319dc3 repairs actual SDK startup and passes one full Level-3 run
at101/105, 105 unique questions, zero failures. Its throughput is still running.
The service saves266864640 bytes of output weights but reports only504784 KV
tokens, below the BF16 baseline522960. This does not establish the cause, but
the existing native head materializes M x130560 INT32 accumulators: at M2048,
that temporary alone is1069547520 bytes, on top of the BF16 output.

Add a separate linear_bounded operator, preserving the old linear, gate and
residual public operators. Only the output head selects the new op. Its fixed
64MiB accumulator budget gives at most128 rows per INT32 result at N130560,
or66846720 bytes. The full BF16 output remains allocated; this does not claim
64MiB for the whole operator. Native/vendor workspace, allocator rounding and
graph pools must be measured separately.

The private implementation partitions only M, leaving K, weights, row scales,
quantization, INT32 reduction and BF16 epilogue unchanged. Inputs at M<=128 use
the same partition as before. Larger M uses more GEMM calls, so peak allocation
and latency must both be measured. There is no benchmark-label branch, model
parameter change, hidden-layer quantization or scheduler/serve-argument change.

CPU geometry checks cover32772 M/budget combinations (M1..8193 and budgets
16/32/64/128MiB), proving each padded INT32 tile fits its budget. This proves
neither GPU correctness nor actual peak-memory behavior.

The serial driver privately builds the new library, then requires13 full-vocab
primitive comparisons against the unchanged original operator, sampled CPU
INT64 references and invalid-budget rejection. It measures alternating graph
timings and live peak allocation, requiring a substantial peak reduction at
M2048 before service work. Next,14 SDK hook cases cover zero, pitched input,
65/127/128/129 row boundaries,256 and2048; M8/M32/M129 include fullgraph and
changed-input replay. The full model then reruns all105 precision questions,
stable warmup and one complete official benchmark round per workload.

Install metadata and the acceptance manifest require linear_bounded plus the
64MiB budget, as well as exactly one compressed output weight. Record actual
KV capacity and pass both throughput and TTFT gates before any merge. Parent
accuracy, synthetic speedup and CPU geometry cannot be copied to this version.

Local Python syntax, Ruff and diff checks pass. Native compilation, GPU tests,
new model accuracy, KV-capacity recovery and serving performance are pending.
The queue waits after the retained-BF16 gate candidate; all GPU work is serial.
Team main was freshly fetched and remains3283672, with no new upstream merge.
