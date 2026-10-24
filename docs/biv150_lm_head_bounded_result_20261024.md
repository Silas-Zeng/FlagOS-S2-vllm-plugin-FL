# Bounded INT32 head: primitive passes, no service KV recovery

Actual source d5f5f581d7af4a5c0f34331381ac313a20e0ac30. Private build passed;
13/13 primitive and14/14 SDK hook checks passed, including changed-input Graph
and CPU integer references. Evidence JSON is archived with this report.

| M | Old Torch peak MiB | Bounded peak MiB | Speedup (old/new) |
|---|---:|---:|---:|
|8|18.024|18.024|1.002|
|32|24.094|24.094|0.999|
|64|48.188|48.188|0.999|
|128|96.375|96.375|0.998|
|129|97.129|65.128|0.680|
|256|192.751|128.750|0.725|
|2048|1542.008|582.000|0.633|

Torch peaks include the output, not just the bounded INT32 tile, and exclude
non-Torch vendor allocations. Each timing ratio is the median of3 alternating
Graph rounds. Small serving M retains the same partition and numerical path;
large M saves temporary allocation but pays extra GEMM launches.

The real service profile again logged M256. Its KV capacity was **504688** tokens,
compared with the old head's504784 and earlier BF16 baseline522960. Reducing the
M256 Torch peak64MiB did **not** recover service KV. The true bottleneck in the
full memory budget remains unproven; this does not establish vendor workspace
as the cause. No memory accounting or scheduler limits were changed.

The fixed benchmark uses at most64 requests, so its head M<=64 computation was
unchanged. Given no capacity recovery and the already failed full old-head
acceptance, continuing a repeat105-question/two-workload run was not prioritized.
At16:34:35 the agent deliberately terminated only the identity-checked own
accuracy child56467; driver53482 naturally entered finally and cleaned its own
server. All identities, reason and action are in early-stop.json. This is an
intentional incomplete model validation, **not** an engine failure, accuracy pass
or measured throughput result. Preserve the operator for future investigation;
do not merge it as a validated service improvement.

After the owned server was gone and GPU/port were free, the existing GQA wave
probe automatically resumed; the latest-team compressed-gate acceptance followed.
