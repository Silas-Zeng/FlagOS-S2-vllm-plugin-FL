# Multi-page fused BF16 GQA decode — unvalidated operator probe

Framework source is team main 3283672. The branch starts from ae09fc6 to reuse
already shared diagnostic utilities, without importing newer upstream framework
code. The shipping GQA source is pinned by SHA256
1a81409ba23a149081f7fc257460ea1512cbfdceeb0af32a2698994958749b9a.

The old shipping loop visits one logical 16-token KV page per online-softmax
iteration. This probe groups 2/4/8 logical pages into a 32/64/128-token tile.
Physical pages remain randomly permuted; only the list of page addresses is
loaded and broadcast. There is no contiguous-physical-page assumption, score
materialization, quantization, model approximation or scheduler change.

Segment boundaries are rounded to the grouped tile width; the unchanged stable
partial reduction receives the same width to identify active segments correctly.
Tail page-table reads are masked by the real KV length before dereferencing.
The potential benefit is fewer online-softmax updates and different matrix tile
utilization; the risks are larger shared memory/register pressure and lower
occupancy. Nothing is installed into serving by this probe.

The grid covers 3 group sizes, 3 warp counts, 2 pipeline stage settings and 6
batch/KV geometries. Inputs use actual service-like packed K/V strides (8192,
256,128,1), arbitrary physical pages, ragged lengths, empty sequences and partial
pages. Both comparator and candidate are checked against sampled independent CPU
FP64 attention; conversion to double happens only after CPU transfer. Three
alternating timing rounds include producer AND partial reduction.

Selected B8 execution tests changed-input Graph replay with changing Q, KV lengths
and page tables, plus NaN-poisoned scratch and all-empty lengths. It preserves
the original 0.005 absolute/relative numerical tolerance. A successful probe is
still not a model precision or throughput result.

CPU checks completed: Ruff, Python bytecode compilation, generated-source Python
syntax for all three group sizes, and 225 segment/tail-page geometries proving
complete non-overlapping token coverage and valid last-page bounds. These checks
do not prove GPU compilation, numerical correctness or performance.

The separate queue waits on an owned predecessor's PID and start_ticks; it then
requires free GPU and closed serving port. It never kills/restarts a service.
