# Native BF16 GQA batch coverage: CPU-only candidate

Base: fresh fetched team main `4229b48b64c7bd21f9f8fc3c5ff8775500f8d02b`
(PR94 evidence-only merge). Production bytes remain exact source main
`b3b87d2a1c13448bd8c6bc24e8045c0458b8b9dd`, which the factory pins explicitly.
This branch leaves every `vllm_fl`/shipping `csrc` file unchanged. It does not
change PR91 or PR94, packages, service flags, the allocator or any GPU queue.

Actual b3 `_pick_segments` (target256,2KV,builder16) yields:

| N | S | Existing native | New diagnostic |
|---:|---:|---|---|
|32|4|native S4|unchanged|
|40|16|TL S16|**not covered**|
|48|4|TL S4|private S4 fixture candidate|
|56|4|TL S4|private S4 fixture candidate|
|64|2|native S2|unchanged|

Evidence: `biv150_gqa_decode.py:340-348` explicitly preserves N40/S16;
`biv150_native_gqa.py:26-28` admits only N64/S2 and N32/S4. In
`native_gqa_s2.cu:150-182`, seq is `blockIdx.x`, Q/table and O/M/L use runtime
int64 strides. The S4 launcher at203-214 already uses `dim3(batch,2,4)`.
The C++ S4 wrapper at45-67 is the fixed-N32 boundary. Thus N48/56 only need new
host scope/row checks, not producer arithmetic. CUDA source is copied by frozen
Git object with three isolated symbol renames; reversing them restores b3 bytes.
The new C++ operator is `cjj_native_gqa_batch_s4_20261028::produce`. Original
32/64 operators are untouched. The extra renamed S2 export is unregistered and
unused; it avoids global symbol collision without changing its body.

The runtime defaults off, imports no framework and installs no service hook.
It only admits a `StockFixture` that creates its own fresh BF16 allocation and
copies literal finite CPU BF16 values. No API admits existing CUDA cache objects
or a claimed format string. Exact key/value object+descriptor ownership rejects
external/encoded nominal-BF16 aliases, private flags and missing receipts. The
runtime also checks typed scalar ABI, dense/padded Q, dense/narrowed scratch,
nonaliasing, storage bounds and finite scale. After a native launch starts,
errors propagate without a fallback retry.

CPU checks cover actual `_pick_segments`, exact source inverses, typed/ownership
mutants and all per-lane O/M/L addresses for both supported N and scratch pitches.
No Torch/Triton import, compiler, SSH or device operation was performed:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_native_gqa_batch_cpu.py
python3 -B tools/bench_biv150_native_gqa_batch.py
python3 -B tools/build_biv150_native_gqa_batch.py
```

Future Root-only commands (not executed here; no queue is created):

```sh
python3 -B tools/build_biv150_native_gqa_batch.py --emit /ABS/NEW_BUILD
python3 -B tools/build_biv150_native_gqa_batch.py --root-build /ABS/NEW_BUILD
python3 -B tools/bench_biv150_native_gqa_batch.py --run-device --root-owned-exclusive \
  --commit EXACT_HEAD --out /ABS/NEW_OUT --library /ABS/NEW_BUILD/cjj_native_gqa_batch_s4_20261028.so \
  --build-proof /ABS/NEW_BUILD/build-proof.json
```

The last command requires Root's explicit `--root-owned-exclusive`: Root has
verified the previous owned job terminal and schedules serially. `ixsmi` process
visibility is not proof of exclusivity or idle state. Before CUDA it requires
port9031 closed and used memory≤1024MiB. Empty/no-process PID output is recorded
as unavailable; no hostPID is invented. No verified container→driver PID mapping
is available, so any visible unowned PID (including a numerical match to the
container PID) conservatively rejects. Full-table and query PID observations
are combined. During the run, PID/start ticks stay fixed, memory above4096MiB
or an unexplained phase increase>1024MiB rejects. Tiny initialization is outside
numerics and recorded as real GPU work. These are additional refusal gates;
namespace-hidden low-memory workloads cannot be ruled out by these observations.
The actual exclusivity assertion comes from Root's external scheduling.

CPU v2 admission replay (no ixsmi/device access):

```sh
python3 -B -W error tools/check_biv150_native_gqa_batch_admission_cpu.py
```

`admission-cpu-v2.json` covers invisible28554MiB rejection, invisible low memory
recorded unavailable, visible foreign/ambiguous PID, query failure, stale owner,
memory growth and run-cap counterexamples. The entire original numeric-loop AST
is unchanged from dc9d451; its original CPU report remains byte-identical.
The short numeric entry tests
N48/56, max context193, dense/ragged lengths including zero and tails, both Q
strides, read-only inputs, canaries, inactive scratch and full independent FP64
output gates. Baseline TL runs first; both arms must pass the unchanged .005
pointwise/relative gates. Full CPU inputs, outputs and partial bits are saved;
actual TL IR/resources and built native library/source/options are retained.
There is no timing, writer, Graph, side-stream,4K/16K or service qualification.

N40 remains unoptimized. A separate S16 instance is not enabled by this patch:
although partition math and runtime strides look generic, b3's static_assert
explicitly permits only2/4 and must change before16 can even compile. That would
need separate compile/empty-segment/numeric evidence. It is not silently covered.

Actual compile, binding, numerical precision, real N48/56 frequency, performance,
model/service coverage and benefit remain unknown/false. Prior N32/64 component
speedups do not establish an N48/56 benefit or any model gain. Production BF16
allocator provenance/hooking is deliberately deferred until this bounded
candidate has actual numeric and complete-component evidence.
