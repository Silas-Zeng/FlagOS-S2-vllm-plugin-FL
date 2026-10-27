# PR83 writer component device evidence

The root scheduler ran exact commit `58bace0b73614d2785d55b70a8376b91cb5e893d`
using `--run-gpu`, after its prior workload completed. The collector records
owner PID636247/start345046048, same start, terminal state Z. This CPU archive
work did not connect to that machine, import a framework, compile or launch.

`raw-v1.tar.gz` contains **all160 original files (4,751,338 bytes) plus the
unchanged original download-manifest**, including owner/log, result, raw cache
outputs, source/slot fixture bytes, compiler/cache records, TTIR/TTGIR/LLVM/ASM/
cubin and the actual saved kernel source. No original bytes were edited or
excluded. The deterministic gzip is393,763 bytes, SHA256
`c59dea2c3a08dbf09031f5d92123bcc0aba3522ed0aa98fc54a13971b28fabdb`.
The original collector manifest SHA256 is
`2280fec62481ef5cd69f563957de4829cf66782a8e00be6857a80da1fbb00347`.
`archive-manifest.json` provides every member's SHA/size and the summary SHA.

The raw result reports completed/passed/requested_writer_gates_passed=true,
errors=[] and component_only=true. There are9 cases: empty has no launch;
8 launch direct checks, including141 old codec vectors/282 valid head rows.
The unequal384/640-pitch case and shared2560 QKV case also change K/V/slots,
restore the cache poison, then execute Graph replay and side stream. This gives
13 full cache checks totaling737,280 bytes, including2 changed Graph and2 side
checks. Other source pitches include256/640/2560; source K/V extents differ.
The negative-only case preserves the cache with NaN-poisoned invalid K rows;
V includes signed zero, subnormal, extreme, Inf and NaN payload bits.

The independent stdlib replay checks all160 original hashes, unchanged source
and CPU/provenance/admission output, actual descriptor extents/pitches/offsets/
aliases, all typed cache pointers at the same layer-page0 base, and explicit
CPU-owned pre-upload slot snapshots. It reconstructs the entire encoded cache
from saved pitched source bits with the unchanged old RN32/RNE CPU codec,
then compares every byte to the saved actual and expected outputs. Reserved,
unwritten slots and the BF16 V field are included, with no tolerance. All13
comparisons pass. All21 original CPU rejection cases and old codec provenance
replay unchanged. Nine new evidence faults each hit their intended rejection
(byte/code/reserved corruption, changed slot, ABI, spill, source stride,
missing Graph check, unearned precision, actual TTIR pointer type).

For8 launched records, actual postlaunch resources are9 registers,0 spills,
0 shared, w1/s1/warp64; use_sme is reported as0/1/3 across alignment variants.
Five distinct compiled hashes exist. All48 per-case saved compiler artifacts
(6 kinds each) are nonempty and match their recorded hashes. The actual TTIR
and TTGIR function parameter lists independently match6 pointers
`BF16,BF16,I64,I8,FP32,BF16` followed by6 I64 scalars. Warm and loaded binary
hashes match. These are actual postlaunch resource and typed-IR records, not
prelaunch-null or CPU signature-schema claims. This is not a formal proof of
every generated instruction or arbitrary dynamic source values.

The saved **actual full cache output** is independently replayable. The original
bench also read back and asserted source/slot/cache outer canaries; however,
its saved source/slot files contain the CPU expected fixture bytes, and it did
not save a separate complete post-read outer-canary/source image. The archive
preserves those original successful assertions/records; it cannot independently
reobserve that GPU readback. It does not relabel those files as actual snapshots.

`precision_gate=false`, performance_measured=false, model/service/full lifecycle
and arbitrary live-slot provenance remain false. This is no service installation,
model105 result, latency/throughput result or all-reader integration proof. It
does not activate cache registration, fused RoPE, preemption/copy/prefix reuse,
Graph epoch or live scheduler admission. PR83 stays Draft, depending on79/80/82;
the typed-view broker86 is a separate CPU integration prerequisite. No entire
K-only model or service acceptance follows from a writer byte contract.

Replay from the committed archive (stdlib only, no extraction/framework):

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -W error \
  tools/archive_biv150_konly_strided_writer_cpu.py --out /tmp/NEW-summary.json
```

The result must byte-match `summary.json`. To independently reconstruct all
three evidence files in a fresh directory from the original root download:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -W error \
  tools/archive_biv150_konly_strided_writer_cpu.py \
  --raw-dir /private/tmp/cjj-konly-strided-writer-raw-v1-20261027 \
  --build-evidence /tmp/NEW-evidence-directory
```

Compare its tar, summary and archive manifest byte-for-byte. The current team
main API was freshly read as DAC1b8f before this evidence-only work. Two normal
HTTP/1.1 fetch attempts timed out at the15-second low-speed limit; neither is
claimed as successful. The tested58b source and existing branch production
base56f are preserved; this update adds only archive/checker/report files.
