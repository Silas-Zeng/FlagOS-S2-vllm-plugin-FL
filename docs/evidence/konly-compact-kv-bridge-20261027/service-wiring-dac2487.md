# Frozen DAC1/2487 service wiring — CPU only

This source freeze implements the actual default-off compact installer on main
`dac1b8fb48e7983e48283032efeee9b592c5fbe5` plus immutable PR89 parent
`2487e75aaf367b0924b12a5d53d15c1dff971be9`. It carries the frozen parent production
files and preserves its legacy mode; compact has separate fixed modules and flag
`VLLM_FL_BIV150_KONLY_COMPACT_SERVICE=1`, mutually exclusive with the old encoded
flag and old Q32-hoist flag. Compact stats use
`VLLM_FL_BIV150_KONLY_COMPACT_STATS_PATH`.

The real get-spec/reshape seams, fixed spec/manager registration, source checks,
C3 private receipt and compile fingerprint are wired. Source/config/allocation/
reshape/bind/UniProc object identity are tested using explicitly mocked framework
modules and target method bodies. The pure full-attention configuration does not
activate the block zeroer; BI-V150 PlatformFL.is_cuda() is false, excluding the
normal-CUDA temporary profiling KV lifecycle. Reinitialization remains rejected.

`install-cpu-dac2487.json` is reproducible with:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_konly_compact_install_cpu.py
```

No actual Tensor, JIT, GPU, IR, device numerics, model, capacity or performance
verification has occurred. The child-process check proves fixed module export
only, not RPC/msgspec transport; the actual qualified UniProc get-spec path
returns objects directly.

## New-main hold

Root identified main `b3b87d2a1c13448bd8c6bc24e8045c0458b8b9dd` / PR92 while this
freeze was being prepared. Its new `_launch_producer` attempts stock native BF16
before the old JIT facade. This freeze is **not licensed for that new main**.
Both private formats must intercept the new producer before native eligibility,
using explicit owned-storage identity. Compact must not rely on stride6272 being
incidentally rejected. Wait for the separately reviewed format-aware PR89 freeze;
do not fix this by changing old source pins alone. Existing 16KiB results and old
performance/model evidence do not qualify the new compact path.
