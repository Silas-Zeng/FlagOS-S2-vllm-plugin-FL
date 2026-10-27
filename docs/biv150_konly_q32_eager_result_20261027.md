# Q32 eager numerical result: implementation hard checks, BF16 proxy failure

Root's runtime freeze2d79d9546a8a240447fee05479c5f6caa00aa296 completed job
628305/start344402049, same-start Z. This is2048 fresh query rows+4096 preceding
context tokens,6144 total KV, two synthetic fixture patterns. Completed=true,
errors=[], actual binding/mechanism=true,2 encoder and6 reader launches. The
native admission receipt completes with the same queried host PID1820333.

Both patterns passed the recorded device hard guards: independent codec/raw
bytes, finite, baseline FP64 reference, full hoisted/unhoisted output bitwise,
output canaries and read-only inputs. Independent CPU replay confirms both full
4194304-element BF16 outputs bitwise (including signed zeros), all three outputs
finite, full expected/actual encoded bytes identical, and byte placement from
saved original V/codes/scales. It recomputes the selected-row metrics from saved
independent FP64 references without importing Torch or executing a device.

| Pattern1 comparison | Relative RMSE | Max absolute |
| --- | --- | --- |
| BF16 semantic control vs original FP64 |0.002308993744776649|0.00031992080811571455|
| Hoisted I8 vs quantized FP64 |0.0022699815155583425|0.0002909624753580503|
| Hoisted I8 vs original BF16 FP64 |0.009644218032913768|0.0012052177378101825|

All pointwise checks pass, but the original-BF16 relative proxy exceeds its
unchanged0.005 limit. Pattern0 passes; pattern1 and global eager_numeric_passed
remain false. This is a synthetic component proxy failure, not a Math105 result.
Quantized FP64/global softmax does not model online/BF16-P rounding. The CPU
checker recomputes metrics using archived FP64 references; it does not regenerate
the full FP64 attention here. Output canary buffers and live before/after input
fingerprints were not serialized, so those guards remain recorded device checks.

Loaded resources are BF16 control101 registers/98304 shared, unhoisted135/57344,
hoisted122/33792; all have known zero spills,16 warps/stage1. These are resource
observations with no timing. The BF16 control remains canonical unhoisted SDK,
not the accepted service main timing baseline.

All93 original raw files/57073824B plus download manifest are archived without
omission in29 tar.gz files, each<2MiB. Large already-compressed PT files are split
into SHA-bound byte fragments and reassembled exactly; no tensor/reference is
truncated. The restricted CPU PT-storage reader accepts only known CPU storage
tags and tensor descriptors, rejects arbitrary pickle globals/CUDA locations,
and requires no Torch installation. --arrays explicitly uses existing NumPy.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_biv150_konly_q32_eager_evidence.py --arrays --extract FRESH_EXTRACT
```

The previous9f/6439 failures and original artifacts stay intact. The saved
baseline audit's nested `source_IR_and_cubin_exact:true` is a stale historical
helper label and **not** evidence of binary equality. The actual outer cubin
flags/SHA are retained and govern interpretation. The supported statement is
exact source/ABI/key IR plus actual ASM mechanism, followed by real numerical
checks. The archived JSON is not edited to repair that label.

Graph/side stream/16K/performance/service/full validation and competition precision
remain false, timing=null, capacity reclaimed=false. There is no service or105
claim. CPU preparation/replay uses no SSH/JIT/GPU or device queue.
