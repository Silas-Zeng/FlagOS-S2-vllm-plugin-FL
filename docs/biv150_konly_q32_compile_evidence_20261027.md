# Preserved Q32 compile-only target evidence

This evidence-only update preserves the actual d6 Q32 compile job without
changing any reader, source factory, ABI, driver, guard or production code.
Job626170/start343488382 reached same-start Z. Its original40 files total
3,047,276B; all original file hashes were rechecked before packing. The original
download manifest is the41st archive member and is retained byte for byte.

Archive: `evidence/konly-prefill-reader-20261027/q32-i8-hoist-v1/original-compile.tar.gz`,
411,921B, SHA256
`f3e4f8e1f4414fc1181818746d568c2e7213211f3924db223a8a1e9d549fe183`.
The manifest inside has SHA256
`ff5011dddc1edaef9d94f52fe5aaf67e01da03147e3ab633bf11e611a234501f`.

The original `compile-summary.json` remains completed=true/errors=[]/0 launches,
source driver d6a42ba24bd6f48cc73e3116394f0848b2610745 and reader
24c741cc152fc7a53f0b8ce27c1e988898fc59ec. I8 hoist remains unsupported/false;
runtime binding, accuracy, precision, performance and service remain unqualified.

| Role | Actual hash | Recorded shared | Registers/spills |
| --- | --- | --- | --- |
| Canonical BF16 control |678ec5b47d05e5f427ebedf674334c3ec7a1e948cd0ac59e514002ec17b42485|98,304B|unknown/null|
| K-only I8 QK |c0a523d9d0d4d5dde57b241df9d22c9c38fa71e480878d18e690b4eb002e24f7|57,344B|unknown/null|

The BF16 control is unhoisted canonical source, **not** the accepted service Q32
baseline. Resource differences do not establish speed. No numerical or timing
work ran, no serving hook was installed and no KV capacity was reclaimed.

The archive retains source, TTIR/TTGIR/LLVM/ASM/cubin, complete per-role metadata,
typed ABI, actual Tensor descriptors, logs, owner record and the private compiler
cache including opaque binaries. Packing/replay does not execute those binaries.
`archive.json` records provenance. The subsequent independent pass commit adds a
CPU original-byte checker and an I8-specific transform; that commit must not
rewrite this original false proof or interpret these compile records as a pass
for physical Q hoisting, numerical accuracy, performance or service use.
