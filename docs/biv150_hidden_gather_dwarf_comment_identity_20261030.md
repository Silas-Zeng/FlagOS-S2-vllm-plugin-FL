# Default-worker gather: one derived DWARF ASM comment

This tools-only child of `663ff902f4638ab4e21b04b303319af4c2f009a0` fixes the
known after-ACK identity rejection before the default-enabled worker smoke.
Production, controller, snapshot, observer, kernel, ABI, resources and C-function
address ownership rules are unchanged. The original 67-request plain-smoke and
formal Math/benchmark paths are unchanged. No GPU or new worker has run here.

`gather-4a9f-source-directory-dwarf-comment-v2` adds exactly one normalization to
the original `5b109cb` rule. The unique line must match
`\t.byte\t1 +; Abbrev [1] 0xb:0x... DW_TAG_compile_unit`. Its hexadecimal comment
must equal the current original ELF's `unit_length - 7`; only that numeric comment
is replaced with reference `0xa4`. The instruction, whitespace and remaining ASM
are preserved. Existing ASM unit-length and source-directory checks still apply.
Full ELF canonicalization is byte-unchanged and must equal the old complete ELF
reference; all other source/TTIR/TTGIR/LLVM/ASM hashes remain exact. This is not a
comment-stripping rule or an executable-section-only ELF check.

The proof records the new rule, actual rule-source SHA, original comment,
ELF unit length and canonical hashes. The source bridge also pins this tool's
bytes. Old `5b` receipts and raw artifacts are not changed or retroactively marked
strict-pass. No new worker numeric, Graph, model or service qualification follows.

The bounded CPU replay uses manifest-pinned actual B16 cache artifacts from
`/private/tmp/cjj-native122-b16k-raw-20261030`. The original rule rejects its single
`0xad` comment; the new rule proves `180 - 7 = 173 = 0xad` and canonical equality
to the complete existing reference. The manifest SHA is
`5924c0a60d5bae231224decfe633def6d12ec38afcdd0d947eff0d31537a7fd8`.
The original independent proof is in the Root-reviewed B16 reviewer, lines140–146.
Only cache/metadata replay is added; this does not repeat the full service review.

Run from this worktree with Python3.12, choosing a fresh output:

```sh
python3 -B -S -W error tools/check_biv150_hidden_gather_service_cpu.py \
  --component-out /private/tmp/cjj-dynamic-hidden-gather-service-object-numeric-v2-raw-20261030/out \
  --actual-b16-cache /private/tmp/cjj-native122-b16k-raw-20261030 \
  --out /private/tmp/native122-dwarf-comment-cpu.json
```

The existing CPU checks retain complete ELF, source, ABI and C-address negatives.
Added cases reject incorrect derived lengths, duplicate/missing compile-unit tags
and other ASM bytes. A finite four-change inverse restores the entire original
identity module. Synthetic path fixtures now emit both actual derived DWARF
numbers; these fixtures are explicitly not device compilation.
