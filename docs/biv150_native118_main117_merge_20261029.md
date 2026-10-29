# PR118 normal merge of accepted PR117 main

This branch starts at frozen PR118 `01dfe84028560b1a982dac33bc788e875ce9cf1b`
and normally merges team main `56f88737fcf3b940760b9ca3027a925cd5a7d77d`.
Merge commit `d09840e3267f0205aab177b2fd7d036838225fe4` has exactly those
two parents. The running `01dfe` checkout and its consumer were not modified.

The only conflict was the adjacent `native_decode` fingerprint assignment and
PR118's pending mixed qualification block. Resolution keeps the accepted
`5898a00a4a4e3314ee8ed1fc40dffde02a766ca6d0e5c116dbd161dccaa3286b` library,
`cjj_native_k8_klookahead_probe::produce`, and the original entire mixed block.

| Compared with | Production difference / retained behavior |
| --- | --- |
| Common ancestor `8ba3bc18` | Union of the independent PR117 identity update and PR118 host integration. |
| Frozen PR118 `01dfe840` | Only three accepted PR117 identity files differ: `compile_semantics.py`, `biv150_konly_encoded_native.py`, and `biv150_konly_encoded_sources.json`. The first has only the library SHA/operator substitutions; the other two are byte-identical to main56. |
| Accepted main `56f88737` | Exactly the original six PR118 production files differ. Reversing its host-edit list recovers main56; mixed adapter/runtime/service/source-pin behavior is unchanged from01df. |

The runner is byte-identical to both parents, including CRLF. Its existing
compute-order `EXPECTED_RUNNER` remains correct. The compute-order mixed-wrapper
source SHA still matches the unchanged PR118 wrapper. Main56's Native module/C3
manifest remain exact; no117 gain is reverted. PR118 is still default OFF;
all four production library/review qualification hashes are `None` and
`service_verified=False`. No118 actual-library result is inserted here. Source
fingerprints change with this merged integration, including when mixed is OFF;
old serialized compile fingerprints are not reused.

## Source proof and admission boundary

`biv150_native_k8_mixed_factory.BASE` remains a455 for the original GPU source.
Only `HOST_BASE=56f88737...` is added for the host inverse. Two JSON anchor strings
next to the fingerprint's native assignment follow the accepted117 namespace.
The new proof records `host_reverse_base` explicitly instead of mislabeling the
host inverse as a455. This is provenance, not an equation or GPU transformation.

The actual saved01df v2 compile summary was read locally. Its four source inputs
(host/CUDA/codec/header), original raw input bytes, host/device flags, symbol map
and compile frontend SHA are all identical to this branch. The factory SHA
changes; `source_transform` differs only by the added host inverse base. All
existing PR118 consumer, numerical contract, decode-step, FP64 reference,
writer/reducer, timing and mixed GPU kernel files remain unchanged from01df.

The following old receipts deliberately cannot be reused automatically:

* Invoking compile-only in this new checkout with `--expected-source-commit
  01dfe840...` fails its existing exact-HEAD check before Torch import.
* `biv150_native_k8_mixed_numeric_contract.admission` compares the complete
  recorded `source_transform` with the current factory proof. Executing its
  unchanged AST guard on the real saved01df compile metadata rejects with
  `PR118 component: actual exact source/options`. Both the consumer and
  decode-step entries call this admission first, so their subsequent numeric
  receipts do not override the refusal.
* The frozen numerical entry's same-S reference is still the explicit8b SO.
  That mathematical comparison is not a new mixed-plus5898 service coexistence
  test. No source change silently relabels it as such.

Root may continue frozen01df component tests and review their evidence in that
scope. A subsequent service identity/qualification update must explicitly bind
the merged host source and accepted5898 parent; it is not granted by this merge.

## Affected CPU checks

```bash
python3 -B -S -W error tools/check_biv150_native118_main117_merge_cpu.py \
  --library /private/tmp/cjj-native117-klookahead-compile-v1-raw-20261029/out/compile/build/cjj_native_k8_klookahead_probe.so \
  --helper /private/tmp/cjj-actual-owned-service-helper-94db-20261028.py \
  --old-compile-summary /private/tmp/cjj-native118-mixed-compile-v2-raw-20261029/out/compile-summary.json \
  --out /private/tmp/native118-main117-merge-cpu.json
```

This reuses the original mixed CPU routing matrix and actual94db fingerprint
checker with reversible in-memory parent identity substitutions. The legacy
test files on disk are not weakened or rewritten. Twelve mixed cases, twelve
rejection/no-retry mutations, both inherited32/64 routes and five unknown-shape
TL fallbacks pass. The accepted5898 ELF is read and hashed, never loaded. The
test also proves the real old compile receipt is refused under the new host
proof. No Torch/Triton import, new device compilation, GPU array replay or
service performance measurement is performed.

[Selected complete CPU report](evidence/native118-main117-merge-20261029/cpu.json)
is reproducible and contains no GPU raw arrays, logs, or binaries. An initial
local checker-only error expected the admission message without its existing
`PR118 component:` prefix; its first script/diagnostic is retained in private
temporary files. The final check asserts the exact original prefixed message;
the production guard was never changed.
