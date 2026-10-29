# PR123 actual production cache-pair numeric entry

This tools-only child of `9b4e032dc6390f4f19c07a7c996ed898ba908d9e` preserves all production files. It checks the two actual production-compiled handles from owner 205019 / start 364277503. New service preparation, model accuracy, service throughput and arbitrary shapes remain unverified.

The actual compile review is `f7d2ff77125dbe832d9cff98965858d638707c79fd706c00fceaff8c9152692a`; summary `12df8d1cbf08f58ae53562a9ba0abec9cde5389917ee129a642365f0cdfb625b`; complete 52-file manifest `621205ab3678cf4a597ce8c2bb363e52bf9c23276f5b55b0e6e059bbb24aafb2`. Final cubins are unhoisted `f0599daf8ec6c1daae817deae3e66839037171f1ded8944548c69100fd622267` and hoisted `d8fba5d8f19305a0928ed9cf10e98256dfdeffbf5cc82dc71c12224f6e57b860`.

The source factory reads SHA-pinned original ce905 numeric source/contract and the unchanged 522c loader/reference dependency. Its generated driver is an exact reversible adaptation of the already executed ce905 driver, not a new mathematical implementation. Original policy, FP64 reference, phases, full outputs, poison/guards, whole cache/read-only bytes, Graph B before fresh eager B, and side C remain unchanged. Total Q2048 / N2 / BQ32 / K64 / w16 / s1 / pitch2048 / 773 16KiB pages remain fixed. The known BF16-proxy failure remains an explicit diagnostic boundary; no threshold changes and no timing are included.

Both arms use the same normally imported production JIT function at the exact original checkout:

`/workspace/experiments/minicpm-s2/code/cjj-native123-service-compile-v1-20261030`

That checkout must remain clean at 9b; module path and all frozen production source digests are checked. The frontend's own commit is recorded separately. A different checkout path is not assumed to have the same Triton key. The original hoisted override path/content remains bound to the actual compile. A new `Adapter.prepare` is deliberately not run: its new PID-specific override path would request another compilation.

The two reviewed disk groups are copied byte-for-byte to a fresh private cache; only group index paths relocate. Warmup one requires empty JIT memory cache. Warmup two requires the precise one-entry key/object mapping captured after the first handle passes the full cubin/ABI/resource check. Each warmup must retain all prior keys/objects and add exactly one object equal to its returned reviewed handle. Clearing, cloning, unrelated cache entries, replacing the first object, or adding multiple objects fails. Original cache validation, real CompiledKernel binding and backend-compilation prohibition remain. No fallback or retry can silently compile a reader.

CPU validation:

```sh
python3 -B -S -W error tools/check_biv150_konly_multi2_service_numeric_cpu.py \
  --raw /private/tmp/cjj-native123-service-compile-v1-raw-20261030 \
  --review /private/tmp/cjj-agent-native123-service-compile-independent-review-20261030.json
python3 -B -S -W error tools/probe_biv150_konly_multi2_service_numeric.py --cpu-preflight \
  --raw /private/tmp/cjj-native123-service-compile-v1-raw-20261030 \
  --review /private/tmp/cjj-agent-native123-service-compile-independent-review-20261030.json \
  --references /private/tmp/cjj-agent-native123-service-numeric-references-20261030
```

97 CPU checks passed, including original policy/loader refusal gates and fake opaque-object tests of the exact added memory-map statements. This does not execute real warmup or establish GPU qualification. CPU report SHA `443795975ac4f0411ab505bfa63a983a39fb113315b147449ec682a84367edce`.

Root-only device command template after source review and exclusive scheduling:

```sh
python3 -B -W error tools/probe_biv150_konly_multi2_service_numeric.py \
  --gpu --root-owned-exclusive --numeric-diagnostic \
  --commit <this-frozen-frontend-commit> --out <new-private-absolute-output> \
  --raw <complete-52-file-compile-archive> \
  --review <exact-f7d2-independent-review.json> --references <original-pinned-reference-directory>
```

Git staging must explicitly retain ce905, 522c0b0617a8d2d862f60ebb03f05f6e08ccfcfe and 844a69f10539c551500e29e2304061aef7d8671b objects and their pinned helper blobs. No original arrays/cache/log/binaries are included in this code change. A successful run qualifies only these two actual compiled handles at the original scope; later service preparation and request-window hits require separate validation.
