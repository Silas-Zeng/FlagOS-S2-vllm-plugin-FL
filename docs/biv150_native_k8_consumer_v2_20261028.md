# Consumer v1 failure and reference-only v2

Actual52c v1 owner706089/start347725174 endedZ. It launched one native producer, zero reducers, then stopped on the first N64/S2/q2048 A partial gate. Maximum/sum, every full guard, readonly input and inactive poison passed. Partial relative RMSE.00129218756 passed but4080 pointwise tests failed; max absolute error.02647874584. v1 remains failed. Final output, remaining cases, Graph and side were not executed. All34 originals/15880353B plus collector manifest are retained in deterministic9-part archives; each gzip is below2MiB. `consumer-v1-failure/archive-index.json` binds every fragment, reconstructed original tar and original per-file collector SHA. Original summary/check/failure/manifest are also readable without unpacking.

The error is in the v1 partial reference. v1 computed unrounded FP64 probabilities relative to the final segment maximum. Original b3 `consume_page` instead updates running max for each16-token page, computes that page'sP andrho, rounds that page'sP via RN32→BF16 before PV, rescales oldacc by the currentrho and adds BF16(P)×BF16(V). The denominator stays unrounded and uses its original FP32FMA. These operations do not commute with rounding once at the final maximum. Exact source inverse proves private377 preserved this original PV/online block; SO22e/kernel/dtype are unchanged.

An independent CPU reference decodes actual full Q/cache/table/length bytes, performs full128-D INT64 QK and dequantized FP64 scores, then follows the mathematical online-page recurrence. It evaluates scores/rho/accumulation in FP64 and explicitly rounds each page's probability to RN32 then BF16, rather than imitating target instructions. Every221184 active partial element was checked:

| Reference | Max absolute error | Relative RMSE | Original .005 pointwise/RRMSE |
|---|---:|---:|---|
| Correct online-page BF16P implementation reference |2.466762693e-6|3.965681533e-8|pass|
| Original ideal FP64 partial diagnostic |.02647874584|.001292187557|fail:4080 pointwise|
| Negative control: noP rounding |.02647874584|.001292187557|fail|
| Negative control: round once after final max |.03135871887|.001017627922|fail|

The unrounded online control equals the original ideal FP64 reference within1.0214e-14, isolating the missing BF16P operation. This supports the reference correction for the observed first case; it does not establish whole-kernel/model/performance qualification.

New v2 source is a counted reversible transform of the frozen52c frontend: change only the fixture-reference import and add separately labelled ideal-partial diagnostic metrics. Its source inverse rejects a guard mutation. The wrapper replaces only `golden.partial`; M/L and **final output's original ideal quantized-score FP64 hard reference are unchanged**. All numerical thresholds stay .005 absolute+.005 relative pointwise plus relative RMSE<=.005. The original ideal partial diagnostic keeps those same thresholds and remains failed. Nothing changes kernel, library, actual datatype, inputs, source pointer ABI, poisoning, Graph ordering or side stream.

The v2 default is stdlib-only. Full CPU replay of all actual arrays, source inverses and both negative controls uses NumPy and imports no Torch/Triton:

```sh
python3 -B -W error tools/probe_biv150_native_k8_consumer_v2.py
python3 -B -W error tools/check_biv150_native_k8_consumer_v2_cpu.py
```

Root-only future device command, same actual codec prerequisite and unchanged SO, new output:

```sh
PYTHONDONTWRITEBYTECODE=1 python tools/probe_biv150_native_k8_consumer_v2.py --gpu \
  --library /workspace/experiments/minicpm-s2/cjj-native-k8-compile-v1-20261028/build/cjj_native_k8_qk_probe.so \
  --codec-summary /workspace/experiments/minicpm-s2/cjj-native-k8-codec-v1-20261028/codec-summary.json \
  --codec-proof-sha256 3c9733ab79e69c77f70068cff2d3da553ad9c2b81b08ea3c396c6e0788b4f263 \
  --out /workspace/experiments/minicpm-s2/NEW_CONSUMER_V2_OUT
```

No native rebuild or author-side Torch/SSH/GPU execution occurred. All v2 device cases/final/Graph/side remain pending. Fullstep performance preparation is parked and must require the actual corrected consumer pass; no performance or model qualification is inherited. precision_gate=false.
