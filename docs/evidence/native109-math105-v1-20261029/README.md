# PR109 completed Math105: original records and independent CPU review

Tested source **fc25c86a13fc273bb7368b83af4ca264524b4e12**, controller **d8a5bd435845bdb3dd1e4d3741f4ca38fe85b4cc152036517cbcf034f0eb4495**, original helper94db. This documentation-only commit starts from shared **45321a9410885ed0fc6aa9be6aeeac13a7f7f89b** and leaves the tested production files unchanged.

| Run | Correct / complete unique questions | Exact accuracy | Max-token truncations | Empty output / request errors |
|---|---:|---:|---:|---:|
| Official parallel8 | 103/105 | 0.9809523809523809 | 1 | 0/0 |
| Supplemental parallel64 | 102/105 | 0.9714285714285714 | 0 | 0/0 |

Both reported rounded scores (.981 and .9714) match the sum of all saved per-question evaluator grades. Predictions, reviews, question/index identities, model output, token arithmetic and reports were checked together. The two runs contain210 distinct API response IDs, the same105 unique questions and complete input prompts. The official protocol remains parallel8; supplemental64 changes only batch size and output directory and is diagnostic. No independent mathematical regrading of the105 answers is claimed. Incorrect-question and truncation records are retained in `independent-review.json`; zero failed predictions does not mean zero truncation.

All41 original files / **11,198,363 bytes** match their original download manifest. All268 source hashes match the immutable fc25 Git tree, following the recorded MoE symlink. The exact executed controller/helper, v6 tuning identities, actual worker C3 fingerprints, native source/schema/import/load receipts and cleanup were checked. Distinct libraries are W8A8 gate5c, stock NativeGQA0c66 with V overlap, and compact Native40f9. This archive contains saved load receipts, not their SO bodies or a fresh remote SO hash.

Original owner44037/start355048326 was collected with the same start and state Z. State completed with cleanup true and no errors; the identified API exited0 and no owned processes remained. No EngineDeadError is present. No independent post-cleanup GPU-idle observation is added by this audit.

Graph bootstrap was **disabled**. Host/capture records do not establish actual request frequency or Native graph-bucket quality coverage. Final stats were not saved, so final lifetime counts/overflow/errors remain unknown. `native_quality_covered=false` and `native_model_precision_verified=false` remain unchanged. These two Math results do not by themselves qualify Native request coverage, throughput, full acceptance, later sources, or a merge.

## Exact preserved audit

- `independent-review.json`: original bytes, SHA256 **daa1948b4df1eecbbfda6f73e6d90ede3c03023a631b4a41ea260f0691dfe9cc**.
- `review_math105.py`: original bytes, SHA256 **65bb1899511baa983319a5fd112736c70a937efb7a599bce6d7cc435ae2be7ef**.
- Three exact dependencies under `dependencies/`: original smoke auditor, original reusable Math105 auditor and actual helper94db. The review pins their SHA values. It executes only the saved evaluator-consistency block of the old Math auditor, observing this run's correctness/truncation counts; all protocol, grade/token/identity and threshold checks remain intact. Historical expected answers/counts do not become targets for this candidate.
- `download-manifest.json` is an unchanged copy of the original41-file manifest.

## Complete archive, not a subset

The deterministic gzip is **1,661,665 bytes**, below2MiB. SHA256 **5f0622616f2587fdd2ff4ef1605493e1cdcc89188df36882cf9449268e9b1b1d**. It contains all41 collected originals plus their original download manifest (42 members), including both complete prediction/review JSONL files, reports/configs/logs/commands, source/C3 manifest and cleanup. No collected raw file is omitted. Generated cache trees were excluded by the original collection, which is not represented as a full out-directory snapshot.

The complete durable workspace copy is:

`/Users/chenjunjie/Documents/flagos/repos/experiment-raw-archives/2026-10-29/cjj-native109-math105-v1-raw-20261029.tar.gz`

Its adjacent `.tar.gz.index.json` and the shared `archive-members.json` list every member/size/SHA. `full-archive-location.json` records the durable location and complete archive hash. The shared gzip is the identical full archive, not merely a pointer to an unshared copy.

## Lightweight verification and optional CPU replay

From a Git checkout containing fc25, with Python3.12+ and only the standard library:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error docs/evidence/native109-math105-v1-20261029/replay.py
```

Default verifies shared file SHA values, the full gzip and all42 original member bytes; it does not rerun the105 review. Packaging checked those bytes and ran this default only. It did not rerun the already completed full CPU review, component numerics, GPU, compilation, service or model.

An explicit later replay is available:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error docs/evidence/native109-math105-v1-20261029/replay.py --run --out /private/tmp/cjj-native109-math105-root-NEW.json
```

That opt-in extracts the full archive to a temporary directory and binds only the auditor's three dependency paths to the exact shared copies, leaving their source bytes/functions/pins/thresholds unchanged. It reads the immutable fc25 Git objects. Temporary raw paths differ in the regenerated report; the original daa194 report remains untouched. `shared-files.json` pins the saved artifacts/dependencies, excluding this replay driver and the hash manifest itself.

All packaging used local existing originals. No SSH, remote compression, network push, PR modification or production change was performed. Root's same-source full-warm/score throughput job is separate and its results are not claimed here.
