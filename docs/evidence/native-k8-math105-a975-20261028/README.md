# Original a975 model quality after reservation renewal

This is the exact a975 implementation that produced the original6512.18/5012.05 throughput, using the same5c/aace/22e libraries, initial v6 cache and fixed service CLI. It does not qualify later63836c7/main105 code.

| Run | Correct | Exact accuracy | Request errors | Empty final | max_tokens truncations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official105, parallel8 | 103/105 | 0.9809523809523809 | 0 | 0 | 0 |
| Separate diagnostic105, parallel64 | 100/105 | 0.9523809523809523 | 0 | 0 | 2 |

Both runs preserve the original questions, prompts,32768 output cap,temperature1 and top_p0.95. The supplemental changes only concurrency and output directory; it is not the official result. Its capped answers are `test/prealgebra/631.json` (incorrect) and `test/geometry/477.json` (correct), both32768 completion tokens. `failed_predictions=[]` means no request failures; it does not mean no truncation.

The original controller172/start352983003 ended normally, completed=true/errors=[]/cleanup_complete=true. The full53-file12312819B original record set and collection manifest are in `original-records.tar.gz`, SHA256 `5158d7ba600638bf6335021f9c1e51a5bf5929f391fdbbc2fd1fb6a443ef1516`. Independent stdlib replay verifies all file and252 source hashes, the C3 and library identities matching the original throughput, both exact evaluator commands,105 unique matching questions/prompts per run, per-item review/prediction binding, exact correct counts and token totals, and owned cleanup. Root report SHA256 `aac4f17cbc676a1642eeae412b4b9296c478dd087a58ac7192e4591ae5616ad3`; the exact local replay script is preserved.

Native request coverage remains unproven in this run. The observer installed but wrapped upstream `CUDAGraphWrapper`; the actual service uses FL `GraphWrapper`. Original logs therefore contain only observer_install, no valid native capture/replay certificates. These records grant the official a975 Math metric, not a native-bucket coverage claim or full merge acceptance. A separate v2 diagnostic fixes that class binding, retains original records, and is first tested with a short service smoke against the merged latest-main source and newly built stock library.
