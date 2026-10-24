# Multi-page GQA result: rejected for performance

Tested source 2ae3d9d9043f8e7158dff836a479f9b81ca72948. All 108 numerical cases
pass, but no measured workload has a winning configuration. Timing includes
producer and reducer on actual packed KV strides and shuffled physical pages.

| Batch | KV bound | Ragged | Best pages/warps/stages | Median speedup |
|---|---:|---|---|---:|
| 1 | 100 | False | 2/4/1 | 0.9222x |
| 8 | 127 | True | 2/4/2 | 0.9687x |
| 31 | 16896 | False | 2/2/1 | 0.7517x |
| 32 | 16384 | False | 2/2/2 | 0.7284x |
| 40 | 17023 | True | 2/2/1 | 0.9213x |
| 64 | 4096 | False | 2/2/1 | 0.7267x |

Do not install or merge this operator. Numerical correctness does not imply
performance acceptance. More pages per online-softmax iteration did not help
the tested layouts; retain the raw negative evidence to avoid repeating it.
