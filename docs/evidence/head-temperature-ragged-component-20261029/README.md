# PR115 ragged head component results

Source `9cc1d9e` on main `a455a3d`. All four shapes pass the requested component checks. Keep this Draft: these measurements do not qualify model accuracy or service throughput.

| Rows | A event ms | B event ms | A/B event | A/B host | Event ms saved |
|---|---:|---:|---:|---:|---:|
| 31 | 2.678552 | 2.568009 | 1.043046 | 1.052754 | 0.110543 |
| 41 | 3.358346 | 3.226235 | 1.040949 | 1.049973 | 0.132110 |
| 42 | 3.476809 | 3.183174 | 1.092246 | 1.083146 | 0.293634 |
| 43 | 3.308462 | 3.154723 | 1.048733 | 1.066263 | 0.153739 |

A is the original head plus original Sampler. B fuses the head temperature epilogue and uses the original remaining sampling path. Both include allocations and sampling costs. Each shape has one ABBA, eight calls per leg; these are screening signals, not repeatability estimates.

Normal, changed, zero, extreme, row-varying temperature, per-request generators, side stream and after-timing eager cases report full scaled/filtered/ID bit equality, equal RNG, finite values, unchanged inputs and full-storage guards. Changed Graph replay checks the deterministic prefix. The independent FP64 check covers five columns per row; it is not a full-vocabulary reference.

## Service boundary

The protected fixture uses a 4096-byte W suffix. Actual R41/R42/R43 LLVM shows an unused final W prefetch reading up to64B beyond logical storage; this fixture pass must not authorize ordinary service weights. R31 does not have that final speculative W load in its compiled loop. Production admission and any required kernel correction remain separate work.

The measured saving is only 0.11–0.29ms per head+Sampler invocation. It must be weighted by actual matching service steps; no model percentage is inferred. Full raw data/cache/compiler outputs are retained locally. This commit publishes only this README and selected JSON.
