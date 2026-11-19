# Native K lookahead: reviewed model acceptance

The candidate overlaps next-page packed K loads with current PV work. The service uses the reviewed `5898a00a...` library and `cjj_native_k8_klookahead_probe` namespace. Codec, attention arithmetic, cache layout, writer and reducer are unchanged. This change has no PR118 mixed-batch extension or PR120/121 host optimization.

Tested production is `5ca2afaa91bc9133d4170a22410daef4e22f5e00` (production identical to `3ca6386`). Before this documentation-only commit, latest team main `8ba3bc18` was verified against GitHub’s current ref and merged from the matching local Git object; the resulting entire tree was identical to tested `5ca`. Main's A tree is exactly tested `f725598`. Fixed competition service CLI and official benchmark functions were retained.

| Matched score group | A total tok/s | B total tok/s | B/A | A/B mean TTFT ms |
|---|---:|---:|---:|---|
| Latest 4K, legacy warmup, observation OFF | 6301.002955 | 6610.252113 | +4.90794% | 3695.335149 / 3637.551000 |
| Retained 16K, full-only warmup, boundary observation ON | 5149.263504 | 5207.724810 | +1.13533% | 98910.550197 / 97961.480963 |
| Earlier 4K, full-only warmup, boundary observation ON | 6844.358554 | 6696.241801 | -2.16407% | 3513.281057 / 3583.445955 |

Each arm/case has one scored run, with 256/256 successful 4K requests or 128/128 successful 16K requests and no failed requests. Latest 4K adds three short 64-request/128-output warmups, then one complete 256-request/1024-output warmup; 16K has one complete 128-request/1024-output warmup. The arms within each group use identical protocols. The unscored warmup results are never substituted for scores. Different groups are not pooled.

The reviewed official precision is 103/105 (0.980952), with supplemental parallel-64 precision 102/105 (0.971429), from the same production files and C3 as the throughput candidate. There were no request/parsing/empty-answer errors; one official answer reached max tokens and was scored correct by the original grader. Supplemental actual Native N32/N64 request replay evidence is a positive lower bound: there is no final observer summary or complete replay census.

Both latest 4K arms still produce six new index metadata variants, with disjoint keys and changed tuning digests. Final configuration history also differs. These observations include client preflight and cannot isolate scored compile cost. This acceptance does not prove statistically stable gains, zero compilation or a noise distribution. The earlier negative pair remains evidence. No additional third 4K or repeated 16K run was used to select the result.

Source, actual libraries, full original files, token arithmetic, benchmark commands, cache observations, owner identity and cleanup were independently reviewed and replayed by Root. The local full paired report SHA is `21049b066d46cd03d4a9e6fa820da78df1c6e8348b33b60e5f840f4a7024d55b`; selected metrics and private archive hashes are in `results.json`. Raw arrays, model outputs, logs, caches and binaries remain private.

Service deployment must select the reviewed `5898a00a4a4e3314ee8ed1fc40dffde02a766ca6d0e5c116dbd161dccaa3286b` artifact, using the existing Native library argument and its matching namespace/C3. The previous `8b86...` artifact is the control and is not interchangeable with this production identity. No service parameters were changed for scoring.
