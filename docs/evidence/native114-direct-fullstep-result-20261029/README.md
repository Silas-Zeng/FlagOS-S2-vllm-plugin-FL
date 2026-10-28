# K-pair direct fullstep: no clear gain

Frozen code `34074fcd9e0bb168de2b1f53aaa2a3ce96cf3112` compared actual old Native `2ad51a98` with K-pair `ef732ef8`, using the same compact N48 inputs, writer, reducer, 44769 pages, and 2560-element source pitches. Both cases are long-context components, not official 4K/16K service results or a comparison with the latest production main.

| Case | Old mean ms | K-pair mean ms | old/new speed ratio |
| --- | ---: | ---: | ---: |
| valid41, base 17408 | 1.3157761097 | 1.3163745403 | 0.9995453949 |
| valid42, base 16896 | 1.2826907635 | 1.2841372490 | 0.9988735741 |

The complete append + attention producer + reducer used 128 warm replays, 64 timed replays, unroll 8, in old/new/new/old order. GPU event and graph-launch/sync wall timings agree. Root independently recomputed both ratios from the original timings. These small differences show no useful improvement; they are not being claimed as statistically significant regressions or proven noise.

The run completed with no errors and reported its numerical/Graph/side checks passed. The earlier K-pair consumer has a separate independent full-array replay (`ac96a073…`). This negative-performance decision did not repeat the complete current fullstep array replay; the original fullstep records are retained with that limit stated explicitly.

Close this Draft without merging this implementation and without expensive model scoring. Actual paired load instructions and lower VGPR count did not translate into fullstep gain in these cases. No assertion is made about all possible K-vectorized implementations.

Only this README and the selected numerical/timing summary are added as results. The complete 114-file, 94,874,160-byte archive, full arrays, caches, logs, and binaries remain local. SHA-256 identifiers and the exact timing values are in `selected-results.json`.
