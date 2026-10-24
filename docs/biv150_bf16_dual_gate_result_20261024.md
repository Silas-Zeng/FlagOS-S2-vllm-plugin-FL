# BF16 dual-accumulator gate result, 2026-10-24

Tested source: 2fd0da99d372b71aadb84423a544181207513cb5. All 140 numerical cases pass.
These are isolated full operator calls with synthetic inputs, not model acceptance.

| M | Best versus original | Same candidate versus best layout |
|---|---:|---:|
| 1 | 0.9919x | 0.9869x |
| 8 | 0.9829x | 0.9769x |
| 16 | 1.0427x | 1.0427x |
| 32 | 1.0329x | 0.9722x |
| 40 | 0.8476x | 0.7914x |
| 64 | 1.0301x | 0.9598x |
| 2048 | 0.8182x | 0.7925x |

The table selects each M by its ratio to the faster measured baseline layout.
Original-NK and contiguous packed-KN baselines use the same weights and fused
activation. Packing cost is excluded from timing; both copies coexist only for
diagnosis. Packing itself is a separate hypothesis, requiring a real service
hook with no persistent duplicate weights and fresh model acceptance.

Only M16 shows a modest gain over both layouts. M32/M64 can beat the original
layout slightly while losing to the packed baseline. M40 and prefill lose.
No service hook is installed, and no full-shape substitution is justified.
Keep the numerical evidence and investigate M16 specialization or layout
independently rather than claiming all-shape or end-to-end improvement.
