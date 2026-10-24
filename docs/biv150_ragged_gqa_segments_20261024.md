# Actual mixed-load ragged GQA segment-width probe

Validated team main5eda0b0 is the base; production GQA source remains exactly
SHA1a81409ba23a149081f7fc257460ea1512cbfdceeb0af32a2698994958749b9a.
No serving hook, scheduling choice, source guard or kernel body is changed.

The live PR28 coverage trace from the personal instance shows eager GQA grids
B15×2×16 and B31×2×8 during16K mixed-prefill execution. Its actual FULL graphs
also show B40 coverage. The old96-case wave extension compared the current width
against8 and16, and therefore does not settle whether narrower1/2/4 segments
help B15/B31/B63. These observations guide this experiment; no improvement is
assumed and no claim that every historical benchmark omitted them is made.

The240 configurations cover B15/16/31/32/63/64, KV4096/5119/16384/17407,
uniform/ragged lengths and segments1/2/4/8/16. Use actual packed KV strides,
random pages, NaN partial scratch, sampled independent CPU FP64 reference and
full-output comparison at unchanged0.005/0.005 tolerances. Changed-input Graph
replays modify query values and lengths, including a zero padded row. Three
alternating warmed rounds time producer plus reducer together. The numeric
case count must be complete; no model or throughput score is claimed.

The comparison is per actual input geometry. Any later dispatch must stay within
builder-allocated scratch, retain current policy for unverified inputs, preserve
actual call-chain/source identities and pass full105-item precision plus both
fixed warmed benchmark workloads and TTFT. The draft is created before GPU use.
It waits for owned PR30 epilogue probe and refuses a busy GPU/9031. Source and
results are private; no shared package changes or teammate machine access.
