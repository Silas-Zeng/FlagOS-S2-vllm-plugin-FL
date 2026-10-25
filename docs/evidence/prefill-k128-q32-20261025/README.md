# Larger KV tile with proved Q32 lifetime — planned component probe

Dependency: PR46. Its BM256/Q32 suffix component is faster but uses202 registers and64KiB shared memory. This probe asks whether KV tile128 can reduce loop/address/online-softmax iterations while keeping peak shared near64KiB, and whether16 warps reduce per-thread register pressure. These are hypotheses, not measured outcomes.

Bounded initial configurations: BM256/K64/w16 and BM256/K128/w16, each against the accepted BM128/K64/w8 hoisted baseline; actual suffix1985/2017 rows first. Record real registers/spills/shared and lowering proof; require numerical and changed-input Graph gates. No production configuration is changed. No GPU run while PR46 service acceptance owns the device.
