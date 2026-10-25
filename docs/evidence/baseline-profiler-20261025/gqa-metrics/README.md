# GQA counter capability audit

These are diagnostic runs of the unchanged shipping 4K GQA producer. They do not measure an optimization. The frozen script is copied verbatim; its original planning docstring says UNEXECUTED, but the results below record its subsequent execution.

The script reproduces the service's packed NHD K/V strides,8192-column block-table stride and narrowed scratch views. The compiled kernel hash is gated to641478e6ddcd3d9c11192fbd0c6d4d472bf8ad3e3bb2fd9fa27b09f18e4dc9ff. Matching strides alone was insufficient: the original full scratch tensor was contiguous and produced Corex use_sme15; the service's narrowed view produces14. This value is derived from tensor metadata, not a requested performance flag.

A/B/C/D producer timings were0.5301024ms/plain-before,0.5305953ms/ixsys-no-counters,0.5312906ms/500Hz-mem_load and0.5298776ms/plain-after. Each used120 Graph replays with32 producer calls, aligned with a named NVTX interval. The producer result stayed numerically equal to its own unchanged baseline; this is not an independent precision oracle.

Counter transport works, but these counters do not establish physical HBM saturation:

- mem_load_throughput reports global+local loads. Requested500Hz produces a7,225,313.9556MB/s plateau; requested10Hz gives144,506.2791MB/s, an exact50x ratio, while stored intervals remain about102ms. These cannot be presented as verified HBM bandwidth.
- llc_mem_req has no recorded unit. Positive counts follow the preceding sample interval; do not integrate using the following dur or assume64-byte transactions.
- issue_stall_reason_synchronization returns zero in the fully contained workload rows. This does not prove zero barrier cost or identify the limiting execution resource.
- Invalid final dur=-1 rows outside the region and partial-boundary rows are excluded. SQLite quick_check, finite values and Graph/API correlation were checked.

The instance reports16 multiprocessors and32GiB memory. Whole-card public capacity/bandwidth must not be substituted for the actual instance's limits. The full kernel trace still establishes GQA producer cost; hardware bandwidth/occupancy attribution remains unresolved.
