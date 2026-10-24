# Equal-capacity partial transpose: correct, no stable gain

Frozen ca70e27, unchanged shipping producer/reducer and split policy, true packed KV/random page tables,13 batches ×4 lengths ×dense/ragged=104 cases.104/104 original bit identity, sampled CPU FP64, poisoned scratch and changed-input/zero-length Graph checks pass. Both producer and reducer are timed with three alternating rounds; no extra transpose copy or extra scratch capacity.

Results remain near unity. Maximum median1.014761 is not consistent across rounds (best minimum across all cases only1.005111); no batch has a strong consistently positive scope across all tested contexts/ragged cases. Main B32/64 ranges are essentially1.0. The arbitrary3% exploration threshold is not the only rejection reason: full raw rounds show no reliable benefit. No service hook or model claim. Close this specific transpose proposal; retain all data and leave other layout/prefetch structures open.
