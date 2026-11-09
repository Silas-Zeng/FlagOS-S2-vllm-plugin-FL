# Native K8 complete-step result

Source `3ebdbebf61e0e90735cd8ad83eb1605627b6ef46`; main4229 production bytes equal b3. Original 377 kernel and existing SO22e unchanged. One ABBA after128 warm replays,8 steps per replay,64 timing replays; append + native producer + original reducer, real width16 scratch and34276 pages.

| case | BF16 ms/step | K8 ms/step | ratio |
|---|---:|---:|---:|
|4K N64/S2|0.4164172113|0.3953698575|1.053234593742|
|16K N32/S4|0.8056746125|0.7645680010|1.053764493720|

36 full snapshots and144 original pointwise/RRMSE array gates passed, including changed Graph before fresh eager, side stream, post-timing bits and runtime complete cache/guards. Root replayed all192 SHA and regenerated the full CPU cold/cache, both lengths and arms phase0 gold, and every phase expected cache chunk digest. GPU cache bodies were discarded by the frozen runner: this verifies expected digests against runtime complete-byte comparison records, not a new read of GPU memory. Numeric thresholds unchanged.

Initial CPU generation, H2D and full GPU pack are separately recorded. New KV pitch256 only. Static warm fixture, one ABBA and16KiB page allocation do not establish service pitch2560, stable/model throughput, capacity or model quality. Native Q8 adds approximation; actual eligible-request model quality remains pending. Official batch8 Math105 may use private fallback, and startup capture evidence is not actual-request frequency.

The shared metadata archive includes original summary, all192-file SHA manifest, log and owner. Root proof and the independent checker are shared. The complete144847098B raw records remain at the remote experiment tag and local path recorded in index.json; tensor blobs are deliberately not embedded in Git. The full replay needs that complete raw root. A128MiB collector cap rejected transport; raising only that cap to192MiB downloaded the same completed run, without a GPU rerun.

Use a detached source worktree at exact3eb, the complete raw root, Python+NumPy and no Torch/Triton:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 python3 -W error tools/check_biv150_native_k8_fullstep_evidence.py --source-root /absolute/frozen-3eb --raw-root /absolute/full-raw --regenerate --out /absolute/new-review.json
```
