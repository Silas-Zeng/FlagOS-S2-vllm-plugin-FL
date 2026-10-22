# BI-V150 operator experiments, 2026-10-23

Integration baseline: team `main` commit `9a43eb1e883a1ba986a4ae8052f1d00d8fd7aa5f`.
This experiment does not merge a newer official framework or modify scheduling.
Device: one BI-V150 32 GiB; COREX 4.5.0, PyTorch 2.10.0+corex.4.5.0.

## Address arithmetic and segment reduction: rejected

`tools/bench_biv150_fastpaths.py` compares the shipping GQA kernel with constant
strides, 32-bit indices and smaller reduction launches. The same seeded tensors
and shuffled physical pages are used in every arm. A warmed CUDA graph is replayed
8 times per sample, with 15 samples reported as a median. Mixed sequence-length
tests cover zero-length graph padding, page boundaries and partial pages; short
sequences also use an independent dense FP32 attention reference.

The attached CSV is operator latency, not end-to-end serving throughput. Address
variants are checked bit-for-bit. Warp-count variants use 0.005 absolute/relative
tolerance because their FP32 reduction tree changes; actual errors are recorded.

The best primary-shape change is below 1%, while some short shapes regress.
32-bit indices alone regress on primary shapes. None of these variants is enabled
in production: the measured result does not justify another specialized path.

Remote raw logs: `/workspace/experiments/minicpm-s2/operator-fastpaths-20261023/`.
