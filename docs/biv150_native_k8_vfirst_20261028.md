# Native K8 V-first source experiment (CPU preparation)

Depends on PR93 frozen `3ebdbebf61e0e90735cd8ad83eb1605627b6ef46`, containing authenticated main `4229b48b64c7bd21f9f8fc3c5ff8775500f8d02b`. CJJ verified main by fresh fetch and GitHub API on2026-10-28. No official framework update. This independent branch changes no production or existing PR93 file/library.

The sole dataflow edit issues the original V SME loads before the original K uint32 global-to-shared packing. The K barrier, V wait8/barrier, tail clearing, codec, QK, FP32 online softmax, page16 BF16 P, BF16 PV, C accumulator, stores, S2/S4 and ABI remain unchanged. K already uses aligned4-byte words; this does not reduce load count. Shared K covers exactly bytes[0,2048); V covers[4096,8192). This nonoverlap is a source safety argument, not proof of the actual backend schedule or useful overlap.

CUDA entry symbols, kernel names, codec export names and Torch namespace are isolated as `cjj_native_k8_vfirst_probe`; codec changes are names only. Exact inverses restore all three parent K8 files byte for byte, then the original parent inverse restores b3 native BF16 source. `COREX_NATIVE_GQA_V_OVERLAP=0` and its static rejection of the wait-moving variant remain.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_native_k8_vfirst_cpu.py
```

The next private frontend will preserve the original build/consumer/fullstep bodies by a narrow inverse-proved host adapter: independent output SO and actual LLVM/ASM first; Root CPU review of real V issue/K packing/barrier/wait ordering; then fresh consumer and complete BF16-vs-variant fullstep FP64/Graph/guards/true-width16/ABBA. Parent22e and BF16aace library identities remain historical prerequisites/controls, not qualifications of this new binary. No compiler, Torch, Triton or GPU was invoked here. Actual binding, scheduling, numerical, performance, model, service and capacity qualification are false; no performance claim follows from the source order.
