# Native K8 paired-word load (CPU/source experiment)

Based on original377/22e through frozen PR93 `3ebdbebf61e0e90735cd8ad83eb1605627b6ef46`, which already contains team main4229. Git fetch was attempted but timed out75002ms; GitHub API independently confirmed4229 as current main and that commit is present locally. This independent branch changes only tools/docs; `vllm_fl/` and `csrc/` remain byte-identical to team main. No V-first, Q-reload or compact-address variant is combined.

## Observed mechanism and history

Original collected `out/biv150_native_k8_qk.s` L529–657 computes lane/global/shared offsets before the full-page loop. L759–803 contains the page-table/page-base work, **8 `ml_lsa_load_dword`**, **8 `ml_slb_store_b32x1`**, a K barrier, then the original four V SME issues/wait/barrier. Original LLVM L562–614 likewise has8 i32 global loads/shared stores followed by those SME calls. These are actual standalone CoreX compiler outputs from the original collector, not a claim that standalone ASM bytes equal the SO's embedded pipeline; SO embedded GPU images were independently matched to the actual objects. The scoped original excerpts, full artifact SHA and collector identity are in `original-mechanism.json`.

There is no runtime K float unpack here: aligned u32 source words are scattered into the I8 TCU shared image. Merely moving the address formula out of the loop is **no-go**, already compiled. Known all-ref history, PR93 source/results, PR96 V-first, PR99 Q reload, PR100 compact-address and the two current research lists were checked; this same Native paired-word path was not found. Old Triton KV8-dequant experiments and INT8 P×V are different dataflows.

The fullstep background is original22e4K .39536985755ms vs BF16 .41641721129ms and16K .76456800103ms vs .80567461252ms, complete append+producer+reducer. Those are background component measurements, not this variant's qualification or a model gain.

## Single change and bound

Original per-lane iterations `word=lane; word+=64` are **not contiguous**: they advance slot by2. The new mapping is exactly `pair=lane; pair<256; pair+=64; word=2*pair; slot=word/32; dim=(word%32)*4`. Each copy reads two adjacent global words in the same slot; dim≤120. Both words are stored separately using original `k_i8_index(slot,dim)/4` and `k_i8_index(slot,dim+4)/4`; their shared positions are not treated as adjacent.

The8B struct and `__builtin_memcpy` preserve original4B alignment rather than adding an8B contract or issuing an unsafe uint64 dereference. Tail slot validity is checked before the copy. The complete shared2KiB byte image and one-writer ownership are preserved; unused shared gap and V image are untouched. Query codec/peak/packing, K→V order, barriers/waits, I8 QK, FP32 online, BF16 P/PV/C chain, output/reducer, ABI/S2/S4 and thresholds are unchanged after the complete symbol/source inverse.

For a full CTA: original64lanes×8×4B = new64×4×8B =2048B K reads. Original wave issue covers2 slots×32lanes×4B=256B; new covers4 slots×16lanes×8B=512B, each slot still one contiguous128B span. Actual coalescing/transactions are unknown. Source target is8→4 global issues, **8 shared stores remain**; K2048+V4096+scale64=6208B/page/head and all math are unchanged. Hence there is no HBM-byte reduction. If f is the old time spent on a perfectly halved K-global-issue portion, an idealized issue-only speedup bound is1/(1-f/2), assuming other costs and occupancy unchanged. f is unmeasured, so no kernel or service percentage is forecast. Register/address liveness can improve or worsen independently; no occupancy claim.

The new namespace/SO is `cjj_native_k8_pairload_probe`. Original377/22e files and all other variants are untouched. Actual scalarization back into8 reads, greater register/spill cost or worse scheduling is a falsifier; do not add volatile, new ISA assumptions or unqualified flags to force a result.

## CPU gate and Root-only next step

```sh
python3 -Werror tools/check_biv150_native_k8_pairload_cpu.py
python3 -Werror tools/compile_biv150_native_k8_pairload.py
```

CPU covers all512words/64lanes, valid0..16, heads0/1, three physical pages and nonzero offsets4/256 (including4-not8-aligned), independent bytewise NHD→TCU image, exact read-address multiset, whole8KiB-stage guards/input readonly and source mutants. It does not simulate device loads, FTZ or allocation.

Only Root may run the explicit zero-launch compile gate:

```sh
python3 tools/compile_biv150_native_k8_pairload.py --compile-only \
  --compiler /usr/local/corex/bin/clang++ --out NEW_EMPTY_DIRECTORY
```

Inspect this variant's own actual LLVM/ASM, alignment/load widths, K scatter image, unchanged MMA/issue/waits, VGPR/SGPR/spills and SO GPU image. No prior qualification transfers. Only worthwhile actual mechanisms/resources proceed to fresh codec/consumer/fullstep bitwise+FP64, pagewise BF16 P reference, all guards/Graph/side and ABBA gates; service/model/performance remain pending.
