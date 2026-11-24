# Private mixed decode: actual 4K/16K implementation evidence

Both runs used frozen **4a3fc7773b37e745082d0d98f8d53713130384ea**, based on team main **dac1b8fb48e7983e48283032efeee9b592c5fbe5**. This evidence commit adds tools/docs only. It does not modify the tested factory, codec, driver, original numerical gates, production package or original result/log files. Root alone scheduled the actual device work; this author performed CPU/file/IR review only.

| Case | Original owner (terminal Z) | Original raw | Split S | Decoder rows / total queries |
| --- | --- | --- | --- | --- |
| mixed4k | 636444/start345080447 | 86 files / 24,505,683B | 2 | 63 / 2048 |
| mixed16k | 637563/start345185303 | 86 files / 24,254,297B | 4 | 31 / 2048 |

The original results have `completed=true`, `errors=[]`, actual compile/typed ABI/native MMA/PV C-chain checks and implementation hard gates true, including stock input bits, encoded-cache byte oracle, readonly/canary/poison and changed-metadata Graph/side checks. **Both original results retain `passed=false`, `gpu_numeric_verified=false`, `original_bf16_proxy_passed=false`**. No timing or service integration occurred. Physical pages remain16KiB; capacity is not reclaimed.

The isolated fixtures retain Q pitch2560, output pitch2048 and query capacity2051. Four controls cover natural leading decode, prefill first plus ragged0/1/15/16/17/63/100/last lengths, interleaved prefill with zeroQ, and no decode. Their prefill table rows are deliberately poisoned; they are not valid inputs for a paired prefill launch. A service integration needs a separate valid prefill table and a disjoint paired reader that skips already handled decode rows.

## Independent CPU output/FP64 replay

`check_biv150_konly_mixed_evidence.py --arrays` verifies archive/chunk/original SHAs, reconstructs the original NPZ byte-for-byte, and recomputes all15 saved FP64 row metrics per case with CPU binary64. Each metric matches its original JSON field exactly. It checks every saved output row: decode outputs finite, empty rows zero, all prefill/padded rows still BF16(42.5). ZeroQ and no-decode full output arrays are bit-identical between the BF16 and K8 arms in both cases. Ordinary nonzero-Q arms differ, as expected from different Q/K operands.

| Maximum sampled relative RMSE | 4K | 16K |
| --- | --- | --- |
| BF16 output vs original FP64 | 0.0022539879 | 0.0023286302 |
| K8 output vs quantized-operand FP64 | 0.0023003915 | 0.0023526529 |
| K8 output vs original FP64 | 0.0104414649 | 0.0103169124 |
| Quantized FP64 vs original FP64 | 0.0100507484 | 0.0100333524 |

Both baseline and quantized-operand pointwise gates have zero violations in all saved reference rows. The unchanged original-BF16 relative gate fails10/15 rows per case; original pointwise violations total371/246. The similar quantized-reference/original-reference gap supports operand quantization as the main sampled error source; it does not prove all possible inputs. The original `.005 + .005*abs(reference)` and relative `max(.005,1.2*baseline)` rules are unchanged.

**Witness limits:** cold K/V and complete encoded-cache tensors were not collected. Saved FP64 references are checked, not independently regenerated. Full output arrays are present, but scratch/max/sum/eager/Graph/side snapshots are absent from the NPZ. The original frozen runner takes integer bit views of all four buffers and asserts exact Graph/eager and side/eager equality; the original result records success. This CPU report verifies that source/assertion coverage and the result flags, and does **not** claim independent partial/Graph-array replay. No rerun was used to rewrite historical evidence.

## Actual LLVM metadata CFG and address dependencies

The new CPU sidecar reviews the actual downloaded LLVM, independently of the old source-only proof. It parses the complete kernel CFG, computes dominators and traces global memory-pointer provenance. In each of the eight producer/reducer records:

1. The CTA sequence bound dominates the two `cu_q` scalar loads. Their addresses depend only on CTA x and constant+1, not lane/tid.
2. Both i32 values are sign-extended before the i64 subtraction. The exact bad predicate is `(end-start)!=1 OR start<0 OR end>2051`; its bad edge only returns.
3. The valid-decode successor dominates **all** non-cu_q global loads/stores, including lengths/table, Q/K/V and partials. This assumes metadata is immutable during a launch, as in the tested stream/Graph protocol.
4. Q/output addresses depend on `q_start`; compact partial addresses retain sequence-index dependence without `q_start`. No prefill/invalid path accesses user data.

These are LLVM CFG/SSA facts, not a new opaque intrinsic C/D-component ABI claim or an independent machine-CFG proof. All original full PV C vectors and softmax/reducer expressions remain; no native padding accumulator is deleted.

| Actual role | LLVM guard line | First non-cu_q access | First global store (4K / 16K) |
| --- | --- | --- | --- |
| BF16 producer | 36 | 41 | 1642 / 1642 |
| K8 producer | 40 | 45 | 2246 / 2246 |
| BF16 reducer | 36 | 41 | 188 / 230 |
| K8 reducer | 36 | 41 | 188 / 230 |

The full actual file names/hashes and per-access lines are in `mixed{4k,16k}-evidence-cpu-v1.json`, under `actual_ir_records[].new_cpu_llvm_cfg_audit`. The guard sidecar rejects64 mutations of the actual LLVM across both cases: wrong sequence bound, wrong qlen, unsigned extension, inverted/bypassed bad edge, ordinary/intrinsic early loads and query/output using seq instead of q_start. These altered strings are CPU negative fixtures, never compiled or represented as device runs. The original `actual_metadata_cfg_verified=false` fields in historical result files are not edited; the new validator report is separate.

Actual original typed ABI and producer mechanism checks replay unchanged: BF16 has32 static native BF16 MMAs including16 PV calls with complete scaled C inputs; K8 has8 native I8 QK plus16 BF16 PV MMAs, and preserved C chains. Instruction counts describe static full/tail code, not a speedup measurement.

| Actual post-launch resource | Registers | Spill | Shared bytes |
| --- | --- | --- | --- |
| BF16 producer | 93 | 0 | 8192 |
| K8 producer | 144 | 0 | 6144 |
| Both reducers | 11 | 0 | 1024 |
| Encoder | 12 | 0 | 0 |

These are original runtime post-launch records tied to each actual hash/cubin; CPU does not independently query the device. Pre-launch null registers remain unknown and are not used as equality proof. The K8 producer has an observed resource cost; lower MMA count alone cannot establish throughput.

## Immutable package and reproduction

All172 original files are individually SHA-verified before packaging. Each run archives84 raw files and249 Git4a source blobs. Each run explicitly omits only two generated pycache files, with their complete SHA/size descriptors in the index; original local and remote directories remain retained. Original large NPZ files are split into ≤1MiB byte chunks and reconstructed against their original SHA. Tar parts stay≤2MiB:4K21 parts/22,137,287B;16K20 parts/21,860,372B. The package contains our source/IR/ASM/cubin, actual generated compiler launcher/utility cache sources/binaries and synthetic seeded component data, no model weights, user conversations, private SQLite or SSH material.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_konly_mixed_evidence.py --case mixed4k --arrays
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_konly_mixed_evidence.py --case mixed16k --arrays
```

Omit `--arrays` for stdlib-only SHA/source/actual ABI/MMA/CFG replay. No command above imports Torch/Triton/vLLM or executes device source. Original source controls remain in `biv150_konly_mixed_decode_20261027.md`. This is private mixed-reader implementation evidence; encoded-prefill/model/whole-service precision and performance remain open. The separate old56 stock-cache pure-decode102/105 signal cannot close those gaps.
