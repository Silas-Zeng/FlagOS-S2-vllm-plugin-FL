# PR110 actual 7892 SO: independent CPU review

The final library's S2/S4 full-page and tail-page paths preserve K-before-V issue and place V wait after QK and P softmax/packing, before V tail clearing/PV. Resources match actual compact40f9: 100 VGPR, 66 SGPR, 8192-byte shared, zero private bytes and spills. This is a recommendation for Root's independent review before the original four-case device consumer; no GPU, Graph, side-stream, fullstep, capacity, model or performance acceptance is granted here.

Frozen input commit: `4d1166792a5247b774d6b1977e60466574e6a34c`. This evidence commit changes no kernel, compile frontend, numerical frontend, writer or production file. Its library is `7892a6aad09d247a322a74906312f250a9d253160202fee23259adac7b7260d4`; the arithmetic/layout reference is actual compact library `40f9b17c634d2344ca54b9f1202e4e10932bf3f3f632cd41d793ddafd0934453`. No qualification is inherited from 8b86 or 40f9.

The audit rehashes all 33 new originals (1,408,237 bytes), both actual-SO inspection exports, and all old compact originals. It binds 2,764 decoded instruction words to embedded final-library ELF `.text`, including the codec, and binds embedded GPU ELFs to the actual compiled objects. S2/S4 each contain 1,339 decoded instructions. Both kernels have 291 arithmetic IR lines alpha-identical to actual40f9, retaining operation order, constants, FP flags and operand sharing within that slice. The unchanged tail-clear loop/guard and final output slice are checked separately; the intended outer tail merge/barrier placement changes with the six source edits, so whole tail-wait regions are not claimed identical.

## Actual execution evidence

All line numbers below refer to the new `out/actual-so-sass.stdout.txt`. Full-page wait/PV blocks appear physically before the QK block; branch CFG and conditional packing paths, rather than line-number order, establish execution order.

| Segment/path | K-ready barrier | V SME | QK | P final OR packing | V wait | Tail clear | V-ready barrier | PV MMAs |
|---|---:|---|---|---|---:|---|---:|---|
| S2 full | 719 | 740, 742, 743, 747 | 736, 745, 749, 751 | 826, 831 | 609 | none | 615 | 641–679 |
| S2 tail | 942 | 950, 953, 956, 959 | 978, 982, 984, 986 | 1083, 1087 | 1090 | 1125, 1137 | 1148 | 1175–1216 |
| S4 full | 2060 | 2081, 2083, 2084, 2088 | 2077, 2086, 2090, 2092 | 2167, 2172 | 1950 | none | 1956 | 1982–2020 |
| S4 tail | 2283 | 2291, 2294, 2297, 2300 | 2319, 2323, 2325, 2327 | 2424, 2428 | 2431 | 2466, 2478 | 2489 | 2516–2557 |

The backend interleaves full-page QK and V SME issue after K is ready. It does **not** put every softmax instruction before wait: full-page rho has one exp before and one after wait; both tail rho exps and both running-L FMAs are after wait. The exact reviewed field is `actual_V_wait_after_QK_and_P_softmax_packing`.

Actual IR address chains use signed page × 12544 bytes for K/V/scales, a host-preoffset V pointer at +4352 bytes with no second device offset, an internal scale offset of +4096 bytes, and the unchanged 512-byte SME descriptor. V/scale equivalent page strides are 6272 BF16 / 3136 F32 elements. Host typed guards are checked in the exact compiled C++ snapshot plus the binary guard string; host machine code was not disassembled. Actual codec `.text` is byte-identical to 40f9 (`936ce115af95401dda3a37cf310b62baa606003a18910ff46de4707fbaba19cf`); actual codec IR retains three unflagged F32 divides, two explicit RNE conversions, and its clamp/scale constants. This does not replace device RN32/FTZ or numerical conformance checks.

## Reproduce locally (CPU only)

From this repository/worktree, with both downloaded raw directories intact:

```sh
python3 -B docs/evidence/native-k8-compact-v-overlap-actual-so-20261029/review_actual_so.py \
  --raw /private/tmp/cjj-native110-compile-v1-raw-20261029 \
  --control-raw /private/tmp/cjj-native-k8-compact-compile-v1-raw-20261028 \
  --report /private/tmp/cjj-native110-independent-review-replay.json \
  --review /private/tmp/cjj-native110-consumer-review-replay.json
```

Expected output hashes:

- Script: `99257bae454408161c50f00053efaca939b86f6d5a49aaee57247e76cac907a2`.
- `review.json`: `0b3d0098f3e8b01b553fc75b44dbe6c5eb4b898d61076149f96e02fc09697708`.
- `consumer-review.json`: `23570370a001f0d5df4b2750b92b48a2b4e7492e2bdfa90a9939e45b86fbbe53`.

The candidate receipt binds the actual compile summary, inspection, IR, SASS, library and source proof, and passes the unchanged CPU `contract.admission`. Exactly the six reviewed static flags are set true; an unknown review field fails closed. It explicitly records `manual_Root_approval=false`; Root must independently replay/review before using it for a GPU run. Root approval is not asserted by this artifact. Remaining device/service/performance fields stay false. Original raw artifacts remain read-only at their supplied locations; this commit includes no large raw copy and does not compile/load the library.

Limitations: the IR is the hash-bound actual `ixobjdump --IR` export, not an independent bitcode decoder; the static comparison is not a whole-program equivalence proof. Embedded resource metadata does not establish loaded occupancy or latency hiding. The only prepared consumer scopes remain N64/S2 and N32/S4; N40/N48, writer/fullstep, 42-layer capacity and actual model accuracy require their own real runs.

## Subsequent Root authorization (separate record)

Root independently read the actual IR/SASS and replayed the initial audit with Python 3.12 byte-identically, then supplied `root-approved-fourcase-review.json` (`90f6e18e142813e732694f453779b876048daaa8c122fd3e2d985dc2969cdb17`). This original record is preserved without modification and authorizes only the original four-case synthetic consumer. It is not a claim that the consumer or any performance/model test passed.

The approved chain is preserved as `initial-review-root-replayed.json` (`f5a40726b7ae1c1db122e710c1921815db887a1f06112901c842f976a1c9d0dc`) and `initial-consumer-review.json` (`70e1c29e51cb03c7ff9ca8f5eab925ad9b97ddf585d983e734d240d2a673e4f0`). The final script adds only an explicit six-field allowlist so a future unknown review field cannot be granted automatically. `initial-to-final-script.patch` records that complete change; reversing it reconstructs initial script SHA `02fc4aff949091969a058bd1ffed64465b9a5baa73b1e7d57d6cf9178bcaa271` exactly. The two audit JSONs differ only in `script_sha256`; the two unapproved receipt JSONs differ only in `independent_audit_report_sha256`. All actual SO checks, identities, resources, conclusions and limitations are identical. These equivalences were checked locally before this evidence was committed. Root approval remains a separate supplied record, never inferred by the agent's audit script.
