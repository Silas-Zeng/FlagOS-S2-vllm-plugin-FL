# PR114 actual K-pair SO: compile/static evidence only

This docs-only child of `52e1d2680d41e44f025b79fccb85867c8e783d39` preserves the actual114 compile, independent audit and Root's separate numeric admission. It does not change production, kernel, consumer or test code. GPU numerics, Graph/side-stream behavior, whole-cache4B-but-not8B alignment, fullstep performance, occupancy and model/service qualification remain **unverified**.

Actual SO: `ef732ef832b6399bf156d24c705b15e2e9e766947c0a456745673953ffbfb8fd`, compared with actual2ad. The original owner PID70801/start356855231 ended in Z with the same start identity. Compilation completed with CUDA initialization recorded false before/after and every frontend launch/input-initialization counter0.

`original-full.tar.gz` is the complete original33-file archive (1,541,176 raw bytes), SHA `bc592793775bacf37351c56d95500c979dc52e71857a47ce4ec2ab723362f385`. The tar also includes the complete download manifest. All archive member sizes/hashes were checked during packaging. Raw files and old reports were not edited. The identical375,880-byte gzip is also retained in the workspace's `repos/experiment-raw-archives/2026-10-29/`.

The independent final report SHA is `49d64203b9eae093000b7dae9f176869fc460ac45f646002ee5d223ebfeeca53`. Root's Python3.12 replay is byte-identical. Root's separate admission SHA is `5fc79ec23dc50108eea06ee34e3c56d7f55bd2ca5d6f3774b2531919e71b1f61`; only `manual_Root_approval` and the added `Root_approval_scope` differ. Read that scope literally: it permits the stated isolated numeric tests and confers no device result.

## What the actual-object review establishes

The final SO's GPU images match its built objects. Source changes invert exactly, compiler/flags and complete metadata ABI match the reference except declared names/register counts, and codec text remains identical. Each producer has1541 static instructions versus1339 in the reference; both mutually exclusive paths are counted, so this is not a dynamic instruction-cost measurement. The291 canonical math lines and semantic flags match.

The actual aligned branch lowers K pair loads to `ml_lsa_load_dwordx2`; a uniform base-alignment test chooses the scalar fallback when required. The audit proves branch-zero reaches only vector loads and nonzero only scalar loads, both rejoin the K-ready barrier, and the original QK/P packing/V wait/tail/PV ordering remains. S2 code was compared for completeness but **is not admitted**; the host scope remains N48/S4.

Static VGPR changes100→97 and SGPR66→69. Shared stays8192B; private bytes, VGPR spills and SGPR spills remain0. These static counts do not prove a residency or performance change. Extra alignment branches and changed scheduling may cost more than the reduced load count.

Retaining scalar K loads does **not** establish that the whole cache can be placed at4B-but-not8B alignment. The V-SME path still needs actual device testing in both2ad and the candidate. No hardware-alignment result or waived numerical requirement is inferred from the static audit.

## Preserved audit mistake

`initial-review-branch-adjacency-assumption.py` incorrectly required the conditional branch to be immediately after every `base &7` test. In the actual tail, `sl_mov_b32 s15,0xffffffff` lies between the AND and `sl_cbr_wcrz`; the move writes that scalar register and does not write WCR. The final script requires this exact intervening instruction and adds branch reachability/exclusivity checks. This corrected an audit assumption, not device code or a numerical gate. No device failure is attributed to it.

The original initial script is retained byte-for-byte. `branch-adjacency-correction.diff` is a derived source comparison, and the JSON records exact actual instruction lines/PCs for both producers. The final script/report pair is separately hash-bound. The earlier intermediate report is not substituted for the final49d report.

## CPU replay and dependencies

The original script uses immutable Git objects at source52e, referencee1c5 and the shared b54 helper commit. All three already-shared helper originals are included here with exact Git paths/hashes in `helper-git-bindings.json`. It also needs the complete actual2ad compile archive already shared under `docs/evidence/native112-n48-compile-20261029/`; that prior archive is not duplicated here. Use the repository with these Git objects available, and extract archives into new directories.

```sh
mkdir /private/tmp/new-native114-raw
tar -xzf docs/evidence/native114-n48-kpair-actual-so-20261029/original-full.tar.gz -C /private/tmp/new-native114-raw
python3 -B docs/evidence/native114-n48-kpair-actual-so-20261029/review_actual_so.py --repo "$PWD" --raw /private/tmp/new-native114-raw --reference-raw /absolute/complete-actual2ad-compile-raw --report /private/tmp/new-native114-static-review.json
```

Packaging checked original hashes, report identity, exact Root admission differences and the complete archive. It did not rerun the full CFG audit, compile code, import Torch, use SSH, launch GPU work or fabricate a later numerical result. The geometry notes a19 were not merged into this child; they are a separate optional dependency.
