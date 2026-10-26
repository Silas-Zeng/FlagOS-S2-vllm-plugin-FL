# Prefill tile/pipeline component exploration — no GPU result yet

The existing PR46 Q32 lineage remains experimental. This branch now includes its strict-binding corrections and current team main34942f7, including the BF16 small-row correction. The only merge conflict was EXPECTED_RUNNER: it now hashes the actual combined runner268804cb; removing only the Q32 installer exactly reconstructs the team runnercbbbabce. All local strict module pins and inverse Q-fragment source reconstruction pass. Q32 controller CPU gates pass. Old partial/confounded model results remain unchanged in their original evidence directories.

The old whole-round v2 diagnostic on the earlier baseline attributes209.50/555.89 retained seconds to non-Graph prefill attention in16K (about37.7%). This is a prioritization clue, not the current candidate's measured hotspot share or an uninstrumented score. Q-fragment hoisting is already main; this probe keeps that mechanism and tests loop/data reuse, rather than claiming it again.

Bounded next configurations: BM256/K64/w16/stage1, BM256/K128/w16/stage1, then BM128/K64/w8/stage2 and BM256/K64/w8/stage2. Compare each against BM128/K64/w8/stage1. Existing BM256/Q32 uses202 registers and64KiB shared; wider KV tiles or extra stages may exceed resources. Record actual registers/spills/shared and lowering proof. All numerical/changed-input poisoned Graph gates must pass before timing. CPU CLI/source checks do not establish stage2 lowering correctness, hardware support or speed.

Production defaults and service/benchmark parameters are unchanged. No new GPU task is queued: PR58 owns the GPU and its exact-source full105 matched control has priority. After that qualification, select the next component from the measured remaining bottleneck; this Draft is not merge-ready.

## Q64 bounded resource hypothesis,2026-10-26

Actual target autotuning rejects shared196608>131072, establishing a131072B
per-kernel launch limit; it does not establish shared capacity perSM or resident
block count. Pinned actual Iluvatar compiler a97f0670 creates CompiledKernel
without loading handles. The shared check is in _init_handles, triggered by
run/launch metadata hook/__getitem__. Q-fragment Manager compiles normal IR,
inspects hash/metadata/asm (no handle init), then transforms/compiles before
launch. Thus a potentially oversized normal BM512 module can in principle be
transformed before launch; hardware/transform capability is still unproven.

Add only BM512/Q64,K64,w16,stage1 to the bounded component CLI and actual adapted
scope/compiler-option CPU gates. Q buffer alone is128KiB; if its lifetime can
reuse K/V shared after hoisting, the final module may reach the observed limit.
This is an estimate, not compiled resources; register pressure/spills and
lowered proof remain mandatory. Existing BM256w8 uses202registers but this is
not a measured BM512/w16 count and must not be extrapolated into occupancy.
No production file/default/service argument changed. All five adapted CPU
configurations pass; no new GPU task is queued while PR59 owns service
acceptance and its serial matched control. After those finish, compare the
bounded prefill candidates against actual main BM128/K64/w8/stage1, including
real2017/1985tails, independent CPU FP64 and changed-input Graph checks before
timing. Component timing is not a whole-model speedup.
