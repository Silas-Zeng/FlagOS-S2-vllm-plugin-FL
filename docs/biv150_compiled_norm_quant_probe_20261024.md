# Compiled native RMSNorm plus quantization probe

Base: validated team main5eda0b0 (PR25), not the closed PR27 experiment.
The former PR27 is closed and its7/18 evidence remains preserved. Its fusion
introduced a BF16 rounding before weight multiply because the source suggests
that cast, but actual selected Inductor output omits that round. This variant
keeps FP32 through multiplication and uses the same libdevice.rsqrt helper as
the inspected generated kernel. It retains BF16 residual storage and final
normalized-output storage, epsilon from the model, exact activation quantization,
32-row padded native INT8 GEMM and existing epilogues.

Only standalone probe entry points are added; production dispatch is unchanged.
The18-case/6-pattern check still compares against the compiled installed native
implementation, independent CPU normalization, exact self-quantization and the
complete gate at unchanged0.005/0.005 tolerance, plus changed-input Graph replay.
CPU reference now models compiled FP32 weighting rather than the eliminated
source cast. Each failure preserves the pattern/input/weight/actual reference in
a private CPU fixture and records a checksum. Three alternating warmed component
rounds follow numeric success. No model precision or throughput is claimed.

The two-pass reduction tiling selected by Inductor may still differ from this
one-row fusion. The libdevice/rounding corrections are hypotheses to test; they
do not by themselves establish bit identity, performance or service integration.
If numeric or component performance validation fails, archive and close this
variant; retain the general fusion direction. Draft PR precedes GPU testing.
Private GPU work must wait for owned PR28 coverage and PR26 reference audit;
never overlap their measurements or change shared packages/competition settings.


## Result:15/18 passed; full-range variant closed

Frozen4a9e7c2 built successfully and passed existing9 gate checks.15/18 fusion
records passed all their data patterns, CPU/self-quantization/full-gate/Graph
checks, with complete-gate gains around2.4–6.1% on successful tested rows.
The pre-weight/rsqrt correction removed the earlier broad random-weight mismatch.
It did not settle the remaining reduction/layout differences: M40 random has1
normalized BF16 mismatch and15/245760 full-gate tolerance violations; M2048
ones/random have4/6 norm mismatches and9/14 out of12582912 gate violations.
Max full-gate difference0.03125, against unchanged0.005/0.005. Self-quantization
on the candidate normalized values remains bit-exact. All three failures are
on the normal pattern, not a corrupted native quantization path.

Raw result and9 checks are retained under `evidence/compiled-norm-quant-20261024/final/`.
Failure input/weight/actual-native-reference CPU fixtures remain in the private
compiled-norm-quant-20261024/probe directory with SHA checksums in the report.
No model installed or score credited. Close this full-range variant with evidence.
The general direction and successful tested M1/8/16/17/32/64/256 pairs
remain useful for a later bounded fusion or an exact compiled-layout rewrite.
Do not select M40 based on its ones-only success, extrapolate to untested rows,
loosen tolerance or describe15/18 as full success.
