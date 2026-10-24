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
