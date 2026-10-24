# Residual gate on current team integration

Base: team main 1ea082f79db65a66f78fdd7b1d958335450fa9f4 (PR22+PR23).
Operator source is byte identical to the measured residual-gate20c991a/8d9f56d:
INT8 gate/up in middle34 layers, small M1..16 activation residual, native fused
GEMM/SwiGLU. QKV/O/down and the first/last4 layers remain BF16.

The earlier residual-only candidate observed4K+9.34%,16K−7.44%; the separate
mixed-attention combination observed4K+12.38%,16K−6.49%. Both passed accuracy
but failed16K throughput/TTFT. Those numbers do not describe this new candidate.

PR22 pins the exact model runner and audits the attention wrapper stack. Simply
adding an operator installer to that runner changes its hash and disables the
team compute-order path. Adding the older remaining-mixed-attention wrapper also
changes its required top-level wrapper. This candidate therefore preserves all
team attention and Q-fragment modules byte-for-byte and adds only the gate installer.
The compute-order helper changes only EXPECTED_RUNNER to the new appended runner
SHA256 677def25d5391887145b0bbabb2222531887c6f7c48b6ea53c1859ad08d8ef61.
Local AST comparison against the final team baseline must confirm every other helper statement unchanged; the
runner's original contents remain an exact prefix. No scheduling, request order,
serving limits or attention implementation is changed by this candidate.

The team's compute-order path gathers scattered single-query rows into the
existing prefix fast path when its audited scope allows. It may reduce the need
for the separate remaining-mixed operator in those calls; this is an integration
hypothesis, not a throughput result or proof all mixed calls use that path.

Owned serial queue: private SO build,9 operator checks,105-question accuracy,
then stable warmups and one formal official4K/16K round. The acceptance runner
requires residual installation metadata and post-ready compute-order selection,
requires Q-fragment install/selection and zero fallback, rejects a scope-rejection marker, and checks request counts, total tokens,
throughput and TTFT. The manifest hashes the exact native/runner/attention/helper
sources and loaded SO. All results require fresh model and service validation.
The source and output are distinct from all older experiments.

Local validation: Ruff, syntax, AST/prefix/source comparisons, git diff check.
No GPU or full-model success is claimed. The queue waits behind the expanded
GQA probe, using both predecessor PID and start ticks; it refuses a busy GPU or
port and never stops another service.

Before publication team main advanced from PR22 to PR23. The unpublished local
commit was rebased, resolving both runner-tail and runner-hash conflicts by
preserving the entire final team runner followed by the gate-only installer and
updating only the final helper hash. Team Q-fragment implementation stays exact.
PR23 reports4K5313.59/16K3764.67 and100/105 on its instance; none of those
results substitutes for this personal-instance candidate's fresh validation.
