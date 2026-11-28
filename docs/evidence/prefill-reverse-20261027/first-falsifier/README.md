# Original unconditional-reversal first falsifier

Frozen source: fcaec8af5e5bb674c16920134c5101a22e853e92.
The original result has completed=true/errors=[], with numerical/Q-hoist guards and known launch/shared metadata checks passing for chunk4k and chunk16k. Median speedups are respectively
1.0382605009776722x and 0.9671327473679263x. Preserve both observations and keep
Draft while designing a general metadata route; two cases are not full acceptance.

`result.json`, the owner JSON and log are unmodified downloads whose SHA256 values
are recorded in `summary.json`. Owner PID603517/start340638734 was terminal Z.
No GPU work or large evidence packaging was started while downloading these files.
All original IR remains in the completed remote experiment directory for later
archiving. `precision_gate=false`, `official_score=false`; no service score.


The resource phase audit distinguishes pre-launch `candidate_resources` null
counts from actual `loaded_resources` and final case snapshots 113/0. Matching
kernel hash, metadata and ASM SHA bind those loaded counts to both exact binaries.
See `resource-phase-audit.json`. Nulls are not compared as evidence; direct driver
local/shared attributes and full resource equivalence remain unknown. The future
default strict gate requires observed counts. Opt-in record mode may time known
changed counts with explicit resource_changed attribution; it never converts
unknown counts into equality.
