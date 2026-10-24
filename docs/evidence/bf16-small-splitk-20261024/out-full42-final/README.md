# Complete actual-out baseline /42-layer BF16 component run

Runtime755404f is unchanged mathematics with a corrected workload reference: torch.mm(x, original same-storage N×K weight view, out=caller buffer), worker-default FlagGems enable policy and inference mode. Actual dispatch tables confirm the CUDA registration in flag_gems/__init__.py; source attribution alone was not treated as proof. The original returning-baseline81-case results remain separate.

All1134/1134 records complete:644 five-pattern full CPU FP64→BF16 candidate passes,602 changed-input Graph CPU passes,540 native CPU passes. All use original atol/rtol .005/.005. Numerous strict failures remain and are not recategorized as passes. Inputs are synthetic,42 real learned QKV/O/down weight identities are recorded. Positive complete-call split4 counts among42 layers are QKV M1/M8/M16=37/25/16;O=40/30/18;down=38/14/7. Minimum speedups in these complete positive sets are1.785/2.120/2.280;1.693/2.121/1.808;1.360/1.595/1.551 respectively. This does not establish model speed or uniform numerical eligibility.

The metadata archive preserves every result, original strict flag, logs and per-file SHA index for all701 .pt counterexamples. The complete127MB compressed archive is preserved on the remote exports path and downloaded to the author's ignored project artifact directory; archives.json identifies both and its SHA. Large tensor artifacts are not embedded in Git. The prior real-oracle81 full binary archive remains in Git.

Next analyze saved BF16 error/ULP distributions on CPU without changing original gates. Any proposed model pilot must explicitly preserve component failures and pass official full Level3/fixed service/TTFT checks. Do not fit a production layer whitelist to these synthetic fixtures.
