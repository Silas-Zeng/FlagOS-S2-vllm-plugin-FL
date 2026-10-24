# Original real-BF16 M8 API fallback diagnostic

A separate process with no fused-module import reproduces large repeated-original results and CPU-reference failures for M8/real-BF16 weights under worker-default FlagGems/inference mode. Compressed M8 and other tested shapes remain exact. Do not classify this as a proven new M64 regression, or call that comparison passed.

The raw archive preserves both probe source revisions, full results/logs and v1/v2/v3 preflight failures. Current preflight separately verifies exact original TensorImpl/min_rows forwarding; selected M32/M64 strict numerical checks are unchanged. The service installer bypasses the patched gate forward for layers lacking _biv150_native_wq, so this synthetic regular-BF16 API test does not prove actual model coverage of the problematic native fallback. Full official model acceptance remains required.
