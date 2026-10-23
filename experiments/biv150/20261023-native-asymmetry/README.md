# Native W8A8 4K / 16K performance asymmetry: historical log audit

The prior candidate beb3518 scored4K5318.130579 total tokens/s and
16K3094.243113; same-session BF16 baseline9a43eb1 scored4835.561243 /
3366.817953. The 16K regression remains unresolved; attention-dominated
work alone explains dilution, not the negative overall result.

Audit reads original server logs, score JSON and manifests, without rerunning
GPU work or changing framework arguments. Raw records, logger samples and
SHA256 identities are in log-audit.json. The formal-window approximation is
result.date minus duration through result.date (completion); periodic logger
samples can straddle phase boundaries and are not per-step tracing.

- Baseline16K: KV capacity522960 tokens;66 formal-window logger samples;
  running30 in40 samples,31 in9,29 in5.
- Native16K: KV capacity545616 tokens (the earlier4K service had545712);
  71 samples; running31 in36,32 in14,30 in9,33 in2.
- Native4K:25 samples; running64 in13,63 in3,62 in6; peak whole-log
  KV utilization58.8%, while both16K logs reached100%.
- None of these three complete service logs contains a preemption/recomputation
  warning. There is no observed evidence supporting a recomputation explanation.
  Absence of logged warnings is not per-step proof of scheduler behavior.

The observed active-batch distribution changed when compressed weights freed
KV memory. A shape/path change remains a causal hypothesis, not an established
explanation of the full8% regression. Captured graph padding can also differ
from logger running counts, so one must not infer kernel width directly from
these histograms. Next diagnostic should compare actual captured shapes and
per-layer/kernel timings around the30–33-active-request regime, with all official
performance results retained. No failed native candidate is merged on this basis.
