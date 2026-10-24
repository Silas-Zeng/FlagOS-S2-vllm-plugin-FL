# Retain the 4K fast path while investigating the 16K regression

Parent is 8493f3d, preserving the exact 8d9f56d gate-plus-mixed-attention source
that measured 5404.54 total tokens/s on 4K (+12.38%) and 3145.57 on 16K (-6.49%).
The parent passed 101/105 precision but failed 16K throughput and TTFT. It remains
shared and excluded from main; this experiment does not discard that work.

This candidate retains original BF16 interior gate/up weights plus their INT8
cache, instead of discarding BF16 storage. At M0..16, both the MLP hook and the
linear-method hook fall back to the actual original methods. This restores the
baseline BF16 operator route, rather than the native C++ at::linear fallback.
At M>=17, the same native INT8 fused gate operator remains. The mixed attention
source, C++/CUDA math, layer scope and all other model operators are unchanged.
There is no case-label switch, scheduler change or serving-parameter change.

The small-M route no longer needs activation residual compensation because it
uses original BF16 weights and activations. The original residual primitive is
still registered and independently checked but is not this model's small-M
route. Metadata explicitly records compressed_only=false, residual_rows=[],
and small_m_original_bf16_rows=[0,16]. Do not reuse the parent's accuracy or
claim that the new candidate necessarily retains its measured 4K speedup.

Keeping original weights adds about 0.8 GiB of INT8 cache over a BF16 model and
substantially more storage than the compressed parent. Actual KV capacity and
memory peaks must be measured. This tradeoff may change real batch shapes;
that is a hypothesis to validate, not an established cause of the regression.

Before the candidate's private build and service acceptance, an independent
probe measures the unchanged shipping GQA kernel plus reduction at batches
30/31/32/33/40/64 and KV bounds4096/16896. It compares split widths2/4/8/16,
with actual packed KV strides, shuffled pages, ragged B40, NaN scratch,
sampled CPU FP64 references and three alternating graph timing rounds.
The existing default target256 policy changes width8 to4 across B31/B32.
This diagnostic does not install any new split policy; it may identify a
separate operator candidate for later full validation.

The owned serial driver then builds privately, checks the unchanged native
primitive/dispatch and mixed operator, requires full 105-question precision,
and runs stable warmup plus one official scored round per workload. Existing
request/token/throughput/TTFT gates remain enforced. It waits for the current
LM-head acceptance by PID/start identity and refuses a busy GPU/port.

Local syntax, Ruff and diff checks pass. GPU diagnostics, new route correctness
in a full model, precision and throughput remain pending. Team main was
reverified as3283672; no newer official framework branch is merged.
