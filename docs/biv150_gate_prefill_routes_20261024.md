# Diagnose gate prefill routes before another model change

The residual gate source 20c991a passes Level-3 at 102/105 and improves warmed
4K throughput 9.34%, but its completed 16K is 7.44% below the original reference
and fails the working TTFT limit. End-to-end measurements do not identify the
responsible kernel. This isolated probe tests whether the current INT8 route
is slower on large prefill M, without changing any serving code.

Compare all five complete gate/SwiGLU paths on the same synthetic weights:

- Original BF16 weight layout and FlagGems mm plus fused activation.
- Packed contiguous KN BF16 weights with the same mm and activation.
- Cached BF16 dequantized weights, for diagnostic lower bound only.
- Existing native gate_silu_auto, including activation quantization and epilogue.
- Transient BF16 weight dequantization plus FlagGems mm and fused activation.

The final route includes full-weight dequantization in every timed call. It
does not assume a persistent BF16 weight cache. Materializing one layer is a
different hypothesis from previously tested fused W8A16 Triton GEMM, which was
slower. Sample M1/8/16/31/32/40/63/64/256/1024/2048; M31 and M63 cover shapes
seen in mixed service work. Timing alternates route order across three rounds.

Require exact dequantized weight reconstruction, exact dynamic/cached BF16
outputs, sampled CPU FP64 projection/activation checks and graph replay at
M8/M31 with changed inputs and NaN-filled temporary weights. Transfer to CPU
before FP64 conversion on this CoreX device. Relative error versus original
BF16 is reported separately for both quantized routes. This remains synthetic
operator evidence, not model precision or whole-service speedup.

The diagnostic holds several weight copies and captures multiple temporary
buffers for timing. Its memory footprint is not a model KV-capacity estimate.
Any service route requires fresh full-model precision, memory and performance
acceptance. No claim is made that transient dequantization is faster.

Local syntax, Ruff and diff checks pass; GPU execution is pending. The serial
queue waits on the current owned acceptance process identity and verifies GPU
idle, port free and the existing audited native SO hash before testing.
