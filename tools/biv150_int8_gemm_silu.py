"""Component entrypoints use the exact same kernels as the bounded service route."""
from vllm_fl.patches.biv150_int8_gemm_silu import CONFIGS, fused_gate, from_quantized, full_gate
