"""Install the measured decode launch policy without editing installed sources.

The Triton program, grid and reduction stay the pinned original. The proxy
sets compiler launch options only for the audited MiniCPM BF16 3D shape.
"""
import hashlib
import importlib
from pathlib import Path


EXPECTED = {
    'vllm.v1.attention.ops.triton_unified_attention':
        '54c495dcea404582a1526a96ae7e472965cbd03c8fa0931c71ef72223bf04e1f',
    'vllm.v1.attention.ops.triton_attention_helpers':
        '59d9d34bbda988436593cfd9bcb05015d17db8ee85c50f6697e45f987acbf0b1',
}


class DecodeLaunch:
    def __init__(self, original):
        self.original = original
        self.selected_launches = 0

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __getitem__(self, grid):
        launch = self.original[grid]

        def invoke(*args, **kwargs):
            if (kwargs.get('IS_3D') is True and
                    kwargs.get('HEAD_SIZE') == 128 and
                    kwargs.get('num_query_heads') == 16 and
                    kwargs.get('num_queries_per_kv') == 8 and
                    kwargs.get('BLOCK_M') == 16 and
                    kwargs.get('BLOCK_SIZE') == 16 and
                    kwargs.get('TILE_SIZE') == 16 and
                    kwargs.get('NUM_SEGMENTS_PER_SEQ') == 16 and
                    str(getattr(kwargs.get('query_ptr'), 'dtype', None)) == 'torch.bfloat16' and
                    not kwargs.get('USE_FP8') and
                    not kwargs.get('Q_IS_FP8') and
                    kwargs.get('KV_QUANT_MODE') == 0 and
                    not kwargs.get('USE_TD') and
                    not kwargs.get('USE_TD_QO') and
                    kwargs.get('USE_CAUSAL') is True and
                    kwargs.get('SLIDING_WINDOW') == 0 and
                    not any(kwargs.get(name) for name in (
                        'USE_ALIBI_SLOPES', 'USE_ALIBI_SQRT', 'USE_QQ_BIAS',
                        'USE_SOFTCAP', 'USE_SINKS', 'USE_PER_SEQ_CAUSAL', 'USE_MM_PREFIX')) and
                    kwargs.get('CHUNK_SIZE') == -1 and
                    kwargs.get('CHUNK_LOOKBACK') == -1 and
                    kwargs.get('num_warps', 4) == 4 and
                    kwargs.get('num_stages', 3) == 3):
                kwargs['num_warps'] = 2
                kwargs['num_stages'] = 1
                if self.selected_launches == 0:
                    print('BIV150_DECODE_LAUNCH_FIRST_SELECTED warps=2 stages=1', flush=True)
                self.selected_launches += 1
            return launch(*args, **kwargs)

        return invoke


def install():
    modules = {}
    for name, expected in EXPECTED.items():
        module = importlib.import_module(name)
        actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError('unaudited attention source: ' + name + ' ' + actual)
        modules[name] = module
    attention = modules['vllm.v1.attention.ops.triton_unified_attention']
    if attention.current_platform.vendor_name != 'iluvatar':
        raise RuntimeError('decode launch policy requires iluvatar')
    backend = importlib.import_module('vllm.v1.attention.backends.triton_attn')
    if backend.unified_attention is not attention.unified_attention:
        raise RuntimeError('attention backend does not use audited function')
    if attention.unified_attention.__globals__ is not vars(attention):
        raise RuntimeError('audited function does not resolve the target kernel global')
    if not isinstance(attention.kernel_unified_attention, DecodeLaunch):
        attention.kernel_unified_attention = DecodeLaunch(attention.kernel_unified_attention)
    return {'installed': True, 'num_warps': 2, 'num_stages': 1,
            'scope': 'audited BF16 MiniCPM 3D decode only'}
