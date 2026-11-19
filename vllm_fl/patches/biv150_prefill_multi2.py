"""Verified BM256/w16 prefill for two sequences; independent of the Q32 policy."""
import hashlib
import importlib
import os
from pathlib import Path

from vllm_fl.patches import biv150_prefill_q32 as q32
from vllm_fl.patches import biv150_q_fragment_service as qfragment

_ENABLE_ENV = 'VLLM_FL_BIV150_PREFILL_MULTI2'
_EXPECTED_Q32 = '3906f34e89b814cafa94ef04a0f268ffc3cf237b99ecb011a62c9abfef56878b'
_EXPECTED_FRAGMENT = 'c3c5bb34086681918cc328a6bbc2b349dd9019899b730fc99f9b2dc02f3ae1a9'
_MANAGER = _POLICY = None


class _UnavailableBeforeLaunch(RuntimeError):
    pass


class _VerifiedManager(qfragment.Manager):
    def _compile(self, *args, **kwargs):
        # This callback runs before the candidate launch. Never return an
        # unproved BM256 kernel when the accepted fallback uses BM128.
        try:
            compiled = super()._compile(*args, **kwargs)
        except Exception as error:
            raise _UnavailableBeforeLaunch(str(error)) from error
        if compiled.hash not in self.optimized:
            raise _UnavailableBeforeLaunch('Multi2 lowering proof did not pass')
        return compiled


class _Policy:
    def __init__(self, base, manager):
        self.base, self.manager = base, manager
        self.disabled = self.reported = False

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __getitem__(self, grid):
        original = self.base[grid]

        def invoke(*args, **kwargs):
            m = self.manager
            eligible = False
            if not args and not self.disabled:
                try:
                    eligible = (
                        kwargs.get('num_seqs') == 2
                        and kwargs['query_ptr'].shape[0] >= 1024
                        and qfragment._eligible(kwargs)
                        and m.active
                        and m.async_mode.get() is None
                        and not m.raw.pre_run_hooks
                        and not m.candidate.pre_run_hooks
                        and 'ir_override' not in kwargs
                        and (m.device is None or kwargs['query_ptr'].device.index == m.device)
                        and not m.attention.torch.cuda.is_current_stream_capturing()
                    )
                except Exception:
                    eligible = False
            if eligible:
                selected = dict(kwargs, BLOCK_M=256, BLOCK_Q=32, num_warps=16)
                # The SDK reserves an extra program PER sequence. A single
                # sequence +1 drops a live tail for e.g. 513+513 or 1025+1.
                selected_grid = (kwargs['query_ptr'].shape[0] // 32 + kwargs['num_seqs'], 2)
                try:
                    compiled = m.wrapper[selected_grid](**selected)
                except _UnavailableBeforeLaunch as error:
                    self.disabled = True
                    print('BIV150_PREFILL_MULTI2_FALLBACK', str(error), flush=True)
                else:
                    if compiled is not None and compiled.hash in m.optimized and not self.reported:
                        self.reported = True
                        print('BIV150_PREFILL_MULTI2_SELECTED', {
                            'hash': compiled.hash, 'rows': kwargs['query_ptr'].shape[0],
                            'num_sequences': kwargs['num_seqs'], 'grid': selected_grid,
                            'block_m': 256, 'block_q': 32, 'num_warps': 16,
                            'num_stages': 1, 'tile_size': 64,
                        }, flush=True)
                    return compiled
            return original(*args, **kwargs)

        return invoke


def install():
    global _MANAGER, _POLICY
    if os.environ.get(_ENABLE_ENV, '1') != '1':
        return {'installed': False, 'reason': 'disabled'}
    for module, expected in ((q32, _EXPECTED_Q32), (qfragment, _EXPECTED_FRAGMENT)):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            return {'installed': False, 'reason': 'unaudited parent source'}
    base, parent = qfragment._MANAGER, q32._MANAGER
    if base is None or parent is None:
        return {'installed': False, 'reason': 'audited Q32 parent required'}
    outer = base.holder.get(base.kernel_key)
    if _MANAGER is not None:
        intact = (outer is _POLICY and _POLICY.manager is _MANAGER
                  and _MANAGER.active and type(_POLICY.base) is q32._Policy
                  and _POLICY.base.manager is parent and _POLICY.base.base is base.wrapper)
        return {'installed': bool(intact), 'reason': 'already installed' if intact else 'binding changed'}
    if (type(outer) is not q32._Policy or outer.manager is not parent
            or outer.base is not base.wrapper or outer.disabled
            or not base.active or not parent.active or parent.raw is not base.raw
            or (base.block_m, base.num_warps, base.tile_size, base.num_stages) != (128, 8, 64, 1)
            or (parent.block_m, parent.num_warps, parent.tile_size, parent.num_stages) != (256, 16, 64, 1)):
        return {'installed': False, 'reason': 'unknown Q32 parent chain'}
    jit = importlib.import_module('triton.runtime.jit')
    raw = base.raw
    candidate = jit.JITFunction(
        raw.fn, version=raw.version, do_not_specialize=list(raw.do_not_specialize),
        do_not_specialize_on_alignment=list(raw.do_not_specialize_on_alignment),
        debug=raw.debug, noinline=raw.noinline, repr=raw._repr,
        launch_metadata=raw.launch_metadata,
    )
    if candidate.device_caches is raw.device_caches or candidate.device_caches is parent.candidate.device_caches:
        return {'installed': False, 'reason': 'private cache aliases parent'}
    private = {base.kernel_key: base.original_kernel}
    manager = _VerifiedManager(base.attention, base.backend, private, base.kernel_key,
                               raw, candidate, block_m=256, num_warps=16,
                               num_stages=1, tile_size=64)
    manager.install()
    policy = _Policy(outer, manager)
    base.holder[base.kernel_key] = policy
    _MANAGER, _POLICY = manager, policy
    return {'installed': True, 'block_m': 256, 'block_q': 32, 'num_warps': 16,
            'num_stages': 1, 'tile_size': 64, 'minimum_query_rows': 1024,
            'num_sequences': 2, 'grid': 'qrows//32+num_sequences',
            'fallback': 'unchanged Q32 parent chain and verified BM128'}
