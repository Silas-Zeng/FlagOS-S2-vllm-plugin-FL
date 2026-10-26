"""Audited K128 prefill with a bounded register budget, retaining Q32 fallback."""
import importlib
import os

from vllm_fl.patches import biv150_prefill_q32 as q32
from vllm_fl.patches import biv150_q_fragment_service as qfragment

_MANAGER = None


class _Manager(q32._VerifiedManager):
    def _compile(self, src, target=None, options=None, _env_vars=None):
        if options is None or options.get('maxnreg') != 128:
            raise q32._UnavailableBeforeLaunch('K128 requires the audited register budget')
        compiled = super()._compile(src, target=target, options=options, _env_vars=_env_vars)
        # Loading the function is cold preparation, before any device launch.
        # Refuse to trade extra spills/shared memory for a nominal register cap.
        try:
            compiled._init_handles()
            if (compiled.n_regs > 128 or compiled.n_spills != 0
                    or compiled.metadata.shared > 65536):
                raise RuntimeError('K128 resource budget was not retained')
        except Exception as error:
            raise q32._UnavailableBeforeLaunch(str(error)) from error
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
                        kwargs['query_ptr'].shape[0] >= 1024
                        and kwargs.get('num_seqs') == 1
                        and qfragment._eligible(kwargs)
                        and m.async_mode.get() is None
                        and not m.raw.pre_run_hooks
                        and not m.candidate.pre_run_hooks
                        and 'ir_override' not in kwargs and 'maxnreg' not in kwargs
                        and (m.device is None or kwargs['query_ptr'].device.index == m.device)
                        and not m.attention.torch.cuda.is_current_stream_capturing()
                    )
                except Exception:
                    eligible = False
            if eligible:
                selected = dict(kwargs, BLOCK_M=256, BLOCK_Q=32, TILE_SIZE=128,
                                num_warps=16, maxnreg=128)
                selected_grid = (kwargs['query_ptr'].shape[0] // 32 + 1, 2)
                try:
                    compiled = m.wrapper[selected_grid](**selected)
                    if compiled is not None and compiled.hash in m.optimized and not self.reported:
                        self.reported = True
                        print('BIV150_PREFILL_K128_SELECTED', {
                            'hash': compiled.hash, 'rows': kwargs['query_ptr'].shape[0],
                            'num_sequences': 1, 'block_m': 256, 'block_q': 32,
                            'num_warps': 16, 'num_stages': 1, 'tile_size': 128,
                            'maxnreg': 128, 'n_regs': compiled.n_regs,
                            'n_spills': compiled.n_spills, 'shared': compiled.metadata.shared,
                        }, flush=True)
                    return compiled
                except q32._UnavailableBeforeLaunch as error:
                    self.disabled = True
                    print('BIV150_PREFILL_K128_FALLBACK', str(error), flush=True)
            return original(*args, **kwargs)

        return invoke


def install():
    global _MANAGER
    if os.environ.get('VLLM_FL_BIV150_PREFILL_K128', '1') != '1':
        return {'installed': False, 'reason': 'disabled'}
    if _MANAGER is not None:
        return {'installed': True, 'reason': 'already installed'}
    base = qfragment._MANAGER
    if base is None:
        return {'installed': False, 'reason': 'audited Q fragment required'}
    policy = base.holder.get(base.kernel_key)
    if (type(policy) is not q32._Policy or policy.manager is not q32._MANAGER
            or policy.base is not base.wrapper):
        return {'installed': False, 'reason': 'audited current Q32 binding required'}
    jit = importlib.import_module('triton.runtime.jit')
    raw = base.raw
    candidate = jit.JITFunction(
        raw.fn, version=raw.version, do_not_specialize=list(raw.do_not_specialize),
        do_not_specialize_on_alignment=list(raw.do_not_specialize_on_alignment),
        debug=raw.debug, noinline=raw.noinline, repr=raw._repr,
        launch_metadata=raw.launch_metadata,
    )
    private = {base.kernel_key: base.original_kernel}
    manager = _Manager(base.attention, base.backend, private, base.kernel_key,
                       raw, candidate, block_m=256, num_warps=16, tile_size=128)
    manager.install()
    base.holder[base.kernel_key] = _Policy(policy, manager)
    _MANAGER = manager
    return {'installed': True, 'block_m': 256, 'block_q': 32,
            'num_warps': 16, 'num_stages': 1, 'tile_size': 128, 'maxnreg': 128,
            'minimum_query_rows': 1024, 'num_sequences': 1,
            'fallback': 'current verified Q32 K64'}
