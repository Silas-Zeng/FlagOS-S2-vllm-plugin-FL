"""Opt-in wider Q tile for large single-sequence prefill after prefix splitting."""
import importlib
import os

from vllm_fl.patches import biv150_q_fragment_service as qfragment

_MANAGER = None


class _UnavailableBeforeLaunch(RuntimeError):
    pass


class _VerifiedManager(qfragment.Manager):
    def _compile(self, *args, **kwargs):
        # The compiler callback has not launched a kernel. An unknown transform
        # must fall back to the existing BM128 route, never an unproved BM256.
        try:
            compiled = super()._compile(*args, **kwargs)
        except Exception as error:
            raise _UnavailableBeforeLaunch(str(error)) from error
        if compiled.hash not in self.optimized:
            raise _UnavailableBeforeLaunch('Q32 lowering proof did not pass')
        return compiled


class _Policy:
    def __init__(self, base, manager):
        self.base, self.manager = base, manager
        self.disabled = False
        self.reported = False

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
                        and 'ir_override' not in kwargs
                        and (m.device is None or kwargs['query_ptr'].device.index == m.device)
                        and not m.attention.torch.cuda.is_current_stream_capturing()
                    )
                except Exception:
                    eligible = False
            if eligible:
                selected = dict(kwargs, BLOCK_M=256, BLOCK_Q=32)
                selected_grid = (kwargs['query_ptr'].shape[0] // 32 + 1, 2)
                try:
                    compiled = m.wrapper[selected_grid](**selected)
                    if compiled is not None and compiled.hash in m.optimized and not self.reported:
                        self.reported = True
                        print('BIV150_PREFILL_Q32_SELECTED', {
                            'hash': compiled.hash, 'rows': kwargs['query_ptr'].shape[0],
                            'num_sequences': 1, 'block_m': 256, 'block_q': 32,
                        }, flush=True)
                    return compiled
                except _UnavailableBeforeLaunch as error:
                    self.disabled = True
                    print('BIV150_PREFILL_Q32_FALLBACK', str(error), flush=True)
            return original(*args, **kwargs)

        return invoke


def install():
    global _MANAGER
    if os.environ.get('VLLM_FL_BIV150_PREFILL_Q32', '0') != '1':
        return {'installed': False, 'reason': 'disabled'}
    if _MANAGER is not None:
        return {'installed': True, 'reason': 'already installed'}
    base = qfragment._MANAGER
    if base is None or base.holder.get(base.kernel_key) is not base.wrapper:
        return {'installed': False, 'reason': 'audited Q-fragment binding required'}
    jit = importlib.import_module('triton.runtime.jit')
    raw = base.raw
    candidate = jit.JITFunction(
        raw.fn, version=raw.version, do_not_specialize=list(raw.do_not_specialize),
        do_not_specialize_on_alignment=list(raw.do_not_specialize_on_alignment),
        debug=raw.debug, noinline=raw.noinline, repr=raw._repr,
        launch_metadata=raw.launch_metadata,
    )
    private = {base.kernel_key: base.original_kernel}
    manager = _VerifiedManager(base.attention, base.backend, private,
                               base.kernel_key, raw, candidate, block_m=256)
    manager.install()
    base.holder[base.kernel_key] = _Policy(base.wrapper, manager)
    _MANAGER = manager
    return {'installed': True, 'block_m': 256, 'block_q': 32,
            'minimum_query_rows': 1024, 'num_sequences': 1,
            'fallback': 'existing verified BM128 Q fragment'}
