"""CPU-only controller fault checks; importing this file never imports Torch."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    source = Path(__file__).resolve().parents[1] / 'vllm_fl/patches/biv150_prefill_q32.py'
    tree = ast.parse(source.read_text())

    class Parent:
        def _compile(self, *args, **kwargs):
            return self.compiled

    namespace = {'qfragment': SimpleNamespace(Manager=Parent, _eligible=lambda k: k.get('eligible', True))}
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), 'exec'), namespace)
    compiled = SimpleNamespace(hash='verified')

    class Launch:
        def __init__(self, result, error=None):
            self.result, self.error, self.calls = result, error, []

        def __getitem__(self, grid):
            def invoke(*args, **kwargs):
                self.calls.append((grid, args, kwargs))
                if self.error:
                    raise self.error
                return self.result
            return invoke

    capturing = [False]

    def setup(error=None):
        original, candidate = Launch('original'), Launch(compiled, error)
        query = SimpleNamespace(shape=[1985], device=SimpleNamespace(index=0))
        manager = SimpleNamespace(
            async_mode=SimpleNamespace(get=lambda: None),
            raw=SimpleNamespace(pre_run_hooks=[]), candidate=SimpleNamespace(pre_run_hooks=[]),
            device=None, wrapper=candidate, optimized={'verified': {}},
            attention=SimpleNamespace(torch=SimpleNamespace(cuda=SimpleNamespace(
                is_current_stream_capturing=lambda: capturing[0]))),
        )
        return namespace['_Policy'](original, manager), original, candidate, {
            'query_ptr': query, 'num_seqs': 1, 'BLOCK_M': 128, 'BLOCK_Q': 16,
            'num_warps': 8, 'num_stages': 1, 'TILE_SIZE': 64,
        }

    checks = []
    with contextlib.redirect_stdout(io.StringIO()):
        policy, old, new, kwargs = setup()
        assert policy[(125, 2)](**kwargs) is compiled
        assert new.calls[0][0] == (63, 2) and new.calls[0][2]['BLOCK_M'] == 256
        assert new.calls[0][2]['num_warps'] == 16
        assert kwargs['BLOCK_M'] == 128 and kwargs['num_warps'] == 8 and not old.calls
        checks.append('selected grid and constants; original arguments unchanged')
        for change in ({'num_seqs': 2}, {'eligible': False}, {'ir_override': 'foreign'}):
            policy, old, new, kwargs = setup();kwargs.update(change)
            assert policy[(125, 2)](**kwargs) == 'original' and not new.calls
        checks.append('unsupported scope and foreign compiler options use original')
        policy, old, new, kwargs = setup();kwargs['query_ptr'].shape = [1023]
        assert policy[(64, 2)](**kwargs) == 'original' and not new.calls
        checks.append('small query uses original')
        capturing[0] = True;policy, old, new, kwargs = setup()
        assert policy[(125, 2)](**kwargs) == 'original' and not new.calls
        capturing[0] = False;checks.append('active capture uses original')
        policy, old, new, kwargs = setup(namespace['_UnavailableBeforeLaunch']('proof failed'))
        assert policy[(125, 2)](**kwargs) == 'original' and policy.disabled
        assert policy[(125, 2)](**kwargs) == 'original' and len(new.calls) == 1
        checks.append('unproved cold compile falls back before launch and disables candidate')
        policy, old, new, kwargs = setup(RuntimeError('device launch error'))
        try:
            policy[(125, 2)](**kwargs)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Device error was hidden')
        assert not old.calls
        checks.append('device failure propagates without retrying original')
        manager = namespace['_VerifiedManager']();manager.compiled = compiled;manager.optimized = {}
        try:
            manager._compile()
        except namespace['_UnavailableBeforeLaunch']:
            pass
        else:
            raise AssertionError('Unproved compiled kernel was accepted')
        manager.optimized = {'verified': {}}
        assert manager._compile() is compiled
        checks.append('compiler returns only a proved kernel')
    print(json.dumps({'passed': True, 'checks': checks,
                      'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                      'limits': 'Controller tests with fake descriptors/launchers; no device numerical or concurrency claim.'}, indent=2))


if __name__ == '__main__':
    main()
