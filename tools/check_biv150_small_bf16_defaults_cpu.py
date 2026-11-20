"""Check default BF16 small-row dispatch and its device-scoped cache identity."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'vllm_fl'
FLAG = 'VLLM_FL_BIV150_STABLE_SMALL_O'


def main():
    spec = importlib.util.spec_from_file_location('isolated_semantics', PACKAGE / 'compile_semantics.py')
    semantics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(semantics)
    model = SimpleNamespace(hf_config=SimpleNamespace(**semantics._GEOMETRY))
    config = SimpleNamespace(model_config=model)

    def fingerprint(vendor, env):
        return semantics.build_fingerprint(config, vendor, package_root=PACKAGE, environ=env, abi={})

    default = fingerprint('iluvatar', {})
    assert default['schema'] == 2
    assert default == fingerprint('iluvatar', {FLAG: '1'})
    assert default != fingerprint('iluvatar', {FLAG: '0'})
    assert fingerprint('metax', {}) is None
    assert fingerprint('cuda', {}) is None
    runner = ast.parse((PACKAGE / 'worker/model_runner.py').read_bytes())
    installs = [node for node in runner.body if isinstance(node, ast.If)
                and 'vllm_fl.patches.biv150_stable_small_o' in ast.unparse(node)]
    assert len(installs) == 1
    assert ast.unparse(installs[0].test) == "getattr(current_platform, 'vendor_name', None) == 'iluvatar'"
    operator = ast.parse((PACKAGE / 'patches/biv150_stable_small_o.py').read_bytes())
    kernel = next(fn for fn in operator.body if isinstance(fn, ast.FunctionDef) and fn.name == '_small_o_kernel')
    assert any(isinstance(dec, ast.Call) and ast.unparse(dec.func) == 'triton.jit'
               for dec in kernel.decorator_list)
    install = next(fn for fn in operator.body if isinstance(fn, ast.FunctionDef) and fn.name == 'install')
    assert f"os.environ.get('{FLAG}', '1')" in ast.unparse(install)
    print(json.dumps({'passed': True, 'default_equals_explicit_on': True,
                      'explicit_off_has_separate_cache': True, 'iluvatar_only': True}, indent=2))


if __name__ == '__main__':
    main()
