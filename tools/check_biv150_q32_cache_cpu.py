"""Check effective default/disabled policy cache identities without Torch."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    root = Path(__file__).resolve().parents[1]
    path = root/'vllm_fl/compile_semantics.py'
    spec = importlib.util.spec_from_file_location('_q32_cache_cpu', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = SimpleNamespace(model_config=SimpleNamespace(
        hf_config=SimpleNamespace(**module._GEOMETRY)))
    def fingerprint(**extra):
        return module.build_fingerprint(config, 'iluvatar', package_root=root/'vllm_fl',
            environ={'VLLM_FL_BIV150_NATIVE_W8A8':'0', **extra}, abi={'cpu_contract':1})
    flag = 'VLLM_FL_BIV150_PREFILL_Q32'
    assert fingerprint() == fingerprint(**{flag:'1'})
    assert fingerprint() != fingerprint(**{flag:'0'})
    for parent in ('VLLM_FL_BIV150_Q_FRAGMENT','VLLM_FL_BIV150_PREFILL_NORMALIZE'):
        assert fingerprint(**{parent:'0',flag:'1'}) == fingerprint(**{parent:'0',flag:'0'})
    print(json.dumps({'passed':True,'gpu_executed':False,
        'default_matches_explicit_enable':True,'explicit_disable_separates_key':True,
        'inactive_parent_ignores_child_flag':True}))


if __name__ == '__main__':
    main()
