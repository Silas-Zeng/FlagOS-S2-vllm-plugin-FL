"""Effective default/off cache semantics and source identity; no Torch/SO load."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace as NS


def main():
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('_multi2_cache',root/'vllm_fl/compile_semantics.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=NS(model_config=NS(hf_config=NS(**module._GEOMETRY)))
    flag='VLLM_FL_BIV150_PREFILL_MULTI2'
    def key(package=None,**extra):
        return module.build_fingerprint(config,'iluvatar',package_root=package or root/'vllm_fl',
            environ={'VLLM_FL_BIV150_NATIVE_W8A8':'0',**extra},abi={'cpu_contract':1})
    assert key()==key(**{flag:'1'}) and key()!=key(**{flag:'0'})
    for parent in ('VLLM_FL_BIV150_PREFILL_Q32','VLLM_FL_BIV150_Q_FRAGMENT','VLLM_FL_BIV150_PREFILL_NORMALIZE'):
        assert key(**{parent:'0',flag:'1'})==key(**{parent:'0',flag:'0'})
    assert key()==key(TRITON_CACHE_DIR='/private/path/not/semantic',BIV150_PROFILE_RECORD='1')
    with tempfile.TemporaryDirectory(prefix='multi2-cache-source-') as temporary:
        package=Path(temporary)/'vllm_fl';shutil.copytree(root/'vllm_fl',package,ignore=shutil.ignore_patterns('__pycache__'))
        assert key(package)==key()
        path=package/'patches/biv150_prefill_multi2.py'
        path.write_text(path.read_text()+'\n# changed source identity fixture\n')
        assert key(package)!=key()
    assert module.build_fingerprint(config,'metax',package_root=root/'vllm_fl',environ={},abi={}) is None
    result=dict(passed=True,gpu_executed=False,precision_gate=False,
        default_equals_explicit_on=True,off_separates_official_compile_key=True,
        inactive_parent_ignores_child_flag=True,new_source_content_in_cache_key=True,
        unrelated_cache_paths_and_profiler_flags_ignored=True,other_vendor_untouched=True)
    path=root/'docs/evidence/prefill-multi2-20261027/service-cache-cpu.json'
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
