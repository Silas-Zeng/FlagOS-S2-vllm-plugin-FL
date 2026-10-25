"""Prove shipping-default edits preserve the GPU-validated explicit-on behavior."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
VALIDATED='8c0ca8cdaa6403d58ddf066dd1854fcd0e9def38'


def old(path):
    return subprocess.check_output(['git','-C',str(ROOT),'show',VALIDATED+':'+path],text=True)


def main():
    operator='vllm_fl/patches/biv150_stable_small_o.py'
    semantics='vllm_fl/compile_semantics.py'
    runner='vllm_fl/worker/model_runner.py'
    order='vllm_fl/patches/biv150_compute_order.py'
    assert (ROOT/operator).read_text()==old(operator).replace(
        "os.environ.get('VLLM_FL_BIV150_STABLE_SMALL_O','0')",
        "os.environ.get('VLLM_FL_BIV150_STABLE_SMALL_O','1')")
    assert (ROOT/semantics).read_text()==old(semantics).replace(
        'env.get(key, "0" if key == "VLLM_FL_BIV150_STABLE_SMALL_O" else "1")','env.get(key, "1")')
    assert (ROOT/runner).read_text()==old(runner).replace(
        '# Independent BF16 O-projection stability experiment; default disabled.',
        '# Audited BF16 small-row MiniCPM projections.')
    previous_sha=hashlib.sha256(old(runner).encode()).hexdigest()
    current_sha=hashlib.sha256((ROOT/runner).read_bytes()).hexdigest()
    assert (ROOT/order).read_text()==old(order).replace(previous_sha,current_sha)
    assert 'if getattr(current_platform, "vendor_name", None) == "iluvatar":\n    from vllm_fl.patches.biv150_stable_small_o' in (ROOT/runner).read_text()
    changed=set(subprocess.check_output(['git','-C',str(ROOT),'diff','--name-only',VALIDATED,'--','vllm_fl'],text=True).splitlines())
    assert changed=={operator,semantics,runner,order},changed
    spec=importlib.util.spec_from_file_location('isolated_semantics',ROOT/semantics)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(**module._GEOMETRY)))
    with tempfile.TemporaryDirectory(prefix='small-bf16-cache-contract-') as folder:
        library=Path(folder)/'do-not-load.so';library.write_bytes(b'CPU fingerprint-only fixture; never load')
        env={'BIV150_NATIVE_W8A8_LIBRARY':str(library)}
        def fingerprint(extra):
            return module.build_fingerprint(config,'iluvatar',package_root=ROOT/'vllm_fl',environ={**env,**extra},abi={'torch':'cpu-contract','cxx11_abi':False})
        assert fingerprint({})==fingerprint({'VLLM_FL_BIV150_STABLE_SMALL_O':'1'})
        assert fingerprint({})['sha256']!=fingerprint({'VLLM_FL_BIV150_STABLE_SMALL_O':'0'})['sha256']
        assert module.build_fingerprint(config,'metax',package_root=ROOT/'vllm_fl',environ=env,abi={}) is None
    report={'passed':True,'gpu_executed':False,'validated_runtime':VALIDATED,
            'change':'default-on matches validated explicit-on; no kernel/linear/model math change',
            'runner_sha256':current_sha,'strict_runner_pin_updated':True,
            'default_equals_explicit_on':True,'explicit_off_separate':True,
            'iluvatar_worker_scope_preserved':True,
            'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sorted(changed)},
            'remaining':'fresh default-unset model smoke and matched-control acceptance before merge'}
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
