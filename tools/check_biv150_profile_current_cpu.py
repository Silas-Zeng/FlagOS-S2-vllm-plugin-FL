"""CPU contracts: private caches, immutable tuning seed and full warm before trace."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import run_biv150_baseline_profile as profile


def main():
    markers=('BIV150_NATIVE_W8A8_INSTALL','BIV150_STABLE_SMALL_O_INSTALL','BIV150_Q_FRAGMENT_INSTALL','BIV150_PREFILL_NORMALIZATION_INSTALL','BIV150_GQA_DECODE_INSTALL','BIV150_MIXED_DECODE_INSTALLED')
    logs='\n'.join(marker+" {'installed': True}" for marker in markers)
    assert len(profile.installed_runtime(logs))==6
    try:profile.installed_runtime(logs.replace("{'installed': True}","{'installed': False}",1))
    except RuntimeError:pass
    else:raise AssertionError('Inactive baseline silently accepted')
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary).resolve();out=root/'out';out.mkdir();seed=root/'seed';seed.mkdir()
        (seed/'fixture.db').write_bytes(b'fixed CPU fixture')
        record=dict(name='fixture.db',sha256=profile.sha256(seed/'fixture.db'))
        (seed/'manifest.json').write_text(json.dumps({'databases':[record]}))
        args=SimpleNamespace(out=out,source=Path(__file__).resolve().parents[1],library=root/'library.so',tuning_snapshot=seed,workload='4096')
        copied=profile.prepare_tuning(args)
        assert copied['databases']==[record] and (out/'flaggems-cache/config_cache/fixture.db').read_bytes()==b'fixed CPU fixture'
        with patch.dict(os.environ,{'TRITON_CACHE_DIR':'/forbidden-global','FLAGGEMS_CACHE_DIR':'/forbidden-global','VLLM_FL_BIV150_SMALL_OUT_LOWERING':'1'}):
            service,bench=profile.make_environment(args,root/'bootstrap')
        for env in [service,bench]:
            assert env['TRITON_CACHE_DIR']==str(out/'triton-cache')
            assert env['FLAGGEMS_CACHE_DIR']==str(out/'flaggems-cache')
            assert env['VLLM_FL_BIV150_NATIVE_W8A8']==env['VLLM_FL_BIV150_STABLE_SMALL_O']=='1'
            assert 'VLLM_FL_BIV150_SMALL_OUT_LOWERING' not in env
        original=subprocess.check_output(['git','-C',str(args.source),'show',profile.MAIN+':tools/run_biv150_baseline_profile.py'],text=True)
        current=Path(profile.__file__).read_text()
        def cleanup(code):
            tree=ast.parse(code)
            return ast.dump(next(n for cls in tree.body if isinstance(cls,ast.ClassDef) and cls.name=='Driver' for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='cleanup'),include_attributes=False)
        assert cleanup(original)==cleanup(current)
        events=[];driver=profile.Driver(args);driver.update=lambda phase,**kw:events.append((phase,kw))
        driver.stop_profile=lambda:None
        def fake_benchmark(load,**geometry):
            events.append(('benchmark',dict(geometry,profile=driver.current_benchmark['profile'])))
            return dict(completed=geometry['count'],num_prompts=geometry['count'],failed=0,max_concurrency=64,total_input_tokens=geometry['input_len']*geometry['count'],total_output_tokens=geometry['output_len']*geometry['count'])
        with patch.object(profile,'original_benchmark',return_value=fake_benchmark),patch.object(profile,'validate_profile_windows',return_value={'passed':True,'windows':[],'errors':[]}):
            driver.workloads()
        cases=[x[1] for x in events if x[0]=='benchmark']
        assert len(cases)==5 and [x['profile'] for x in cases]==[False,False,False,False,True]
        assert cases[-2]['input_len']==cases[-1]['input_len']==4096
        assert cases[-2]['output_len']==cases[-1]['output_len']==1024
        assert cases[-2]['count']==cases[-1]['count']==256 and cases[-2]['seed']==cases[-1]['seed']==0
        assert 'torch' not in sys.modules
    print(json.dumps(dict(passed=True,production_baseline=profile.MAIN,private_caches=True,full_workload_warmup_before_profile=True,owned_cleanup_ast_unchanged=True,runtime_installation_required=True,torch_imported=False)))


if __name__=='__main__':main()
