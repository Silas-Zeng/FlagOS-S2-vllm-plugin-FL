#!/usr/bin/env python3
"""Default entry and synthetic typed-ABI rejection contracts, not actual GPU headers."""
import copy
import json
from pathlib import Path
import subprocess
import sys

from biv150_konly_mixed_factory import build, NAMES
from biv150_konly_mixed_ir import typed_abi

ROOT=Path(__file__).resolve().parents[1]
BASE=[('segm_output_ptr','!tt.ptr<f32>'),('segm_max_ptr','!tt.ptr<f32>'),('segm_expsum_ptr','!tt.ptr<f32>'),
      ('query_ptr','!tt.ptr<bf16>'),('query_start_len_ptr','!tt.ptr<i32>')]
REST=[('value_cache_ptr','!tt.ptr<bf16>'),('seq_lens_ptr','!tt.ptr<i32>'),('block_table_ptr','!tt.ptr<i32>'),
      ('softmax_scale_log2e','f32'),('block_table_stride','i64'),('query_stride_token','i64'),('k_stride_block','i64'),
      ('k_stride_slot','i64'),('k_stride_head','i64'),('v_stride_block','i64'),('v_stride_slot','i64')]


def run():
    source,_=build((ROOT/'vllm_fl/patches/biv150_gqa_decode.py').read_text());mutants=0
    for mode in ('bf16','k8'):
        # Literal pointer/type ledger, independent of the parser's expected_abi().
        extra=[] if mode=='bf16' else [('query_code_ptr','!tt.ptr<i8>'),('query_scale_ptr','!tt.ptr<f32>'),('key_scale_ptr','!tt.ptr<f32>')]
        typed=BASE+extra+[('key_cache_ptr','!tt.ptr<bf16>' if mode=='bf16' else '!tt.ptr<i8>')]+REST
        ll=['ptr addrspace(1)' if t.startswith('!tt.ptr') else 'float' if t=='f32' else t for _,t in typed]+['ptr addrspace(1)']*2
        artifacts=dict(ttir='tt.func public @'+NAMES[mode]+'('+', '.join('%'+n+': '+t for n,t in typed)+')',
            llir='define iluvatar_kernel void @'+NAMES[mode]+'('+', '.join(t+' %'+str(i) for i,t in enumerate(ll))+')')
        assert typed_abi(artifacts,source[mode],NAMES[mode],mode)['typed_abi_match']
        for kind,old,new in [('ttir','%query_start_len_ptr: !tt.ptr<i32>','%query_start_len_ptr: !tt.ptr<i64>'),
                             ('ttir','%k_stride_block: i64','%k_stride_block: i32'),
                             ('ttir','%key_cache_ptr:', '%wrong_cache_ptr:'),
                             ('llir','float %','i64 %')]:
            assert old in artifacts[kind];changed=copy.deepcopy(artifacts);changed[kind]=changed[kind].replace(old,new,1)
            try: typed_abi(changed,source[mode],NAMES[mode],mode)
            except ValueError: pass
            else: raise AssertionError('Changed typed ABI accepted')
            mutants+=1
    plan=json.loads(subprocess.check_output([sys.executable,str(ROOT/'tools/bench_biv150_konly_mixed_decode.py')],text=True))
    assert plan['cpu_only'] and plan['compile_only_ready'] and not plan['numeric_ready']
    assert not any(plan[k] for k in ('torch_imported','triton_imported','numpy_imported','gpu_executed','service_installed','timing_prepared'))
    assert plan['cases']['mixed4k']['segments']==2 and plan['cases']['mixed16k']['segments']==4
    assert not {'torch','triton','numpy','vllm'}.intersection(sys.modules)
    return dict(passed=True,cpu_only=True,default_cli_no_framework=True,synthetic_header_mutants=mutants,
                actual_gpu_headers_replayed=False,actual_typed_abi_verified=False,numeric_ready=False,
                gpu_executed=False,service_installed=False)


if __name__=='__main__':print(json.dumps(run(),indent=2))
