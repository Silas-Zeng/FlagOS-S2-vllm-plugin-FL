#!/usr/bin/env python3
"""CPU default; explicit no-attention BF16 canary indexing diagnostic only."""
import argparse
import importlib
import json
import os
from pathlib import Path
import struct
import sys
sys.dont_write_bytecode=True
import bench_biv150_konly_2d128_numeric as frozen

SHAPE=(2050,16,128)
TOKEN_BYTES=16*128*2


def bytes_cpu(tensor,torch):
    if tensor.device.type!='cpu':raise ValueError('Full authority conversion must already be CPU')
    return tensor.contiguous().view(-1).view(torch.uint8).numpy().tobytes()


def expected():
    front=struct.pack('<H',0x428e)*(16*128) # BF16 71
    middle=struct.pack('<H',0xc060)*(2048*16*128) # BF16 -3.5
    return front+middle+front


def describe_bytes(data):
    from collections import Counter
    counts=Counter(struct.unpack('<'+'H'*(len(data)//2),data))
    return dict(bytes=len(data),sha256=frozen.actual.sha(data),
                BF16_bits_counts={hex(k):v for k,v in sorted(counts.items())},
                first16_BF16_bits=[hex(x) for x in struct.unpack_from('<16H',data)])


def run(args):
    actual=frozen.actual;parent=frozen.parent
    actual.need(args.commit and args.out and not args.out.exists(),'exact commit and fresh output')
    actual.need(parent.git('rev-parse','HEAD')==args.commit and not parent.git('status','--porcelain'),'clean exact diagnostic freeze')
    actual.need(not parent.git('diff',frozen.descriptors.MAIN,'--','vllm_fl','csrc'),'production differs from DAC')
    actual.need(not {'torch','triton','numpy'}.intersection(sys.modules),'cold admission before framework')
    args.out.mkdir(parents=True)
    result=dict(commit=args.commit,completed=False,errors=[],diagnostic_only=True,shape=SHAPE,
                private_attention_launches=0,private_encoder_launches=0,
                private_attention_or_codec_JIT_executed=False,performance_measured=False,
                precision_gate=False,model_or_service_verified=False,methods={})
    def save():parent.save(args.out/'canary-index-summary.json',result)
    save()
    try:
        result['installed_sources']=parent.installed_sources();result['admission']=parent.idle_before_cuda();save()
        os.environ['TRITON_CACHE_DIR']=str(args.out/'triton-cache')
        import numpy as np
        import torch
        from vllm.triton_utils import triton
        import biv150_prefill_probe_fixtures as fixture
        fixture.torch=torch;torch.set_num_threads(4);torch.set_flush_denormal(False)
        result['modules']={name:dict(path=str(module.__file__),version=getattr(module,'__version__',None))
            for name,module in [('numpy',np),('torch',torch),('triton',triton),('fixture',fixture)]}
        for module_name,pin in [('vllm.v1.attention.ops.triton_unified_attention',parent.factory.PINS['sdk.py']),
                                ('vllm.v1.attention.ops.triton_attention_helpers',parent.factory.PINS['helpers.py'])]:
            module=importlib.import_module(module_name);path=Path(module.__file__)
            actual.need(actual.sha(path.read_bytes())==pin,'loaded SDK/helper source identity')
            result['modules'][module_name]=dict(path=str(path),sha256=pin)
        target=triton.runtime.driver.active.get_current_target()
        actual.need((target.backend,target.arch,target.warp_size)==('corex',71,64),'actual target')
        result['target']=dict(backend=target.backend,arch=target.arch,warp_size=target.warp_size)
        buffer=torch.empty(SHAPE,device='cuda',dtype=torch.bfloat16)
        buffer.fill_(71);buffer[1:2049].fill_(-3.5);torch.cuda.synchronize()
        # One full synchronous CPU copy, with byte slicing as the authority;
        # no GPU advanced/negative indexing participates in expected results.
        full_cpu=buffer.cpu();authority=bytes_cpu(full_cpu,torch)
        (args.out/'full-storage-before.bin').write_bytes(authority)
        actual.need(authority==expected(),'fresh full-storage marker/71 setup differs')
        reference=authority[:TOKEN_BYTES]+authority[-TOKEN_BYTES:]
        # CPU indexing additionally agrees with the raw-byte authoritative slice.
        cpu_pair=torch.stack((full_cpu[0],full_cpu[-1]))
        actual.need(bytes_cpu(cpu_pair,torch)==reference,'CPU indices versus storage authority')
        result['authority']=dict(full=describe_bytes(authority),canaries=describe_bytes(reference),
            expected_setup_exact=True,CPU_indices_match_full_storage_bytes=True,
            view=frozen.descriptors.descriptors({'full':buffer,'interior':buffer[1:2049]}))
        methods=[('GPU_advanced_0_minus1',lambda:buffer[[0,-1]],reference),
                 ('GPU_advanced_0_2049',lambda:buffer[[0,2049]],reference),
                 ('GPU_single_minus1',lambda:buffer[-1],reference[TOKEN_BYTES:]),
                 ('GPU_single_2049',lambda:buffer[2049],reference[TOKEN_BYTES:])]
        for name,read,expected_bits in methods:
            value=read();torch.cuda.synchronize();data=bytes_cpu(value.cpu(),torch)
            (args.out/(name+'.bin')).write_bytes(data)
            after=bytes_cpu(buffer.cpu(),torch)
            (args.out/(name+'-full-after.bin')).write_bytes(after)
            result['methods'][name]=dict(**describe_bytes(data),matches_full_storage_authority=data==expected_bits,
                readonly_before_sha256=actual.sha(authority),readonly_after_sha256=actual.sha(after),
                full_source_unchanged=after==authority,actual_view=frozen.descriptors.descriptors({'read':value})['read'])
            save();actual.need(after==authority,'indexing modified source buffer')
        result['completed']=True
        result['all_indexing_methods_correct']=all(v['matches_full_storage_authority'] for v in result['methods'].values())
        result['negative_advanced_disagrees_with_full_CPU_authority']=not result['methods']['GPU_advanced_0_minus1']['matches_full_storage_authority']
        save();return result
    except Exception as exc:
        result['errors'].append(dict(type=type(exc).__name__,message=str(exc)));save();raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-gpu',action='store_true')
    p.add_argument('--commit');p.add_argument('--out',type=Path);args=p.parse_args()
    if args.run_gpu:result=run(args)
    else:
        if args.commit or args.out:p.error('device flags require explicit --run-gpu')
        b=expected();result=dict(CPU_only=True,shape=SHAPE,total_bytes=len(b),authority_canary=describe_bytes(b[:TOKEN_BYTES]+b[-TOKEN_BYTES:]),
            full_expected_sha256=frozen.actual.sha(b),GPU_executed=False,private_attention_launches=0,
            indexing_methods=['advanced[0,-1]','advanced[0,2049]','single[-1]','single[2049]'],
            actual_indexing_verified=False,precision_gate=False,performance_measured=False)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':main()
