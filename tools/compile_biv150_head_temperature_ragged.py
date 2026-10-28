"""Default CPU plan; explicit Root compile-only may initialize CUDA/allocate, never launch head."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from bench_biv150_head_temperature_ragged import ROOT, ROWS, plan, preflight, source_check


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--compile-only',action='store_true');p.add_argument('--root-owned-exclusive',action='store_true');p.add_argument('--commit');p.add_argument('--out',type=Path);p.add_argument('--row',type=int,choices=ROWS,action='append');a=p.parse_args()
    specification=dict(plan(),mode='compile-only',head_attention_writer_encoder_launches=0,input_initializations=0,
                       CUDA_context_and_allocations='required only in explicit compile mode',
                       resources='prelaunch metadata only, not loaded registers/occupancy')
    if not a.compile_only:
        if any((a.root_owned_exclusive,a.commit,a.out,a.row)):p.error('compile arguments require --compile-only')
        print(json.dumps(specification,ensure_ascii=False,sort_keys=True,indent=2));return
    if not all((a.root_owned_exclusive,a.commit,a.out)):p.error('exact commit/new out/Root exclusive acknowledgement required')
    source=source_check(a.commit);admission=preflight(a);a.out.mkdir(parents=True,exist_ok=False)
    result=dict(specification,completed=False,errors=[],source_commit=a.commit,source_sha256=source,admission=admission,records=[])
    def save():(a.out/'compile-summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    save()
    try:
        os.environ.update(TRITON_CACHE_DIR=str(a.out/'triton-cache'),PYTHONDONTWRITEBYTECODE='1',VLLM_FL_BIV150_NATIVE_W8A8='0',VLLM_FL_BIV150_NATIVE_GQA='0',VLLM_FL_BIV150_HEAD_TEMPERATURE='1',VLLM_FL_BIV150_HEAD_TEMPERATURE_RAGGED='1')
        sys.path.insert(0,str(ROOT))
        import torch,triton
        import vllm_fl
        from vllm_fl.patches import biv150_lm_head_temperature_kernel as candidate
        jit_module=importlib.import_module('triton.runtime.jit')
        assert hashlib.sha256(Path(jit_module.__file__).read_bytes()).hexdigest()=='66a168ae959e719186771e66815e807f1e449e66c5283ef596ec3eb0e35cb775','actual JIT source changed'
        assert Path(vllm_fl.__file__).resolve()==ROOT/'vllm_fl/__init__.py','package import shadowed'
        assert Path(candidate.__file__).resolve()==ROOT/'vllm_fl/patches/biv150_lm_head_temperature_kernel.py','candidate import shadowed'
        head_temperature_kernel=candidate.head_temperature_kernel
        assert head_temperature_kernel.fn.__globals__ is vars(candidate)
        result['actual_imports']=dict(torch_file=str(torch.__file__),torch_version=str(torch.__version__),package_file=str(vllm_fl.__file__),candidate_file=str(candidate.__file__))
        assert 'BI-V150' in torch.cuda.get_device_properties(0).name
        assert torch.cuda.get_device_properties(0).multi_processor_count==16
        assert triton.runtime.driver.active.get_current_target().warp_size==64
        result['CUDA_context_initialized']=bool(torch.cuda.is_initialized())
        w=torch.empty((130560,2048),dtype=torch.bfloat16,device='cuda')
        result.update(allocations=True,gpu_executed=True,CUDA_context_initialized=bool(torch.cuda.is_initialized()))
        save()
        for rows in a.row or ROWS:
            x=torch.empty((rows,2048),dtype=torch.bfloat16,device='cuda')
            out=torch.empty((rows,130560),dtype=torch.float32,device='cuda')
            temperature=torch.empty((rows,),dtype=torch.float32,device='cuda')
            c=specification['rows'][str(rows)];options={k:v for k,v in c.items() if k!='grid'}
            kernel=head_temperature_kernel.warmup(x,w,w,out,rows,130560,2048,2048,1,2048,1,130560,1,0,temperature,
                BIAS=False,GROUP_M=8,SPLIT_K=1,EVEN_K=True,num_ctas=1,grid=tuple(c['grid']),**options)
            assert {'ttir','ttgir','llir','asm','cubin'}<=set(kernel.asm),'fresh actual IR/ASM/cubin incomplete'
            for kind,value in kernel.asm.items():
                if isinstance(value,(str,bytes)):(a.out/('rows'+str(rows)+'.'+kind)).write_bytes(value.encode() if isinstance(value,str) else value)
            result['records'].append(dict(rows=rows,grid=c['grid'],hash=getattr(kernel.metadata,'hash',None),
                metadata=str(kernel.metadata),signature=str(getattr(kernel.src,'signature',None)),
                constants=str(getattr(kernel.src,'constants',None)),n_regs=getattr(kernel,'n_regs',None),
                n_spills=getattr(kernel,'n_spills',None),resource_phase='prelaunch',numerical_qualified=False))
            save()
        assert source_check(a.commit)==source
        result.update(completed=True,gpu_executed=True,CUDA_context_initialized=True,allocations=True)
    except BaseException as e:result['errors'].append(repr(e));save();raise
    finally:save()

if __name__=='__main__':main()
