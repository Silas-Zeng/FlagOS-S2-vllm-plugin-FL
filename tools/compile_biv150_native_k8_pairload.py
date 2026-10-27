#!/usr/bin/env python3
"""Stdlib CPU plan; explicit --compile-only builds an isolated K-pairload SO, 0 launches."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import biv150_native_k8_pairload_factory as f
from compile_biv150_native_k8_qk import CUDA_FLAGS


def plan():
    return dict(CPU_only=True,source_proof=f.verify(),library=f.LIBRARY,operator=f.NAMESPACE+'::produce',
      parent=f.PARENT,profiles=[[64,2],[32,4]],Q_pitches=[2048,2560],page_bytes=16384,
      comparison='original 377/22e K-then-V, not V-first',explicit_cuda_flags=CUDA_FLAGS,
      unchanged_source='RN32/RNE, peak/shuffle/pack order, K/V issue/waits, QK/PV/C chain, typed ABI, S2/S4',
      actual_global_K_loads=None,actual_VGPR=None,actual_SGPR=None,actual_spills=None,
      actual_occupancy_gain_verified=False,actual_issue_order_verified=False,
      previous22e_qualifications_inherited=False,device_compile=False,loaded_resources_verified=False,
      numeric_verified=False,performance_verified=False,precision_gate=False,model_verified=False,
      attention_launches=0,codec_launches=0,encoder_launches=0,reducer_launches=0,input_initializations=0,
      next_gate='inspect actual producer IR/ASM K paired-load/scatter/liveness and resources before own numeric/ABBA',
      future_numeric='reuse 9695/3eb exact FP64/BF16-P reference, guards, Graph/side, narrowed width16 and ABBA; require this variant own receipts')


def compile_only(a):
    out=a.out.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('New empty output directory required')
    result=plan();out.mkdir(parents=True,exist_ok=True)
    (out/'cpu-plan.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    sources=f.build()
    for name,source in sources.items():(out/name).write_text(source)
    result.update(compile_requested=True,compile_passed=False,completed=False,errors=[],commands=[])
    try:
        os.environ.setdefault('MAX_JOBS','1');os.environ.setdefault('TORCH_CUDA_ARCH_LIST','7.1')
        import torch
        from torch.utils.cpp_extension import load
        result['torch']=dict(version=torch.__version__,file=torch.__file__,cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),cuda_initialized_before=torch.cuda.is_initialized())
        if torch.__version__!='2.10.0' or bool(torch._C._GLIBCXX_USE_CXX11_ABI):raise ValueError('Frozen 377 Torch/host ABI changed')
        compiler=Path(a.compiler).resolve()
        if not compiler.is_file():raise ValueError('Explicit installed CoreX compiler missing')
        version=subprocess.run([str(compiler),'--version'],capture_output=True,text=True,check=False)
        (out/'compiler-version.txt').write_text(version.stdout+version.stderr)
        result['compiler']=dict(path=str(compiler),sha256=f.sha(compiler.read_bytes()),version_exit=version.returncode)
        for name in ('biv150_native_k8_pairload_qk.cu','biv150_native_k8_pairload_codec.cu'):
            source=f.ROOT/'tools/kernels'/name
            for suffix,flags in (('ll',['-S','-emit-llvm']),('s',['-S'])):
                target=out/(source.stem+'.'+suffix)
                cmd=[str(compiler),*CUDA_FLAGS,'--cuda-device-only',*flags,str(source),'-o',str(target)]
                p=subprocess.run(cmd,capture_output=True,text=True,check=False)
                (out/(target.name+'.stdout.txt')).write_text(p.stdout)
                (out/(target.name+'.stderr.txt')).write_text(p.stderr)
                result['commands'].append(dict(argv=cmd,exit_code=p.returncode))
                if p.returncode or not target.is_file() or not target.stat().st_size:raise RuntimeError('Compiler failure: '+name+' '+suffix)
        build=out/'build';build.mkdir()
        paths=[str(f.ROOT/'tools/kernels'/name) for name in sources if name.endswith(('.cpp','.cu'))]
        library=load(name=f.NAMESPACE,sources=paths,build_directory=str(build),extra_cflags=['-O3','-std=c++17'],
          extra_cuda_cflags=CUDA_FLAGS,is_python_module=False,verbose=True)
        result['library']=dict(path=str(library),sha256=f.sha(Path(library).read_bytes()))
        result['torch']['cuda_initialized_after']=torch.cuda.is_initialized()
        result.update(compile_passed=True,completed=True,device_compile=True)
    except Exception as exc:
        result['errors'].append(repr(exc));(out/'failure.txt').write_text(traceback.format_exc())
    result['artifact_manifest']=[dict(relative=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=f.sha(p.read_bytes()))
      for p in sorted(out.rglob('*')) if p.is_file() and p.name!='compile-summary.json' and '__pycache__' not in p.parts]
    (out/'compile-summary.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps(result,sort_keys=True,indent=2));return 0 if result['compile_passed'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--compile-only',action='store_true');p.add_argument('--out',type=Path)
    p.add_argument('--compiler',default='/usr/local/corex/bin/clang++');a=p.parse_args()
    if not a.compile_only:
        if a.out:raise ValueError('--out requires explicit --compile-only')
        assert not {'torch','triton','numpy','vllm'}.intersection(sys.modules)
        print(json.dumps(plan(),sort_keys=True,indent=2));return 0
    if a.out is None:raise ValueError('Explicit --compile-only requires --out')
    return compile_only(a)


if __name__=='__main__':raise SystemExit(main())
