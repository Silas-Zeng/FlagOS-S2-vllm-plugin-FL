#!/usr/bin/env python3
"""Default CPU plan; explicit isolated compact12544 NativeK8 V-overlap compile-only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import biv150_native_k8_compact_n48_factory as factory

SOURCES = [factory.path(role) for role in ('qk.cpp', 'qk.cu', 'codec.cu')]
HEADER = factory.path('codec.cuh')
CUDA_FLAGS = ['-O3', '-std=c++17', '-D__ILUVATAR__', '-D__MR__',
              '-DCOREX_NATIVE_GQA_V_OVERLAP=1', '-x', 'ivcore',
              '--cuda-gpu-arch=ivcore11', '--cuda-path=/usr/local/corex']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_plan():
    return dict(source_transform=factory.verify(), explicit_cuda_flags=CUDA_FLAGS,
                private_source={str(p.relative_to(factory.ROOT)): sha(p) for p in SOURCES+[HEADER]},
                requested='deferred-V-wait compile-only; no device qualification',
                attention_launches=0, encoder_launches=0, codec_launches=0,
                reducer_launches=0, input_initializations=0,
                numeric_verified=False, performance_verified=False, precision_gate=False,
                actual_SO_instruction_order_verified=False, readonly_CPU_default=True)


def compile_only(args,result):
    out=args.out.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    (out/'cpu-plan.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    for source in SOURCES+[HEADER]:(out/source.name).write_bytes(source.read_bytes())
    result=dict(result,compile_requested=True,compile_passed=False,completed=False,errors=[],commands=[])
    try:
        # Explicit device mode only. Same flags/library loading mechanism as b3;
        # loading registers definitions but this frontend never calls an op.
        os.environ.setdefault('MAX_JOBS','1');os.environ.setdefault('TORCH_CUDA_ARCH_LIST','7.1')
        import torch
        from torch.utils.cpp_extension import load
        result['torch']=dict(version=torch.__version__,file=torch.__file__,cuda_initialized_before=torch.cuda.is_initialized())
        compiler=Path(args.compiler).resolve()
        if not compiler.is_file():raise ValueError('Explicit CoreX compiler path missing: '+str(compiler))
        version=subprocess.run([str(compiler),'--version'],capture_output=True,text=True,check=False)
        (out/'compiler-version.txt').write_text(version.stdout+version.stderr)
        result['compiler']=dict(path=str(compiler),sha256=sha(compiler),version_exit=version.returncode)
        # Preserve exact commands and failed outputs. A successful compile is not
        # RN32/native-ISA proof: actual LLVM/ASM still needs independent review.
        for source in SOURCES[1:]:
            for suffix,flags in (('ll',['-S','-emit-llvm']),('s',['-S'])):
                target=out/(source.stem+'.'+suffix)
                command=[str(compiler),*CUDA_FLAGS,'--cuda-device-only',*flags,str(source),'-o',str(target)]
                proc=subprocess.run(command,capture_output=True,text=True,check=False)
                (out/(source.stem+'.'+suffix+'.stdout.txt')).write_text(proc.stdout)
                (out/(source.stem+'.'+suffix+'.stderr.txt')).write_text(proc.stderr)
                result['commands'].append(dict(argv=command,exit_code=proc.returncode))
                if proc.returncode!=0:raise RuntimeError('Target compiler failed: '+source.name+' -> '+suffix)
                if not target.is_file() or not target.stat().st_size:raise ValueError('Missing actual target artifact')
        build=out/'build';build.mkdir()
        library=load(name='cjj_native_k8_compact_n48_s4_probe',sources=list(map(str,SOURCES)),build_directory=str(build),
          extra_cflags=['-O3','-std=c++17'],extra_cuda_cflags=CUDA_FLAGS,is_python_module=False,verbose=True)
        result['library']=dict(path=str(library),sha256=sha(library))
        result['torch']['cuda_initialized_after']=torch.cuda.is_initialized()
        result['compile_passed']=True;result['device_compile']=True;result['completed']=True
    except Exception as exc:
        result['errors'].append(repr(exc));(out/'failure.txt').write_text(traceback.format_exc())
    result['artifact_manifest']=[dict(relative=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p))
      for p in sorted(out.rglob('*')) if p.is_file() and p.name!='compile-summary.json' and '__pycache__' not in p.parts]
    (out/'compile-summary.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps(result,sort_keys=True,indent=2))
    return 0 if result['compile_passed'] else 1

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compile-only', action='store_true')
    p.add_argument('--root-owned-exclusive', action='store_true')
    p.add_argument('--out', type=Path)
    p.add_argument('--compiler', default='/usr/local/corex/bin/clang++')
    a=p.parse_args(); result=cpu_plan()
    if not a.compile_only:
        if a.out or a.root_owned_exclusive:p.error('Compile arguments require --compile-only')
        assert not {'torch', 'triton', 'numpy'}.intersection(sys.modules)
        print(json.dumps(result, sort_keys=True, indent=2)); return 0
    if not a.root_owned_exclusive:p.error('Root must confirm prior owner terminal and exclusive device')
    if a.out is None or not a.out.is_absolute() or a.out.exists():
        p.error('Use a fresh absolute output directory')
    if a.out.resolve().is_relative_to(factory.ROOT):p.error('Output must be outside source checkout')
    return compile_only(a,result)

if __name__=='__main__':raise SystemExit(main())
