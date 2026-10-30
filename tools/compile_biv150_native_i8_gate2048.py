#!/usr/bin/env python3
"""Default CPU plan; explicit Root compile-only builds/exports, never calls an op."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from check_biv150_native_i8_gate2048_cpu import BASE,ROOT,check

SOURCES=[ROOT/'tools/kernels'/('biv150_native_i8_gate2048.'+ext) for ext in ('cpp','cu')]
FLAGS=['-O3','-std=c++17','-D__ILUVATAR__','-D__MR__','-x','ivcore',
       '--cuda-gpu-arch=ivcore11','--cuda-path=/usr/local/corex']
SDK_PINS={'iluvatar_mma.h':'36a0b4f3e5d518780f4549f7a42c9938c982b493c26f0696daa22091c635e17d',
          'iluvatar_mma.hpp':'ddb5cd4a1d53aa2cacad501a1b2d138715c045bbf837a162d6c20a144ca45094'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')


def compile_only(args,plan):
    if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain']).strip():
        raise ValueError('Compile requires a clean frozen checkout')
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD']).decode().strip()
    subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',BASE,commit],check=True)
    out=args.out.resolve();out.mkdir(parents=True)
    result=dict(plan,source_commit=commit,source_root=str(ROOT),compile_passed=False,
                completed=False,commands=[],errors=[])
    write(out/'cpu-plan.json',plan)
    try:
        # Bind the exact header definitions used for the CPU ownership proof.
        headers=Path('/usr/local/corex/include/IX/crt-ixsdk')
        result['SDK_headers']={name:sha(headers/name) for name in SDK_PINS}
        if result['SDK_headers']!=SDK_PINS:
            raise ValueError('Installed SDK header identity differs; review before compiling')
        compiler=Path(args.compiler).resolve();objdump=Path(args.objdump).resolve()
        result['compiler']=dict(path=str(compiler),sha256=sha(compiler))
        result['objdump']=dict(path=str(objdump),sha256=sha(objdump))
        def command(argv,stem):
            p=subprocess.run(argv,capture_output=True,text=True)
            (out/(stem+'.stdout.txt')).write_text(p.stdout)
            (out/(stem+'.stderr.txt')).write_text(p.stderr)
            result['commands'].append(dict(argv=argv,exit_code=p.returncode))
            if p.returncode:raise RuntimeError('Command failed: '+stem)
            return p.stdout
        command([str(compiler),'--version'],'compiler-version')
        for source in SOURCES:(out/source.name).write_bytes(source.read_bytes())
        for suffix,extra in (('ll',['-S','-emit-llvm']),('s',['-S'])):
            target=out/('kernel.'+suffix)
            command([str(compiler),*FLAGS,'--cuda-device-only',*extra,str(SOURCES[1]),'-o',str(target)],'standalone-'+suffix)
            if not target.is_file() or not target.stat().st_size:raise RuntimeError('Missing standalone artifact')
        os.environ.setdefault('MAX_JOBS','1');os.environ.setdefault('TORCH_CUDA_ARCH_LIST','7.1')
        import torch
        from torch.utils.cpp_extension import load
        result['torch']=dict(version=torch.__version__,file=torch.__file__,cuda_initialized_before=torch.cuda.is_initialized())
        if result['torch']['cuda_initialized_before']:raise RuntimeError('Preexisting CUDA context')
        build=out/'build';build.mkdir()
        library=load(name='cjj_native_i8_gate2048_probe',sources=list(map(str,SOURCES)),
            build_directory=str(build),extra_cflags=['-O3','-std=c++17'],
            extra_cuda_cflags=FLAGS,is_python_module=False,verbose=True)
        result['library']=dict(path=str(library),sha256=sha(library))
        for mode in ('IR','sass'):
            if not command([str(objdump),'--'+mode,str(library)],'actual-SO-'+mode).strip():
                raise RuntimeError('Empty final SO export')
        result['torch']['cuda_initialized_after']=torch.cuda.is_initialized()
        if result['torch']['cuda_initialized_after']:raise RuntimeError('Unexpected CUDA initialization')
        result.update(compile_passed=True,completed=True)
    except Exception as e:
        result['errors'].append(repr(e));(out/'failure.txt').write_text(traceback.format_exc())
    result['artifacts']=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(out.rglob('*')) if p.is_file()]
    write(out/'compile-summary.json',result)
    print(json.dumps(result,sort_keys=True,indent=2))
    return 0 if result['compile_passed'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compile-only',action='store_true')
    p.add_argument('--root-owned-exclusive',action='store_true')
    p.add_argument('--out',type=Path)
    p.add_argument('--compiler',default='/usr/local/corex/bin/clang++')
    p.add_argument('--objdump',default='/usr/local/corex/bin/ixobjdump')
    a=p.parse_args()
    plan=dict(cpu_check=check(),sources={str(x.relative_to(ROOT)):sha(x) for x in SOURCES},
        CUDA_flags=FLAGS,SDK_header_pins=SDK_PINS,default_enabled=False,
        service_installed=False,quantizer_modified=False,numeric_verified=False,
        performance_verified=False,precision_gate=False)
    if not a.compile_only:
        if a.out or a.root_owned_exclusive:p.error('Compile options need --compile-only')
        print(json.dumps(plan,sort_keys=True,indent=2));return 0
    if not a.root_owned_exclusive:p.error('Root must own the compile schedule')
    if a.out is None or not a.out.is_absolute() or a.out.exists():p.error('Use a fresh absolute output directory')
    if a.out.resolve().is_relative_to(ROOT):p.error('Output must be outside this checkout')
    return compile_only(a,plan)


if __name__=='__main__':
    raise SystemExit(main())
