#!/usr/bin/env python3
"""Default stdlib plan; Root-only explicit compile, no attention/codec launch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import biv150_native_k8_klookahead_factory as factory

SOURCES = [factory.path(r) for r in ('qk.cpp', 'qk.cu', 'codec.cu')]
HEADER = factory.path('codec.cuh')
CUDA_FLAGS = ['-O3', '-std=c++17', '-D__ILUVATAR__', '-D__MR__',
              '-DCOREX_NATIVE_GQA_V_OVERLAP=1', '-x', 'ivcore',
              '--cuda-gpu-arch=ivcore11', '--cuda-path=/usr/local/corex']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2)+'\n')


def cpu_plan():
    return dict(source_transform=factory.verify(), explicit_cuda_flags=CUDA_FLAGS,
        sources={str(p.relative_to(factory.ROOT)):sha(p) for p in SOURCES+[HEADER]},
        requested='only packed K-next read after original V wait+barrier; publish after current PV',
        frontend_explicit_kernel_launches=0, input_initializations=0,
        actual_SO_instruction_order_verified=False, actual_counter_scope_verified=False,
        numeric_verified=False, performance_verified=False, precision_gate=False,
        service_installed=False, default_stdlib_only=True)


def compile_only(args, plan):
    out = args.out.resolve()
    if out.exists():
        raise ValueError('Use a new output directory')
    # Refuse a moving source root before importing any GPU package.
    dirty = subprocess.check_output(['git','-C',str(factory.ROOT),'status','--porcelain']).decode()
    if dirty:
        raise ValueError('Compile requires the clean frozen source commit')
    commit = subprocess.check_output(['git','-C',str(factory.ROOT),'rev-parse','HEAD']).decode().strip()
    subprocess.run(['git','-C',str(factory.ROOT),'merge-base','--is-ancestor',factory.BASE,commit],check=True)
    out.mkdir(parents=True)
    result = dict(plan, source_root=str(factory.ROOT), source_commit=commit,
                  compile_requested=True, compile_passed=False, completed=False,
                  commands=[], errors=[], actual_SO_export_completed=False)
    write(out/'cpu-plan.json', plan)
    for source in SOURCES+[HEADER]:
        (out/source.name).write_bytes(source.read_bytes())
    try:
        compiler = Path(args.compiler).resolve()
        objdump = Path(args.objdump).resolve()
        if not compiler.is_file() or not objdump.is_file():
            raise ValueError('Explicit existing CoreX compiler and ixobjdump required')
        version = subprocess.run([str(compiler),'--version'],capture_output=True,text=True)
        (out/'compiler-version.stdout.txt').write_text(version.stdout)
        (out/'compiler-version.stderr.txt').write_text(version.stderr)
        result['compiler'] = dict(path=str(compiler),sha256=sha(compiler),version_exit=version.returncode)
        result['objdump'] = dict(path=str(objdump),sha256=sha(objdump))
        if version.returncode:
            raise RuntimeError('Compiler version command failed')
        os.environ.setdefault('MAX_JOBS','1')
        os.environ.setdefault('TORCH_CUDA_ARCH_LIST','7.1')
        import torch
        from torch.utils.cpp_extension import load
        result['torch'] = dict(version=torch.__version__,file=torch.__file__,
                               cuda_initialized_before=torch.cuda.is_initialized())
        if result['torch']['cuda_initialized_before']:
            raise ValueError('Compile-only must start without a CUDA context')
        # Standalone artifacts are diagnostic, not proof of the final extension.
        for source in SOURCES[1:]:
            for suffix, flags in (('ll',['-S','-emit-llvm']),('s',['-S'])):
                target = out/(source.stem+'.'+suffix)
                command = [str(compiler),*CUDA_FLAGS,'--cuda-device-only',*flags,str(source),'-o',str(target)]
                proc = subprocess.run(command,capture_output=True,text=True)
                (out/(source.stem+'.'+suffix+'.stdout.txt')).write_text(proc.stdout)
                (out/(source.stem+'.'+suffix+'.stderr.txt')).write_text(proc.stderr)
                result['commands'].append(dict(argv=command,exit_code=proc.returncode))
                if proc.returncode or not target.is_file() or not target.stat().st_size:
                    raise RuntimeError('Missing/failed standalone artifact: '+str(target))
        build = out/'build';build.mkdir()
        library = load(name='cjj_native_k8_klookahead_probe',sources=list(map(str,SOURCES)),
            build_directory=str(build),extra_cflags=['-O3','-std=c++17'],
            extra_cuda_cflags=CUDA_FLAGS,is_python_module=False,verbose=True)
        result['library'] = dict(path=str(library),sha256=sha(library))
        # No op is called; preserve final SO exports, not only standalone text.
        for mode in ('IR','sass'):
            command = [str(objdump),'--'+mode,str(library)]
            proc = subprocess.run(command,capture_output=True,text=True)
            (out/('actual-SO-'+mode+'.stdout.txt')).write_text(proc.stdout)
            (out/('actual-SO-'+mode+'.stderr.txt')).write_text(proc.stderr)
            result['commands'].append(dict(argv=command,exit_code=proc.returncode))
            if proc.returncode or not proc.stdout.strip():
                raise RuntimeError('Final SO export failed: '+mode)
        result['actual_SO_export_completed'] = True
        result['torch']['cuda_initialized_after'] = torch.cuda.is_initialized()
        if result['torch']['cuda_initialized_after']:
            raise ValueError('Unexpected CUDA initialization in compile-only')
        result.update(compile_passed=True,completed=True)
    except Exception as exc:
        result['errors'].append(repr(exc))
        (out/'failure.txt').write_text(traceback.format_exc())
    result['artifact_manifest'] = [dict(relative=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(out.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    write(out/'compile-summary.json', result)
    # Only a review template. No boolean here licenses a consumer/frontend.
    write(out/'assembly-review-template.json',dict(
        compile_summary_sha256=sha(out/'compile-summary.json'),
        library=result.get('library'),source_commit=commit,source_root=str(factory.ROOT),
        approved=False,actual_SO_ELF_bytes_bound=False,
        both_S2_S4_full_next_last_and_original_tail_audited=False,
        Knext_global_after_current_V_wait_and_ready_barrier=False,
        no_new_global_pending_in_current_V_wait8_window=False,
        Knext_global_issued_before_current_PV=False,
        Knext_shared_publish_after_current_PV=False,
        Knext_dependency_store_then_end_barrier_before_next_QK=False,
        no_speculative_tail_or_segment_read=False,
        original_V_SME_wait_tail_QK_softmax_PV_math_preserved=False,
        resources={'S2':None,'S4':None},numeric_verified=False,performance_verified=False))
    print(json.dumps(result,sort_keys=True,indent=2))
    return 0 if result['compile_passed'] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compile-only',action='store_true')
    p.add_argument('--root-owned-exclusive',action='store_true')
    p.add_argument('--out',type=Path)
    p.add_argument('--compiler',default='/usr/local/corex/bin/clang++')
    p.add_argument('--objdump',default='/usr/local/corex/bin/ixobjdump')
    a = p.parse_args()
    plan = cpu_plan()
    if not a.compile_only:
        if a.out or a.root_owned_exclusive:
            p.error('Compile arguments require --compile-only')
        assert not {'torch','triton','numpy'}.intersection(sys.modules)
        print(json.dumps(plan,sort_keys=True,indent=2));return 0
    if not a.root_owned_exclusive:
        p.error('Root must confirm terminal prior owner and exclusive schedule')
    if a.out is None or not a.out.is_absolute() or a.out.exists():
        p.error('Use a fresh absolute output directory')
    if a.out.resolve().is_relative_to(factory.ROOT):
        p.error('Output must be outside source checkout')
    return compile_only(a,plan)


if __name__ == '__main__':
    raise SystemExit(main())
