#!/usr/bin/env python3
"""CPU plan by default; explicit target compile-only never initializes inputs/launches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from biv150_native_k8_source_factory import ROOT,verify
from biv150_native_k8_qk_cpu import plan

SOURCES=[ROOT/'tools/kernels'/name for name in ('biv150_native_k8_qk.cpp','biv150_native_k8_qk.cu','biv150_native_k8_codec.cu')]
HEADER=ROOT/'tools/kernels/biv150_native_k8_codec.cuh'
CUDA_FLAGS=['-O3','-std=c++17','-D__ILUVATAR__','-D__MR__','-DCOREX_NATIVE_GQA_V_OVERLAP=0',
            '-x','ivcore','--cuda-gpu-arch=ivcore11','--cuda-path=/usr/local/corex']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def cpu_plan():
    result=plan()
    result.update(source_transform=verify(),private_source={str(p.relative_to(ROOT)):sha(p) for p in SOURCES+[HEADER]},
        requested='compile-only, not native numeric/performance acceptance',
        explicit_cuda_flags=CUDA_FLAGS,
        typed_ABI=dict(query='BF16 [batch64 or batch32,16,128], one query per seq; strides2048/2560,128,1',
          key_code='signed I8 [P,16,2,128], stride16384/256/128/1, full layer base',
          scale_full='F32 [P*4096], same full layer base; internal offset1024 floats',
          value_offset='BF16 [P,16,2,128], stride8192/256/128/1, data_ptr=fullbase+8192B',
          lengths='I32 [64 or32], owned CPU-preupload content prerequisite',
          table='I32 [rows,columns], owned CPU-preupload lengths and physical-page bounds prerequisite',
          partial='F32 [rows,16,S,128]',maximum_and_sum='F32 [rows,16,S]',
          input_finiteness='external owned finite Q/K/scale fixture; no GPU.item qualification',
          nonzero_layer_storage_offset=True,unknown_service_metadata_admitted=False),
        prototype_sources_now_present=True,native_CUDA_source_implemented=True,
        codec_intrinsic_header_compatibility='unknown until actual target compile',
        codec_requested_semantics='__fdiv_rn scale and quotient, __float2int_rn integer RNE; no fast-math fallback',
        codec_actual_IR_verified=False,attention_launches=0,encoder_launches=0,codec_launches=0,reducer_launches=0,input_initializations=0,
        append=dict(implementation='existing PR83 encode_append; not compiled/launched here',
          pointers='K I8, scale F32 and V BF16 all FULL page0 bases for PR83 writer; consumer V differs',
          runtime_args=['key_row_stride','value_row_stride','key_rows','value_rows','slot_count','cache_pages'],
          source_shape='BF16 K/V [rows,2,128], separate row pitches>=256; slotcount<=both extents',
          slot_contract='owned CPU provenance, invalid/negative before load mask, no duplicate/alias live writes',
          initial_encode_timed_separately=True,full_service_reencode=False),
        readonly_CPU_default=True)
    return result

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
        library=load(name='cjj_native_k8_qk_probe',sources=list(map(str,SOURCES)),build_directory=str(build),
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile-only',action='store_true')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--compiler',default='/usr/local/corex/bin/clang++',help='actual target CoreX clang driver')
    args=parser.parse_args();result=cpu_plan()
    if not args.compile_only:
        if args.out:raise ValueError('--out is for explicit compile-only; default prints a CPU plan')
        assert not {'torch','triton','numpy'}.intersection(sys.modules)
        print(json.dumps(result,sort_keys=True,indent=2));return 0
    if args.out is None:raise ValueError('--compile-only requires --out NEW_EMPTY_DIRECTORY')
    return compile_only(args,result)

if __name__=='__main__':raise SystemExit(main())
