#!/usr/bin/env python3
"""Default CPU plan; frozen-source compile and final-SO export, no kernel call."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import traceback
import biv150_native_k8_mixed_factory as factory

SOURCES=[factory.path(r) for r in ('qk.cpp','qk.cu','codec.cu')]
HEADER=factory.path('codec.cuh')
# Apply to host AND CUDA translation units. No source/mathematical text changes.
SYMBOL_RENAMES={
 'launch_cjj_native_k8_v_overlap_s2':'launch_cjj_native_k8_mixed_s2',
 'launch_cjj_native_k8_v_overlap_s4':'launch_cjj_native_k8_mixed_s4',
 'cjj_launch_native_k8_v_overlap_codec':'cjj_launch_native_k8_mixed_codec',
 'cjj_native_k8_v_overlap_gqa_producer':'cjj_native_k8_mixed_gqa_producer',
 'cjj_native_k8_v_overlap_codec_probe':'cjj_native_k8_mixed_codec_probe',
 'cjj_k8_scale_rn32':'cjj_mixed_k8_scale_rn32',
 'cjj_k8_code_rne':'cjj_mixed_k8_code_rne',
}
SYMBOL_FLAGS=['-D'+old+'='+new for old,new in SYMBOL_RENAMES.items()]
HOST_FLAGS=['-O3','-std=c++17',*SYMBOL_FLAGS]
CUDA_FLAGS=['-O3','-std=c++17','-D__ILUVATAR__','-D__MR__',
 '-DCOREX_NATIVE_GQA_V_OVERLAP=1','-x','ivcore',
 '--cuda-gpu-arch=ivcore11','--cuda-path=/usr/local/corex',*SYMBOL_FLAGS]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')


def frozen_source(root,base,expected):
    root=Path(root).resolve()
    if not re.fullmatch('[0-9a-f]{40}',expected or ''):raise ValueError('Explicit full frozen commit required')
    def git(*args):return subprocess.check_output(['git','-C',str(root),*args],text=True)
    if git('status','--porcelain','--untracked-files=all'):raise ValueError('Source must be clean, including untracked files')
    head=git('rev-parse','HEAD').strip()
    if head!=expected:raise ValueError('Source HEAD differs from explicit frozen commit')
    subprocess.run(['git','-C',str(root),'merge-base','--is-ancestor',base,head],check=True)
    return {'source_root':str(root),'source_commit':head,'base':base,'clean':True}


def cpu_plan():
    assert len(set(SYMBOL_RENAMES.values()))==len(SYMBOL_RENAMES)==7
    assert all(HOST_FLAGS.count(x)==CUDA_FLAGS.count(x)==1 for x in SYMBOL_FLAGS)
    return dict(source_transform=factory.verify(),explicit_host_flags=HOST_FLAGS,explicit_cuda_flags=CUDA_FLAGS,
      symbol_renames=SYMBOL_RENAMES,source_files={str(p.relative_to(factory.ROOT)):sha(p) for p in SOURCES+[HEADER]},
      frontend_sha256=sha(__file__),factory_sha256=sha(factory.__file__),
      requested='same CUDA/codec bytes, explicit isolated host/device symbols and exact mixed rows',
      attention_launches=0,encoder_launches=0,codec_launches=0,reducer_launches=0,input_initializations=0,
      actual_SO_export_completed=False,actual_symbol_binding_verified=False,
      actual_GPU_math_or_resources_verified=False,numeric_verified=False,performance_verified=False,
      precision_gate=False,service_installed=False,readonly_CPU_default=True)


def command(result,out,name,argv):
    p=subprocess.run(argv,capture_output=True,text=True)
    (out/(name+'.stdout.txt')).write_text(p.stdout);(out/(name+'.stderr.txt')).write_text(p.stderr)
    result['commands'].append({'argv':argv,'exit_code':p.returncode})
    return p


def identifier_probe_text():
    return '\n'.join(['CJJ_MIXED_SYMBOL_PROBE_BEGIN',*SYMBOL_RENAMES,'CJJ_MIXED_SYMBOL_PROBE_END'])+'\n'


def verify_identifier_probe(text):
    # CUDA preprocessing can emit implicit runtime-wrapper declarations. Only
    # the uniquely delimited source probe is compared, never assumed bare text.
    tokens=re.findall(r'\b[A-Za-z_]\w*\b',text)
    begin=[i for i,x in enumerate(tokens) if x=='CJJ_MIXED_SYMBOL_PROBE_BEGIN']
    end=[i for i,x in enumerate(tokens) if x=='CJJ_MIXED_SYMBOL_PROBE_END']
    if len(begin)!=1 or len(end)!=1 or tokens[begin[0]+1:end[0]]!=list(SYMBOL_RENAMES.values()):
        raise ValueError('Actual preprocessor symbol expansion differs')
    return list(SYMBOL_RENAMES.values())


def elf_project_symbols(path):
    """Bounded ELF64 symbol-name inventory, not an instruction/math audit."""
    raw=Path(path).read_bytes()
    if raw[:6]!=b'\x7fELF\x02\x01' or struct.unpack_from('<H',raw,18)[0]!=62:raise ValueError('Expected ELF64 little-endian host x86_64 SO')
    shoff=struct.unpack_from('<Q',raw,40)[0];size,count=struct.unpack_from('<HH',raw,58)
    if size!=64 or not 0<count<65535 or shoff+size*count>len(raw):raise ValueError('Unsupported ELF section table')
    sections=[struct.unpack_from('<IIQQQQIIQQ',raw,shoff+i*size) for i in range(count)];found=[]
    for s in sections:
        if s[1] not in (2,11):continue
        if s[9]!=24 or s[5]%24 or s[6]>=count:raise ValueError('Invalid ELF symbol table')
        strings=sections[s[6]];start,length=strings[4:6]
        if start+length>len(raw) or s[4]+s[5]>len(raw):raise ValueError('ELF table outside file')
        table=raw[start:start+length]
        for pos in range(s[4],s[4]+s[5],24):
            name,info,other,section,value,nbytes=struct.unpack_from('<IBBHQQ',raw,pos)
            if name>=len(table):raise ValueError('ELF string index')
            end=table.find(b'\0',name)
            if end<0:raise ValueError('ELF unterminated symbol')
            text=table[name:end].decode('utf-8','strict')
            if any(x in text for x in (*SYMBOL_RENAMES,*SYMBOL_RENAMES.values())):
                found.append({'name':text,'defined':section!=0,'binding':info>>4,'type':info&15,'table_type':s[1]})
    old=[r for r in found if any(x in r['name'] for x in SYMBOL_RENAMES)]
    if old:raise ValueError('Old project symbol remains in final ELF: '+repr(old))
    launchers=list(SYMBOL_RENAMES.values())[:3]
    if not all(any(r['name']==x and r['defined'] and r['type']==2 for r in found) for x in launchers):
        raise ValueError('Missing isolated exported launcher')
    for x in list(SYMBOL_RENAMES.values())[3:5]:
        if not any(x in r['name'] and r['defined'] for r in found):raise ValueError('Missing renamed device stub/kernel symbol: '+x)
    return {'library_sha256':sha(path),'symbols':found,'old_project_symbol_count':0,
            'instruction_math_verified':False,'runtime_launch_performed':False}


def launcher_owners(library):
    """Only dlsym/dladdr; do not call any launcher/device stub or CUDA API."""
    import ctypes
    class Info(ctypes.Structure):
        _fields_=[('dli_fname',ctypes.c_char_p),('dli_fbase',ctypes.c_void_p),
                  ('dli_sname',ctypes.c_char_p),('dli_saddr',ctypes.c_void_p)]
    library=Path(library).resolve();handle=ctypes.CDLL(str(library),mode=os.RTLD_NOW|os.RTLD_LOCAL)
    dladdr=ctypes.CDLL(None).dladdr;dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(Info)];dladdr.restype=ctypes.c_int
    rows=[]
    for symbol in list(SYMBOL_RENAMES.values())[:3]:
        address=ctypes.cast(getattr(handle,symbol),ctypes.c_void_p);info=Info()
        if not dladdr(address,ctypes.byref(info)) or not info.dli_fname:raise ValueError('dladdr failed: '+symbol)
        owner=Path(os.fsdecode(info.dli_fname)).resolve()
        if owner!=library or not info.dli_sname or info.dli_sname.decode()!=symbol:raise ValueError('Launcher resolved to foreign SO/symbol: '+symbol)
        rows.append({'symbol':symbol,'owner':str(owner),'library_sha256':sha(owner),'called':False})
    return rows


def compile_only(args,plan):
    out=args.out.resolve()
    if out.exists():raise ValueError('Use new output directory')
    frozen=frozen_source(factory.ROOT,factory.BASE,args.expected_source_commit)
    out.mkdir(parents=True)
    result=dict(plan,**frozen,compile_requested=True,compile_passed=False,completed=False,commands=[],errors=[],
                torch={'cuda_initialized_before':None,'cuda_initialized_after':None})
    write(out/'cpu-plan.json',plan)
    for p in SOURCES+[HEADER]:(out/p.name).write_bytes(p.read_bytes())
    torch=None
    try:
        compiler=Path(args.compiler).resolve();objdump=Path(args.objdump).resolve()
        if not compiler.is_file() or not objdump.is_file():raise ValueError('Existing CoreX compiler and ixobjdump required')
        result['compiler']={'path':str(compiler),'sha256':sha(compiler)};result['objdump']={'path':str(objdump),'sha256':sha(objdump)}
        if command(result,out,'compiler-version',[str(compiler),'--version']).returncode:raise RuntimeError('Compiler version failed')
        # Actual compiler preprocessor, identifier-only probe, both host and device
        # flag sets. These outputs do not replace final ELF/IR/SASS inspection.
        probe=out/'symbol-preprocessor-probe.cc';probe.write_text(identifier_probe_text())
        for mode,flags in [('host',[*HOST_FLAGS,'-x','c++']),('device',[*CUDA_FLAGS,'--cuda-device-only'])]:
            p=command(result,out,'symbol-preprocessor-'+mode,[str(compiler),*flags,'-E','-P',str(probe)])
            if p.returncode:raise RuntimeError('Symbol preprocessor failed: '+mode)
            verify_identifier_probe(p.stdout)
        result['actual_identifier_preprocessing_verified']=True
        os.environ.setdefault('MAX_JOBS','1');os.environ.setdefault('TORCH_CUDA_ARCH_LIST','7.1')
        import torch
        from torch.utils.cpp_extension import load
        result['torch'].update(version=torch.__version__,file=torch.__file__,cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),
                               cuda_initialized_before=torch.cuda.is_initialized())
        if result['torch']['cuda_initialized_before']:raise ValueError('Compile-only must begin without CUDA initialization')
        for source in SOURCES[1:]:
            for suffix,flags in [('ll',['-S','-emit-llvm']),('s',['-S'])]:
                target=out/(source.stem+'.'+suffix)
                p=command(result,out,source.stem+'.'+suffix,[str(compiler),*CUDA_FLAGS,'--cuda-device-only',*flags,str(source),'-o',str(target)])
                if p.returncode or not target.is_file() or not target.stat().st_size:raise RuntimeError('Standalone export failed: '+str(target))
        build=out/'build';build.mkdir()
        library=load(name=factory.NAMESPACE,sources=list(map(str,SOURCES)),build_directory=str(build),
                     extra_cflags=HOST_FLAGS,extra_cuda_cflags=CUDA_FLAGS,is_python_module=False,verbose=True)
        result['library']={'path':str(library),'sha256':sha(library)}
        export_errors=[]
        for mode in ('IR','sass'):
            p=command(result,out,'actual-SO-'+mode,[str(objdump),'--'+mode,str(library)])
            if p.returncode or not p.stdout.strip():export_errors.append(mode)
        if export_errors:raise RuntimeError('Final SO export failed: '+','.join(export_errors))
        result['actual_SO_export_completed']=True
        symbols=elf_project_symbols(library);write(out/'actual-SO-symbols.json',symbols)
        result['actual_host_launcher_owners']=launcher_owners(library)
        result['actual_symbol_binding_verified']=True
        if cpu_plan()!=plan or frozen_source(factory.ROOT,factory.BASE,args.expected_source_commit)!=frozen:
            raise ValueError('Frozen source/flags moved during compile')
        result['compile_passed']=True
    except Exception as exc:
        result['errors'].append(repr(exc));(out/'failure.txt').write_text(traceback.format_exc())
    finally:
        if torch is not None:
            try:
                result['torch']['cuda_initialized_after']=torch.cuda.is_initialized()
                if result['torch']['cuda_initialized_after']:result['errors'].append('Unexpected CUDA initialization in compile-only')
            except Exception as exc:result['errors'].append('CUDA-state observation failed: '+repr(exc))
    result['compile_passed']=result['compile_passed'] and not result['errors']
    result['completed']=result['compile_passed']
    result['artifact_manifest']=[{'relative':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':sha(p)}
        for p in sorted(out.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    write(out/'compile-summary.json',result)
    write(out/'assembly-review-template.json',{'compile_summary_sha256':sha(out/'compile-summary.json'),
        'source_commit':frozen['source_commit'],'library':result.get('library'),'symbol_renames':SYMBOL_RENAMES,
        'approved':False,'actual_SO_ELF_IR_SASS_bound':False,'normalized_GPU_math_matches_accepted8b':False,
        'new_rows31_63_host_ABI_verified':False,'dual_library_coexistence_verified':False,
        'full_tail_wait_and_codec_verified':False,'resources':{'S2':None,'S4':None},
        'numeric_verified':False,'service_verified':False})
    print(json.dumps(result,indent=2,sort_keys=True));return 0 if result['compile_passed'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compile-only',action='store_true');p.add_argument('--root-owned-exclusive',action='store_true')
    p.add_argument('--out',type=Path);p.add_argument('--expected-source-commit')
    p.add_argument('--compiler',default='/usr/local/corex/bin/clang++');p.add_argument('--objdump',default='/usr/local/corex/bin/ixobjdump')
    a=p.parse_args();plan=cpu_plan()
    if not a.compile_only:
        if a.out or a.root_owned_exclusive or a.expected_source_commit:p.error('Compile arguments require --compile-only')
        assert not {'torch','triton','numpy'}.intersection(sys.modules)
        print(json.dumps(plan,indent=2,sort_keys=True));return 0
    if not a.root_owned_exclusive:p.error('Root exclusive ownership required')
    if a.out is None or not a.out.is_absolute() or a.out.exists() or a.out.resolve().is_relative_to(factory.ROOT):
        p.error('Fresh absolute output outside source root required')
    if not re.fullmatch('[0-9a-f]{40}',a.expected_source_commit or ''):p.error('Explicit full --expected-source-commit required')
    return compile_only(a,plan)

if __name__=='__main__':raise SystemExit(main())
