#!/usr/bin/env python3
"""CPU source/preprocessor/frozen-Git checks; synthetic ELF is parser testing only."""
import contextlib,hashlib,io,json,re,shutil,struct,subprocess,sys,tempfile
from types import SimpleNamespace
from pathlib import Path
import compile_biv150_native_k8_mixed as build


def rejected(fn):
    try:fn()
    except (ValueError,RuntimeError,subprocess.CalledProcessError):return
    raise AssertionError('Expected rejection')


def synthetic_elf(path,names):
    # Small symbol-table fixture, with no executable code and never dlopened.
    strings=b'\0';symbols=b'\0'*24
    for name in names:
        offset=len(strings);strings+=name.encode()+b'\0'
        symbols+=struct.pack('<IBBHQQ',offset,0x12,0,3,0,0)
    str_off=64; sym_off=(str_off+len(strings)+7)//8*8; shoff=(sym_off+len(symbols)+7)//8*8
    header=bytearray(64);header[:6]=b'\x7fELF\x02\x01';header[6]=1
    struct.pack_into('<HHI',header,16,3,62,1);struct.pack_into('<Q',header,40,shoff)
    struct.pack_into('<HHHHHH',header,52,64,0,0,64,4,0)
    raw=header+strings;raw.extend(b'\0'*(sym_off-len(raw)));raw.extend(symbols);raw.extend(b'\0'*(shoff-len(raw)))
    raw.extend(b'\0'*64)
    raw.extend(struct.pack('<IIQQQQIIQQ',0,3,0,0,str_off,len(strings),0,0,1,0))
    raw.extend(struct.pack('<IIQQQQIIQQ',0,11,0,0,sym_off,len(symbols),1,1,8,24))
    raw.extend(struct.pack('<IIQQQQIIQQ',0,1,6,0,64,0,0,0,1,0))
    path.write_bytes(raw)


def run():
    plan=build.cpu_plan();record={'passed':False,'CPU_only':True,'source_transform':plan['source_transform'],
      'symbol_renames':build.SYMBOL_RENAMES,'host_device_macro_flags_identical':True,
      'actual_CoreX_preprocessing_or_compile_executed':False,'actual_new_SO_verified':False,'GPU_executed':False}
    for role in build.factory.FILES:
        p=build.factory.path(role);text=p.read_text();expanded=text
        for old,new in build.SYMBOL_RENAMES.items():expanded=re.sub(r'\b'+old+r'\b',new,expanded)
        inverse=expanded
        for old,new in build.SYMBOL_RENAMES.items():inverse=re.sub(r'\b'+new+r'\b',old,inverse)
        assert inverse==text
    assert all(any(re.search(r'\b'+name+r'\b',build.factory.path(role).read_text()) for role in build.factory.FILES)
               for name in build.SYMBOL_RENAMES)
    assert all(build.HOST_FLAGS.count(x)==build.CUDA_FLAGS.count(x)==1 for x in build.SYMBOL_FLAGS)
    with tempfile.TemporaryDirectory(prefix='cjj-native-mixed-build-cpu-') as td:
        root=Path(td).resolve();repo=root/'fixture-repo';repo.mkdir()
        def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.PIPE,text=True).strip()
        git('init','-q');(repo/'fixture.txt').write_text('CPU only\n');git('add','fixture.txt')
        git('-c','user.name=CPU fixture','-c','user.email=cpu-fixture@example.invalid','commit','-qm','CPU fixture')
        head=git('rev-parse','HEAD');assert build.frozen_source(repo,head,head)['clean']
        rejected(lambda:build.frozen_source(repo,head,'0'*40))
        rejected(lambda:build.frozen_source(repo,head,None))
        (repo/'fixture.txt').write_text('dirty\n');rejected(lambda:build.frozen_source(repo,head,head))
        (repo/'fixture.txt').write_text('CPU only\n');(repo/'untracked.txt').write_text('CPU only\n')
        rejected(lambda:build.frozen_source(repo,head,head));(repo/'untracked.txt').unlink()
        git('-c','user.name=CPU fixture','-c','user.email=cpu-fixture@example.invalid','commit','--allow-empty','-qm','CPU changed head')
        rejected(lambda:build.frozen_source(repo,head,head))
        record['frozen_source_tests']={'clean_accepted':True,'wrong_expected_ref_rejected':True,'missing_expected_ref_rejected':True,
          'tracked_dirty_rejected':True,'untracked_rejected':True,'moving_clean_HEAD_rejected':True}
        fake_compiler=root/'CPU-version-failure.py';fake_compiler.write_text('#!/usr/bin/env python3\nimport sys\nprint("CPU injected compiler version failure")\nprint("CPU stderr retained",file=sys.stderr)\nsys.exit(7)\n');fake_compiler.chmod(0o700)
        fake_objdump=root/'unused-CPU-objdump';fake_objdump.write_text('CPU fixture never executed\n')
        prior=(build.factory.ROOT,build.factory.BASE);current=git('rev-parse','HEAD');failure_out=root/'failure-out'
        try:
            build.factory.ROOT=repo;build.factory.BASE=current
            args=SimpleNamespace(out=failure_out,expected_source_commit=current,compiler=str(fake_compiler),objdump=str(fake_objdump))
            with contextlib.redirect_stdout(io.StringIO()):status=build.compile_only(args,{'CPU_failure_fixture':True})
        finally:build.factory.ROOT,build.factory.BASE=prior
        failed=json.loads((failure_out/'compile-summary.json').read_bytes())
        assert status==1 and not failed['completed'] and not failed['compile_passed'] and failed['errors']
        assert failed['torch']=={'cuda_initialized_before':None,'cuda_initialized_after':None}
        assert failed['commands'][0]['exit_code']==7 and (failure_out/'failure.txt').is_file()
        assert 'CPU stderr retained' in (failure_out/'compiler-version.stderr.txt').read_text()
        assert 'torch' not in sys.modules
        record['actual_frontend_CPU_failure_test']={'version_exit':7,'stdout_stderr_trace_summary_retained':True,
          'before_Torch_import':True,'unknown_CUDA_state_not_falsely_recorded_as_false':True}

        compiler=shutil.which('clang++') or shutil.which('g++')
        assert compiler,'CPU C++ preprocessor required for this check; no installation attempted'
        proc=subprocess.run([compiler,*build.HOST_FLAGS,'-E','-P','-x','c++','-'],
            input=build.identifier_probe_text(),capture_output=True,text=True,check=True)
        assert build.verify_identifier_probe(proc.stdout)==list(build.SYMBOL_RENAMES.values())
        assert build.verify_identifier_probe('struct implicit_CUDA_wrapper;\n'+proc.stdout+'\nvoid runtime_tail();')==list(build.SYMBOL_RENAMES.values())
        rejected(lambda:build.verify_identifier_probe(proc.stdout.replace('CJJ_MIXED_SYMBOL_PROBE_END','MISSING')))
        rejected(lambda:build.verify_identifier_probe(proc.stdout+proc.stdout))
        rejected(lambda:build.verify_identifier_probe(proc.stdout.replace(next(iter(build.SYMBOL_RENAMES.values())),'wrong_symbol')))
        record['CPU_preprocessor']={'compiler':compiler,'stdout':proc.stdout,'stderr':proc.stderr,
          'host_identifier_probe_only':True,'not_actual_CoreX_or_full_translation_unit':True}
        names=list(build.SYMBOL_RENAMES.values())[:5];elf=root/'not-executable-symbol-fixture.so'
        synthetic_elf(elf,names);inventory=build.elf_project_symbols(elf)
        assert inventory['old_project_symbol_count']==0
        synthetic_elf(elf,names+[next(iter(build.SYMBOL_RENAMES))]);rejected(lambda:build.elf_project_symbols(elf))
        synthetic_elf(elf,names[1:]);rejected(lambda:build.elf_project_symbols(elf))
        synthetic_elf(elf,names);raw=bytearray(elf.read_bytes());struct.pack_into('<H',raw,18,183);elf.write_bytes(raw)
        rejected(lambda:build.elf_project_symbols(elf))
        synthetic_elf(elf,names);raw=bytearray(elf.read_bytes());struct.pack_into('<Q',raw,40,len(raw)+1);elf.write_bytes(raw)
        rejected(lambda:build.elf_project_symbols(elf))
        record['synthetic_ELF_parser_tests']={'new_symbol_names_accepted':True,'old_project_symbol_rejected':True,
          'missing_launcher_rejected':True,'wrong_host_machine_rejected':True,'out_of_bounds_section_table_rejected':True,
          'actual_ELF_instruction_or_binding_qualification':False}
    from check_biv150_native_k8_service_cpu import SO
    assert build.sha(SO)=='8b86cb688ad72dcf9859994ed2092620312cdd5ae96d194904b2bde7926aacc5'
    rejected(lambda:build.elf_project_symbols(SO))
    record['actual_old8b_SO_rejected_as_new_isolated_library']=True
    record['passed']=True
    return record

if __name__=='__main__':print(json.dumps(run(),indent=2,sort_keys=True))
