#!/usr/bin/env python3
"""Local CPU audit of actual ef73 K-pair machine lowering; no device qualification."""
import argparse, ast, hashlib, json, re, struct, subprocess, sys
from pathlib import Path
SOURCE='52e1d2680d41e44f025b79fccb85867c8e783d39'
SO='ef732ef832b6399bf156d24c705b15e2e9e766947c0a456745673953ffbfb8fd'
OLD='2ad51a9812502e7dd8d81b99debd2160086a52510a93063f7ce4b0fbd8c837b1'
ELF_PIN='09ac1c98f2f1a1ff25ad02fcc819b25fc00bdb442a55c2ec1ad69defc61ea6e7'
HELPER_PIN='02fc4aff949091969a058bd1ffed64465b9a5baa73b1e7d57d6cf9178bcaa271'
INSPECT_PIN='b7a95a64fe60ebb475a78e37505b8eeb563fcf47d1c7efeba1caa2c3c0cf3380'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def reuse(raw,pin,names):
    assert sha(raw)==pin
    nodes=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'hash-bound old review helpers','exec'),globals())
def rename(text):
    for x,y in RENAMES:text=text.replace(str(len(x))+x,str(len(y))+y).replace(x,y)
    return text

def machine_paths(rows):
    at,dom,targets=machine_cfg(rows)
    dominates=lambda x,y:at[rows[x]['pc']] in dom[at[rows[y]['pc']]] and (at[rows[x]['pc']]!=at[rows[y]['pc']] or x<y)
    find=lambda op:[i for i,r in enumerate(rows) if r['op']==op]
    sme=find('sl_sme_load_16x1b64_rowxfb16');wait=[i for i,r in enumerate(rows) if r['op']=='sl_wait' and r['args']=='g2scnt(0)'];qk=find('ml_matrix_mad_i32_i8_m16n16k32');pv=find('ml_matrix_mad_f32_bf16_m16n16k16')
    assert (len(sme),len(wait),len(qk),len(pv))==(8,2,8,16)
    locate=lambda ids:[dict(index0=i,line=rows[i]['line'],PC=hex(rows[i]['pc']),op=rows[i]['op'],args=rows[i]['args']) for i in ids]
    paths=[]
    for n,label in enumerate(('full','tail')):
        vi=sme[n*4:n*4+4];qi=qk[n*4:n*4+4];pi=pv[n*8:n*8+8];wi=wait[n]
        kb=max(i for i in range(vi[0]) if rows[i]['op']=='sl_barrier')
        start=max(i for i in range(kb) if rows[i]['op']=='sl_mul_i64_i32' and rows[i]['args'].endswith(', 0x3100'))
        vector=[i for i in range(start,kb) if rows[i]['op']=='ml_lsa_load_dwordx2'];scalar=[i for i in range(start,kb) if rows[i]['op']=='ml_lsa_load_dword'];stores=[i for i in range(start,kb) if rows[i]['op']=='ml_slb_store_b32x1']
        assert (len(vector),len(scalar),len(stores))==(4,8,16)
        for i in vector+scalar+stores:must_reach(rows,targets,i,kb)
        assert all(dominates(kb,i) for i in vi)
        align=[i for i in range(start,kb) if rows[i]['op']=='sl_and_b32' and rows[i]['args'].endswith(', 0x7')];assert len(align)==1
        test=align[0];branch=test+(1 if label=='full' else 2)
        if label=='tail':assert (rows[test+1]['op'],rows[test+1]['args'])==('sl_mov_b32','s15, 0xffffffff') # Does not write WCR.
        assert rows[branch]['op']=='sl_cbr_wcrz'
        bypc={r['pc']:i for i,r in enumerate(rows)}
        def reachable_to_join(first):
            seen=set();todo=[first]
            while todo:
                i=todo.pop()
                if i==kb or i in seen:continue
                seen.add(i);r=rows[i]
                assert r['op'] not in ('sl_exit','sl_end','sl_ret')
                todo.extend([bypc[targets[r['pc']]]] if r['op']=='sl_jump' else [i+1,bypc[targets[r['pc']]]] if r['op'].startswith('sl_cbr_') else [i+1])
            return seen
        aligned=reachable_to_join(bypc[targets[rows[branch]['pc']]])
        scalar_only=reachable_to_join(branch+1)
        assert set(vector)<=aligned and not set(scalar)&aligned
        assert set(scalar)<=scalar_only and not set(vector)&scalar_only
        # Uniform scalar branch tests the global page/head base, not a lane offset.
        assert all(dominates(test,i) for i in vector+scalar)
        vb=next(i for i in range(wi+1,pi[0]) if rows[i]['op']=='sl_barrier')
        exp=[i for i in range(qi[-1]+1,len(rows)) if rows[i]['op']=='ml_exp_f32'][:2]
        end=next(i for i in range(qi[-1]+1,len(rows)) if rows[i]['op'].startswith('sl_cbr_')) if label=='full' else wi
        psums=[i for i in range(qi[-1]+1,end) if rows[i]['op']=='ml_add_f32'];casts=[i for i in range(qi[-1]+1,end) if rows[i]['op']=='ml_cvt_bf16_f32']
        assert (len(exp),len(psums),len(casts))==(2,8,2)
        assert all(dominates(i,wi) for i in vi+qi+exp+psums+casts)
        packing=[i for i in range(casts[-1]+1,len(rows)) if rows[i]['op']=='ml_or_b32'][:2]
        shifts=[i for i in range(casts[-1]+1,packing[-1]) if rows[i]['op']=='ml_shl_b32' and rows[i]['args'].endswith(', 0x10')];assert len(packing)==len(shifts)==2
        for i in packing+shifts:must_reach(rows,targets,i,wi)
        assert all(dominates(wi,i) and dominates(vb,i) for i in pi)
        clears=[i for i in range(wi+1,vb) if rows[i]['op']=='ml_slb_store_b32x1'];assert len(clears)==(0 if label=='full' else 2) and all(dominates(wi,i) for i in clears)
        loads=[i for i in range(vb+1,pi[-1]) if rows[i]['op'].startswith('ml_slb_') and 'load' in rows[i]['op']];assert len(loads)==16 and all(dominates(wi,i) for i in loads)
        paths.append(dict(path=label,page_stride=locate([start]),base_alignment_and_branch=locate(list(range(test,branch+1))),
            vector_loads=locate(vector),scalar_loads=locate(scalar),shared_K_stores=locate(stores),K_ready_barrier=locate([kb]),
            V_SME=locate(vi),QK=locate(qi),P_exp_casts_pack=locate(exp+casts+packing+shifts),V_wait=locate([wi]),
            tail_clear=locate(clears),V_ready_barrier=locate([vb]),PV_reads=locate(loads),PV_MMA=locate(pi),
            all_K_paths_rejoin_barrier=True,P_packing_before_wait=True,wait_dominates_tail_and_PV=True))
    return paths

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);p.add_argument('--raw',type=Path,required=True);p.add_argument('--reference-raw',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args();assert not a.report.exists()
    git=lambda rev,path:subprocess.check_output(['git','-C',str(a.repo),'show',rev+':'+path])
    reuse(git(SOURCE,'tools/audit_biv150_native_k8_compile_cpu.py'),ELF_PIN,('sha','msgpack','elf','embedded'))
    reuse(git('b54eb6b7edd64f0704443fa8e0c64052126116eb','docs/evidence/native-k8-compact-v-overlap-actual-so-20261029/original_root_replayed_review.py'),HELPER_PIN,('sha','elf_text','machine_cfg','llvm_functions','raw_manifest','actual_exports','must_reach','numeric_lines','alpha'))
    reuse(git(SOURCE,'docs/evidence/native112-n48-compile-20261029/review_actual_so.py'),INSPECT_PIN,('full_metadata','inspect'))
    factory=ast.parse(git(SOURCE,'tools/biv150_native_k8_compact_n48_kpair_factory.py'))
    nodes=[n for n in factory.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('NAMESPACE','RENAMES','SCALAR_LOOP','PAIR_LOOP')]
    constants={'__builtins__':{}};exec(compile(ast.Module(body=nodes,type_ignores=[]),'frozen literal source transformations','exec'),constants)
    global RENAMES
    RENAMES=constants['RENAMES']
    m=raw_manifest(a.raw);oldm=raw_manifest(a.reference_raw);assert (m['raw_files'],m['raw_bytes'])==(33,1541176)
    assert m['original_owner']==dict(pid=70801,start_ticks=356855231,same_start=True,state='Z')
    owner=json.loads((a.raw/'exports/cjj-native114-n48-kpair-compile-v1-20261029-owner.json').read_text());assert owner['source_commit']==SOURCE
    lib,texts,summary,inspection=actual_exports(a.raw,SO,'cjj_native_k8_compact_n48_kpair_probe.so');oldlib,oldtexts,olds,_=actual_exports(a.reference_raw,OLD,'cjj_native_k8_compact_n48_s4_probe.so')
    assert summary['completed'] and summary['compile_passed'] and not summary['errors'] and summary['torch']==olds['torch']
    assert not summary['torch']['cuda_initialized_before'] and not summary['torch']['cuda_initialized_after']
    assert summary['compiler']==olds['compiler'] and summary['explicit_cuda_flags']==olds['explicit_cuda_flags']
    assert all(summary[k]==0 for k in ('attention_launches','encoder_launches','codec_launches','reducer_launches','input_initializations'))
    assert len(summary['commands'])==4 and all(c['exit_code']==0 for c in summary['commands'])
    for row in summary['artifact_manifest']:
        b=(a.raw/'out'/row['relative']).read_bytes();assert (len(b),sha(b))==(row['bytes'],row['sha256'])
    inverse=[]
    for rel,pin in summary['private_source'].items():
        b=(a.raw/'out'/Path(rel).name).read_bytes();assert sha(b)==pin and b==git(SOURCE,rel)
        text=b.decode()
        for x,y in reversed(RENAMES):text=text.replace(y,x)
        if rel.endswith('_qk.cu'):assert text.count(constants['PAIR_LOOP'])==1;text=text.replace(constants['PAIR_LOOP'],constants['SCALAR_LOOP'])
        oldrel=rel.replace('_n48_kpair_','_n48_');assert text.encode()==git('e1c5a8fc3c267f1faf9aa790f3c8f80cc65c12a8',oldrel);inverse.append(dict(path=rel,sha256=pin,inverse_exact=True))
    meta,symbols,functions,ir,modules=inspect(lib,texts);ometa,osymbols,ofunctions,oir,omodules=inspect(oldlib,oldtexts)
    objects=[]
    for role in ('qk','codec'):objects+=embedded((a.raw/'out/build'/('biv150_native_k8_compact_n48_kpair_'+role+'.cuda.o')).read_bytes())
    assert sorted(x['sha256'] for x in meta)==sorted(x['sha256'] for x in objects)
    oldmodules=rename(json.dumps(omodules,sort_keys=True));assert oldmodules.replace('".vgpr_count": 100','".vgpr_count": 97').replace('".sgpr_count": 66','".sgpr_count": 69')==json.dumps(modules,sort_keys=True)
    result=[]
    for oldname,oldrows in ofunctions.items():
        name=rename(oldname);rows=functions[name];fn,line=ir[name];oldfn=oir[oldname][0]
        if '_producerILj' in name:
            assert len(rows)==1541 and len(oldrows)==1339
            numeric=alpha(numeric_lines(fn));assert len(numeric)==291 and numeric==alpha(numeric_lines(oldfn))
            align=[dict(line=line+i,text=x) for i,x in enumerate(fn.splitlines()) if re.search(r' = and i64 %\d+, 7$',x)];assert len(align)==2
            assert len(re.findall(r'= load i32, ptr addrspace\(1\) %\d+, align 8',fn))==8
            paths=machine_paths(rows)
            result.append(dict(symbol=name,instructions=1541,reference_instructions=1339,math_lines=291,
                math_canonical_sha256=sha('\n'.join(numeric).encode()),math_sequence_and_semantic_flags_equal=True,
                IR_alignment_tests=align,IR_pair_transport='two adjacent I32 loads align8/align4 become one dwordx2 in final SO',paths=paths,
                scalar_fallback_static_code_retained=True,actual_branch_zero_selects_only_vector_and_nonzero_only_scalar=True,actual_vector_lowering_verified=True,
                resource=symbols[name][2],SASS_first_line=rows[0]['line'],SASS_last_line=rows[-1]['line'],IR_line=line,host_admitted='ILj4EE' in name))
        else:
            assert b''.join(r['word'] for r in rows)==b''.join(r['word'] for r in oldrows)
            assert rename(oldfn.split('{\n',1)[1])==fn.split('{\n',1)[1]
    codec=next(x for x in meta if len(x['kernels'])==1);assert codec['text_sha256']=='936ce115af95401dda3a37cf310b62baa606003a18910ff46de4707fbaba19cf'
    assert not {'torch','triton','numpy'}&set(sys.modules)
    report=dict(schema=1,completed=True,errors=[],CPU_only=True,source_commit=SOURCE,SO_sha256=SO,reference_SO_sha256=OLD,
        script_sha256=sha(Path(__file__).read_bytes()),helper_pins=dict(ELF=ELF_PIN,CFG_IR=HELPER_PIN,inspect=INSPECT_PIN),
        raw_files_rehashed=m['raw_files'],raw_bytes_rehashed=m['raw_bytes'],download_manifest_sha256=sha((a.raw/'download-manifest.json').read_bytes()),
        reference_manifest_sha256=sha((a.reference_raw/'download-manifest.json').read_bytes()),original_owner=m['original_owner'],
        compile_summary_sha256=sha((a.raw/'out/compile-summary.json').read_bytes()),inspection_sha256=sha((a.raw/'out/actual-so-inspection.json').read_bytes()),
        source_inverse=inverse,compile_flags_and_compiler_unchanged=True,CUDA_initialized_before=False,CUDA_initialized_after=False,frontend_kernel_counts_zero=True,
        actual_embedded_GPU_images=meta,actual_GPU_images_equal_to_built_objects=True,actual_full_metadata_ABI_equal_except_names_and_register_counts=True,
        instructions_bound=sum(len(x) for x in functions.values()),actual_codec_text_unchanged=True,producer_reviews=result,
        static_resource_delta=dict(VGPR=[100,97],SGPR=[66,69],shared_bytes=[8192,8192],private_bytes=[0,0],VGPR_spills=[0,0],SGPR_spills=[0,0]),
        risks=['1541 vs1339 is static code size including mutually exclusive vector/scalar paths, not measured dynamic instruction cost.',
          'Extra uniform per-page alignment test and altered register allocation/scheduling may outweigh load-instruction reduction.',
          'Runtime allocation granularity/occupancy unknown; 97/69 does not prove a residency change.',
          'Original2ad and candidate whole-cache 4B-but-not8B V-SME device support both remain untested. Retained scalar K path does not establish either.',
          'Canonical numeric-line equality/source inverse are CPU evidence, not actual output verification; no numerics or speedup claimed.',
          'Unused S2 compared but not admitted; private host scope remains N48 S4.',
          'Actual IR output retains original signature wrap/truncation. Full metadata .args establishes ABI.'],
        actual_static_review_passed=True,actual_K_vector_lowering_verified=True,GPU_numeric_verified=False,
        hardware_alignment_verified=False,Graph_verified=False,side_stream_verified=False,fullstep_verified=False,
        occupancy_verified=False,performance_verified=False,service_verified=False,manual_Root_approval=False)
    a.report.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');print(json.dumps(dict(report=str(a.report),sha256=sha(a.report.read_bytes()),instructions_bound=report['instructions_bound'],static_review=True,GPU_numeric_verified=False)))
if __name__=='__main__':main()
