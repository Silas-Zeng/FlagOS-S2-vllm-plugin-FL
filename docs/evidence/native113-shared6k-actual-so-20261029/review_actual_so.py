#!/usr/bin/env python3
"""CPU-only actual 5cf6/2ad ELF/IR/instruction comparison; never load either SO."""
import argparse, ast, hashlib, json, re, struct, subprocess, sys
from pathlib import Path
SO='5cf698e8a60272d86b2545364f580faf4c40114e0044e91557c1a137f2521bed'
OLD='2ad51a9812502e7dd8d81b99debd2160086a52510a93063f7ce4b0fbd8c837b1'
SOURCE='1dc7ce5c2dd9050712aa9d5e7cba59fe3699f8d4'
ELF_PIN='09ac1c98f2f1a1ff25ad02fcc819b25fc00bdb442a55c2ec1ad69defc61ea6e7'
HELPER_PIN='02fc4aff949091969a058bd1ffed64465b9a5baa73b1e7d57d6cf9178bcaa271'
INSPECT_PIN='b7a95a64fe60ebb475a78e37505b8eeb563fcf47d1c7efeba1caa2c3c0cf3380'
# Zero-based instruction positions in each actual 1339-instruction producer.
DELTA={
580:('sl_mov_b32 s55, 0x1000','sl_mov_b32 s55, 0x800'),
581:('sl_mov_b32 s57, 0x1400','sl_mov_b32 s57, 0xc00'),
583:('sl_mov_b32 s59, 0x1800','sl_mov_b32 s59, 0x1000'),
585:('sl_mov_b32 s61, 0x1c00','sl_mov_b32 s61, 0x1400'),
730:('ml_lsa_load_dword v32, s[12:15], unsigned v44, s55, #0, ca|mie','ml_lsa_load_dword v32, s[12:15], unsigned v44, s59, #0, ca|mie'),
941:('sl_mov_b32 s13, 0x1000','sl_mov_b32 s13, 0x800'),
944:('sl_mov_b32 s15, 0x1400','sl_mov_b32 s13, 0xc00'),
945:('sl_mov_b32 s14, 0x40','sl_mov_b32 s12, 0x40'),
946:('sl_sme_load_16x1b64_rowxfb16 s[36:39], s[14:15], 0x0, 0x0, 0x0, 0x1','sl_sme_load_16x1b64_rowxfb16 s[36:39], s[12:13], 0x0, 0x0, 0x0, 0x1'),
947:('sl_mov_b32 s15, 0x1800','sl_mov_b32 s13, 0x1000'),
948:('sl_mov_b32 s14, 0x80','sl_mov_b32 s12, 0x80'),
949:('sl_sme_load_16x1b64_rowxfb16 s[36:39], s[14:15], 0x0, 0x0, 0x0, 0x1','sl_sme_load_16x1b64_rowxfb16 s[36:39], s[12:13], 0x0, 0x0, 0x0, 0x1'),
950:('sl_mov_b32 s15, 0x1c00','sl_mov_b32 s15, 0x1400'),
1092:('sl_mov_b32 s14, 0x1000','sl_mov_b32 s14, 0x800'),
1142:('sl_mov_b32 s2, 0x1000','sl_mov_b32 s2, 0x800'),
1156:('sl_mov_b32 s2, 0x1400','sl_mov_b32 s2, 0xc00'),
1175:('sl_mov_b32 s2, 0x1800','sl_mov_b32 s2, 0x1000'),
1194:('sl_mov_b32 s2, 0x1c00','sl_mov_b32 s2, 0x1400')}
def sha(b):return hashlib.sha256(b).hexdigest()
def reuse(raw,pin,names):
    assert sha(raw)==pin
    ns=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in ns}==set(names)
    exec(compile(ast.Module(body=ns,type_ignores=[]),'hash-bound existing helpers','exec'),globals())
def rename(text):
    for old,new in RENAMES:
        text=text.replace(str(len(old))+old,str(len(new))+new).replace(old,new)
    return text

def shared_ir(old):
    text=rename(old);changes=[]
    def gep(m):
        n=int(m[2])
        if n==1024:return m[0] # Original second I8 K panel; leave its shared address untouched.
        assert n in range(4096,7681,512)
        changes.append(('V_shared_GEP',n,n-2048));return m[1]+str(n-2048)
    text=re.sub(r'(@_ZZ43cjj_native_k8_compact_n48_shared6k_producer[^\n ]+E5stage, i32 )(\d+)',gep,text)
    def sme(m):
        n=int(m[2]);assert n in (4096,5120,6144,7168)
        changes.append(('V_SME_destination',n,n-2048));return m[1]+str(n-2048)
    text=re.sub(r'(@llvm.bi.sme.load.16x1b64.rowxfb16\(i32 add \(i32 ptrtoint \(ptr addrspacecast \(ptr addrspace\(3\) @_ZZ[^\n]+?to i32\), i32 )(\d+)',sme,text)
    assert text.count(' = or disjoint i32 %1268, 1024')==1
    text=text.replace(' = or disjoint i32 %1268, 1024',' = add nuw nsw i32 %1268, 512')
    assert len(changes)==24 and sum(x[0]=='V_shared_GEP' for x in changes)==16
    return text,changes

def host_binding(raw,device):
    h=struct.unpack_from('<16sHHIQQQIHHHHHH',raw);assert h[2]==62
    rows=[struct.unpack_from('<IIQQQQIIQQ',raw,h[6]+i*h[11]) for i in range(h[12])]
    names=raw[rows[h[13]][4]:rows[h[13]][4]+rows[h[13]][5]]
    sec={names[r[0]:names.index(b'\0',r[0])].decode():r for r in rows}
    def data(r):return raw[r[4]:r[4]+r[5]]
    def memory(addr,n):
        r=next(r for r in rows if r[1]!=8 and r[3]<=addr and addr+n<=r[3]+r[5]);return raw[r[4]+addr-r[3]:r[4]+addr-r[3]+n]
    symbols=[];s=sec['.dynsym'];names=data(rows[s[6]])
    for i in range(0,s[5],s[9]):
        n,info,other,index,value,size=struct.unpack_from('<IBBHQQ',data(s),i)
        symbols.append(dict(name=names[n:names.index(b'\0',n)].decode(),binding=info>>4,type=info&15,visibility=other&3,section=index,value=value,size=size))
    stub='_Z58__device_stub__cjj_native_k8_compact_n48_shared6k_producerILj4EEvPK13__ix_bfloat16PKaPKfS2_PKiS8_fllllllllPfS9_S9_'
    index=next(i for i,s in enumerate(symbols) if s['name']==stub);s=symbols[index]
    assert {k:s[k] for k in ('binding','type','visibility','section','value','size')}==dict(binding=2,type=2,visibility=0,section=12,value=0x189c0,size=381)
    rels={}
    for key in ('.rela.dyn','.rela.plt'):
        r=sec[key]
        for i in range(0,r[5],r[9]):
            address,info,addend=struct.unpack_from('<QQq',data(r),i);rels[address]=(symbols[info>>32]['name'],info&0xffffffff,addend)
    got=0x2de58;assert rels[got]==(stub,6,0)
    def rip(pc,prefix):
        b=memory(pc,len(prefix)+4);assert b.startswith(prefix);return pc+len(b)+struct.unpack('<i',b[len(prefix):])[0]
    assert rip(0x187f5,b'\x48\x8b\x35')==rip(0x18b1c,b'\x48\x8b\x3d')==got
    nameaddr=rip(0x187fc,b'\x48\x8d\x0d');assert nameaddr==0x1968e
    assert memory(nameaddr,len(device)+1)==device.encode()+b'\0'
    assert memory(0x18803,15)==bytes.fromhex('4889df4889ca41b8ffffffff4531c9')
    register=rip(0x18812,b'\xe8');launch=rip(0x18b30,b'\xe8')
    assert rels[rip(register,b'\xff\x25')]==('__ixRegisterFunction',7,0)
    assert rels[rip(launch,b'\xff\x25')]==('ixLaunchKernel',7,0)
    return dict(dynsym=s,device_symbol=device,GOT=hex(got),GOT_relocation='R_X86_64_GLOB_DAT',
      registration_stub_load_PC='0x187f5',registration_name_PC=hex(nameaddr),registration_call_PC='0x18812',
      launch_stub_load_PC='0x18b1c',launch_call_PC='0x18b30',same_GOT_and_registered_device_name_verified=True,
      query_api='runtime cuda/ixFuncGetAttributes + cuda/ixOccupancyMaxActiveBlocksPerMultiprocessor; not CUfunction',
      requested_query_threads=64,requested_query_dynamic_shared_bytes=0,
      actual_runtime_handle_resolved=False,runtime_attributes=None,active_blocks_per_multiprocessor=None,
      runtime_limits='Load exact existing SO through normal Torch process, dlsym this export, dladdr exact SO; do not call stub. ABI helper must compile against installed SDK. Query failure must not block same-input ABBA.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);p.add_argument('--raw',type=Path,required=True);p.add_argument('--reference-raw',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args();assert not a.report.exists()
    git=lambda ref,path:subprocess.check_output(['git','-C',str(a.repo),'show',ref+':'+path])
    reuse(git(SOURCE,'tools/audit_biv150_native_k8_compile_cpu.py'),ELF_PIN,('sha','msgpack','elf','embedded'))
    reuse(git('b54eb6b7edd64f0704443fa8e0c64052126116eb','docs/evidence/native-k8-compact-v-overlap-actual-so-20261029/original_root_replayed_review.py'),HELPER_PIN,('sha','elf_text','machine_cfg','llvm_functions','raw_manifest','actual_exports'))
    reuse(git(SOURCE,'docs/evidence/native112-n48-compile-20261029/review_actual_so.py'),INSPECT_PIN,('full_metadata','inspect'))
    global RENAMES
    factory=ast.parse(git(SOURCE,'tools/biv150_native_k8_compact_n48_shared6k_factory.py'))
    namespace=next(ast.literal_eval(n.value) for n in factory.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='NAMESPACE')
    class ResolveNamespace(ast.NodeTransformer):
        def visit_Name(self,node):
            assert node.id=='NAMESPACE';return ast.Constant(namespace)
    constants={n.targets[0].id:ast.literal_eval(ResolveNamespace().visit(n.value)) for n in factory.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('RENAMES','CUDA_EDITS')};RENAMES=constants['RENAMES']
    m=raw_manifest(a.raw);oldm=raw_manifest(a.reference_raw);assert (m['raw_files'],m['raw_bytes'])==(33,1426395)
    assert m['original_owner']==dict(pid=70555,start_ticks=356808360,same_start=True,state='Z')
    owner=json.loads((a.raw/'exports/cjj-native113-n48-shared6k-compile-v1-20261029-owner.json').read_text());assert owner['source_commit']==SOURCE
    lib,texts,summary,inspection=actual_exports(a.raw,SO,'cjj_native_k8_compact_n48_shared6k_probe.so')
    oldlib,oldtexts,oldsummary,_=actual_exports(a.reference_raw,OLD,'cjj_native_k8_compact_n48_s4_probe.so')
    assert summary['completed'] and summary['compile_passed'] and not summary['errors']
    assert summary['torch']==oldsummary['torch'] and summary['explicit_cuda_flags']==oldsummary['explicit_cuda_flags'] and summary['compiler']==oldsummary['compiler']
    assert not summary['torch']['cuda_initialized_before'] and not summary['torch']['cuda_initialized_after']
    assert all(summary[x]==0 for x in ('attention_launches','encoder_launches','codec_launches','reducer_launches','input_initializations'))
    assert len(summary['commands'])==4 and all(c['exit_code']==0 for c in summary['commands'])
    for row in summary['artifact_manifest']:
        b=(a.raw/'out'/row['relative']).read_bytes();assert (len(b),sha(b))==(row['bytes'],row['sha256'])
    source_inverse=[]
    for rel,pin in summary['private_source'].items():
        b=(a.raw/'out'/Path(rel).name).read_bytes();assert sha(b)==pin and b==git(SOURCE,rel)
        oldrel=rel.replace('_n48_shared6k_','_n48_');text=b.decode()
        for x,y in reversed(RENAMES):text=text.replace(y,x)
        if rel.endswith('_qk.cu'):
            for x,y in reversed(constants['CUDA_EDITS']):assert text.count(y)==1;text=text.replace(y,x)
        assert text.encode()==git('e1c5a8fc3c267f1faf9aa790f3c8f80cc65c12a8',oldrel)
        source_inverse.append(dict(path=rel,sha256=pin,inverse_exact=True))
    meta,symbols,funcs,ir,modules=inspect(lib,texts);ometa,osymbols,ofuncs,oir,omodules=inspect(oldlib,oldtexts)
    objects=[]
    for role in ('qk','codec'):objects+=embedded((a.raw/'out/build'/('biv150_native_k8_compact_n48_shared6k_'+role+'.cuda.o')).read_bytes())
    assert sorted(x['sha256'] for x in meta)==sorted(x['sha256'] for x in objects)
    oldmods=rename(json.dumps(omodules,sort_keys=True));assert oldmods.replace('".group_segment_fixed_size": 8192','".group_segment_fixed_size": 6144')==json.dumps(modules,sort_keys=True)
    for pattern in (r'^attributes #\d+ = .*$',r'^!\d+ = .*$'):
        assert rename('\n'.join(re.findall(pattern,oldtexts['IR'],re.M)))=='\n'.join(re.findall(pattern,texts['IR'],re.M))
    assert 'clear_k_tail' not in texts['IR'] and 'k_half_index' not in texts['IR']
    assert texts['IR'].count('global [1536 x i32] undef, comdat, align 16')==2
    assert 'global [2048 x i32]' not in texts['IR']
    prior=json.loads(git(SOURCE,'docs/evidence/native112-n48-compile-20261029/actual-so-review.json'))
    comparisons=[]
    for oldname,oldrows in ofuncs.items():
        name=rename(oldname);rows=funcs[name];assert len(rows)==len(oldrows)
        isproducer='_producerILj' in name;changes=[]
        for i,(x,y) in enumerate(zip(oldrows,rows)):
            xa=x['op']+' '+x['args'];ya=y['op']+' '+y['args']
            if xa!=ya:
                assert isproducer and (xa,ya)==DELTA[i]
                changes.append(dict(index0=i,old_line=x['line'],line=y['line'],old_PC=hex(x['pc']),PC=hex(y['pc']),old=xa,new=ya,old_word=x['word'].hex(),word=y['word'].hex()))
            else:assert x['word']==y['word']
            assert x['op']==y['op']
        assert [r['index0'] for r in changes]==(list(DELTA) if isproducer else [])
        oldbody=oir[oldname][0].split('{\n',1)[1];body=ir[name][0].split('{\n',1)[1]
        if isproducer:
            normalized,irs=shared_ir(oldbody);assert normalized==body
            a0,d0,t0=machine_cfg(oldrows);a1,d1,t1=machine_cfg(rows)
            ix0={r['pc']:i for i,r in enumerate(oldrows)};ix1={r['pc']:i for i,r in enumerate(rows)}
            assert {ix0[k]:ix0[v] for k,v in t0.items()}=={ix1[k]:ix1[v] for k,v in t1.items()}
            assert {ix0[k]:sorted(ix0[v] for v in vs) for k,vs in d0.items()}=={ix1[k]:sorted(ix1[v] for v in vs) for k,vs in d1.items()}
            # Global scale load is still +4096 bytes; changed register holds that exact constant.
            assert rows[583]['args']=='s59, 0x1000' and rows[730]['args'].endswith('s59, #0, ca|mie')
            assert not any(re.match(r's59(?:,|$)|s\[(?:5[0-9]):(?:5[9]|6[0-9])\]',r['args']) and r['op'] not in ('sl_sme_load_16x1b64_rowxfb16',) for r in rows[584:730])
        else:assert rename(oldbody)==body;irs=[]
        comparisons.append(dict(symbol=name,instructions=len(rows),changed_instructions=changes,IR_address_translations=irs,
          actual_IR_and_ABI_checked=True,IR_fullbody_only_declared_shared_translations=isproducer,IR_body_sha256=sha(body.encode()),
          IR_line=ir[name][1],SASS_first_line=rows[0]['line'],SASS_last_line=rows[-1]['line'],
          all_other_machine_words_exact=True,CFG_branches_and_dominators_equal=isproducer,host_admitted='ILj4EE' in name or 'codec_probe' in name))
    s4=next(name for name in funcs if 'producerILj4EE' in name);resource=symbols[s4][2]
    expected={'.vgpr_count':100,'.sgpr_count':66,'.group_segment_fixed_size':6144,'.private_segment_fixed_size':0,'.vgpr_spill_count':0,'.sgpr_spill_count':0,'.warp_size':64,'.kernarg_segment_size':144,'.kernarg_segment_align':8}
    assert all(resource[k]==v for k,v in expected.items())
    codec=next(x for x in meta if len(x['kernels'])==1);assert codec['text_sha256']=='936ce115af95401dda3a37cf310b62baa606003a18910ff46de4707fbaba19cf'
    oldoffsets=[]
    for i in range(1024):
        slot=(i//128)*2;dim=i%128;r=slot//2;panel=dim//16;c=dim%16
        half=panel*256+(r//4)*128+((r*32+c*2+(slot&1)+panel*64)&127);v=half//2
        assert 0<=v<1024 and (v|1024)-(v+512)==512;oldoffsets.append(v)
    assert sorted(oldoffsets)==list(range(1024))
    host=host_binding(lib,s4)
    assert not {'torch','triton','numpy'}&set(sys.modules)
    report=dict(schema=1,completed=True,errors=[],CPU_only=True,source_commit=SOURCE,SO_sha256=SO,reference_SO_sha256=OLD,
      script_sha256=sha(Path(__file__).read_bytes()),helper_pins=dict(ELF=ELF_PIN,CFG_IR=HELPER_PIN,inspect=INSPECT_PIN),
      raw_files_rehashed=m['raw_files'],raw_bytes_rehashed=m['raw_bytes'],download_manifest_sha256=sha((a.raw/'download-manifest.json').read_bytes()),
      reference_manifest_sha256=sha((a.reference_raw/'download-manifest.json').read_bytes()),original_owner=m['original_owner'],
      compile_summary_sha256=sha((a.raw/'out/compile-summary.json').read_bytes()),inspection_sha256=sha((a.raw/'out/actual-so-inspection.json').read_bytes()),
      source_inverse=source_inverse,compile_flags_and_compiler_unchanged=True,CUDA_initialized_before=False,CUDA_initialized_after=False,frontend_kernel_counts_zero=True,
      actual_embedded_GPU_images=meta,actual_GPU_images_equal_to_built_objects=True,actual_full_metadata_ABI_equal_except_names_and_shared=True,
      instructions_bound=sum(len(x) for x in funcs.values()),functions=comparisons,actual_S4_resources=expected,
      actual_codec_text_unchanged=True,host_runtime_query_binding=host,
      shared_translation=dict(K_bytes=[0,2048],old_V_bytes=[4096,8192],new_V_bytes=[2048,6144],half_open=True,
        V_tail_word_domain=1024,all_old_offsets_exactly_old_minus_new_512_words=True,
        tail_IR='or disjoint %1268,1024 -> add nuw nsw %1268,512; %1268 in 0..1023',
        scale_global_offset_bytes=4096,scale_register_reallocation='old s55=0x1000 -> new s59=0x1000',
        full_and_tail_global_page_stride_bytes=12544,V_external_offset_bytes_once=4352,SME_row_bytes=512,
        dead_BF16_K_helpers_absent_in_actual_IR=True),
      inherited_reference_S4_CFG=prior['inherited_S4_CFG'],inherited_CFG_original_line_numbers_are_7892=True,
      CFG_transfer_basis='Complete instruction order/opcodes/branches/dominators match 2ad; only 18 enumerated address/register operands differ; exact IR fullbody comparison admits only V shared relocation and proved tail OR-to-add.',
      rho_runningL_boundary='QK and P softmax/packing precede V wait; rho exp/running-L FMA may follow it. V wait dominates tail clear and PV.',
      proof_limits=['ELF static resources are not occupancy, measured residency, runtime attributes or speedup.',
        'Actual-SO IR is hash-bound ixobjdump export; wrapped/truncated long signatures are not repaired or claimed recompilable. Complete .args checks ABI.',
        'K/V shared address domains and actual instructions checked; 2KiB destination hardware SME acceptance and output numerics remain untested.',
        'Unused S2 compared but host scope remains S4 N48; no S2 admission.',
        'No SO dlopen, runtime dlsym, attributes/occupancy query, Torch import, compile, SSH or GPU executed by this reviewer.'],
      actual_static_review_passed=True,hardware_SME_alignment_verified=False,GPU_numeric_verified=False,Graph_verified=False,side_stream_verified=False,
      fullstep_verified=False,occupancy_verified=False,performance_verified=False,model_verified=False,service_verified=False,manual_Root_approval=False)
    a.report.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(report=str(a.report),sha256=sha(a.report.read_bytes()),instructions_bound=report['instructions_bound'],static_review=True,GPU_numeric_verified=False)))
if __name__=='__main__':main()
