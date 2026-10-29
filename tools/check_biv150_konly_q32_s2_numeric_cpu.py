#!/usr/bin/env python3
"""Narrow stdlib source/tail/policy checks, plus optional actual-archive replay."""
import argparse
import ast
import copy
import json
from pathlib import Path
import sys
import struct
import tempfile
import types
sys.dont_write_bytecode=True
import biv150_konly_q32_s2_numeric_source as s
import biv150_konly_q32_s2_numeric_contract as f


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--s1-raw',type=Path);p.add_argument('--s2-raw',type=Path);p.add_argument('--static-review',type=Path);a=p.parse_args()
    checks=[]
    def check(value,label):s.need(value,label);checks.append(label)
    def reject(fn,label):
        try:fn()
        except (ValueError,AssertionError,FileNotFoundError,KeyError):checks.append(label);return
        raise ValueError('mutation accepted: '+label)
    plan=f.plan();driver=s.run_source();tree=ast.parse(driver)
    check(plan['stages']=={'s1':1,'s2':2} and plan['num_seqs']==1,'both actual N1 Q32 roles')
    check(not any(x for x in ('elapsed_time','perf_counter','Event(','performance_measured=True') if x in driver),'numeric only: timing and timing warmup removed')
    check("graph_b={mode:stage('Graph-B-first'" in driver and driver.index("graph_b={mode:stage('Graph-B-first'")<driver.index("stage('eager-B-after-Graph'"),'Graph B before fresh eager B unchanged')
    check("c.need(readonly==immutable and byte_cpu(cache_after)==encoded" in driver and "byte_cpu(outputs['s1'])==byte_cpu(outputs['s2'])" in driver,'full cache/readonly/bitwise checks retained')
    check("references(cpu,codes,scales,torch,np)" in driver and "source_cpu[name][:,:256].copy_(cpu['packed'][physical,index,cols]" in driver,'independent original CPU cache/writer source')
    check('actual cache miss' not in driver and 'reviewed_cache.warmup' in driver,'original miss-prohibiting warmup path retained')
    for pattern,length in f.LENGTHS.items():
        prefix=length-2048;visible=[prefix+i+1 for i in range(2048)];positions=list(range(prefix,length))
        check(min(visible)>0 and max(visible)==length and len(positions)==len(set(positions))==2048,'valid causal/write range pattern'+str(pattern))
        check((length+15)//16==384 and 2*((length+15)//16)+5==773,'unchanged physical descriptor pattern'+str(pattern))
    check(f.LENGTHS[1]%16==15 and f.LENGTHS[1]%64==63 and 6143 not in range(4095,6143),'tail invalid lane is outside writer slots')
    d=s.dependencies()
    with tempfile.TemporaryDirectory(prefix='cjj-q32-s2-numeric-cpu-') as tmp:
        folder=Path(tmp)/'references';f.export_references(folder)
        sources,_=d.contract.reference_sources(folder);text=sources['biv150_konly_q32_numeric_contract.py'];expanded=f.reference_source(text)
        check("<(length-2048)+torch.tensor(rows)[:,None]+1" in expanded and 'only Q2048/KV6143 or6144' in expanded,'reference length and causal prefix parameterized together')
        old=ast.parse(text);new=ast.parse(expanded)
        unaffected=lambda t:{n.name:ast.dump(n) for n in t.body if isinstance(n,ast.FunctionDef) and n.name!='references'}
        check(unaffected(old)==unaffected(new),'codec/metrics/gates/policy full AST unchanged')
        ref,policy,_=f.load_reference(folder)
        hard={k:True for k in ('finite','codec','output_canaries','read_only_inputs','actual_IR','hoisted_unhoisted_bitwise','quantized_reference')}
        check(policy(hard,False,True)['original_BF16_quality_passed'] is False,'diagnostic retains original BF16 failure')
        reject(lambda:policy(hard,False,False),'strict default rejects BF16 quality failure')
        for key in hard:
            reject(lambda k=key:policy(dict(hard,**{k:False}),False,True),'diagnostic cannot bypass '+key)
        # Snapshot recorder scopes stay separate; no production widening.
        check(f.snapshot_module('s1') is not None and f.snapshot_module('s2') is not None,'original S1/private S2 recorders load without framework')
    actual=None
    if any((a.s1_raw,a.s2_raw,a.static_review)):
        s.need(all((a.s1_raw,a.s2_raw,a.static_review)),'all actual inputs required')
        pre=f.preflight(a.s1_raw,a.s2_raw,a.static_review);check(True,'all actual3219+27 files SHA and S1 native replay')
        # Reuse the existing pure ELF/msgpack decoder definitions, not its
        # native-extension assumptions or admission. This helper is CPU only.
        decoder_commit='6b33cec889d9462dc3f167eff4be42a3c5108713'
        body=s.git('show',decoder_commit+':tools/audit_biv150_native_k8_compile_cpu.py')
        check(s.sha(body)=='09ac1c98f2f1a1ff25ad02fcc819b25fc00bdb442a55c2ec1ad69defc61ea6e7','immutable existing ELF decoder')
        ns={'struct':struct};defs=[n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef) and n.name in ('elf','msgpack')]
        check(len(defs)==2,'only two CPU decoder definitions reused')
        exec(compile(ast.Module(body=defs,type_ignores=[]),'unchanged-native96-ELF-decoder','exec'),ns)
        binary={}
        for role,raw in (('s1',a.s1_raw),('s2',a.s2_raw)):
            cache=raw/'out/triton-cache'/pre['cache'][role]['directory'];blob=(cache/'kernel_cjj_konly_q32.cubin').read_bytes()
            elf=ns['elf'](blob);check(elf['machine']==248 and elf['extent']==len(blob),'actual complete GPU ELF '+role)
            note=elf['sections']['.note']['bytes'];at=0;metadata=None
            while at<len(note):
                n,d,k=struct.unpack_from('<III',note,at);at+=12;name=note[at:at+n].rstrip(b'\0');at=(at+n+3)//4*4;desc=note[at:at+d];at=(at+d+3)//4*4
                if name==b'Iluvatar' and k==10:
                    check(metadata is None,'unique actual GPU note '+role);metadata=ns['msgpack'](desc)
            check(len(metadata['iluvatar.kernels'])==1,'one actual N1 kernel '+role);entry=metadata['iluvatar.kernels'][0]
            check(entry['.name']=='kernel_cjj_konly_q32' and entry['.vgpr_count']==f.RESOURCES[role]['n_regs'] and entry['.sgpr_spill_count']==f.RESOURCES[role]['ELF_sgpr_spills'] and entry['.vgpr_spill_count']==entry['.private_segment_fixed_size']==entry['.group_segment_fixed_size']==0,'actual ELF resources '+role)
            binary[role]=dict(metadata=entry,text_sha256=s.sha(elf['sections']['.text']['bytes']))
        check(binary['s1']['metadata']['.args']==binary['s2']['metadata']['.args'] and binary['s1']['metadata']['.kernarg_segment_size']==binary['s2']['metadata']['.kernarg_segment_size']==216,'full actual ELF typed argument descriptors equal')
        for role,key,value in (('s2','ABI',{}),('s1','hash',f.HASHES['s2'])):
            records=copy.deepcopy(pre['records']);records[role][key]=value
            reject(lambda:f.check_pair(records,pre['metadata']),'reject role/typed ABI mutation '+role+'/'+key)
        for key,value in (('use_sme',0),('num_stages',True),('shared',None)):
            meta=copy.deepcopy(pre['metadata']);meta['s1'][key]=value
            reject(lambda:f.check_pair(pre['records'],meta),'reject metadata mutation '+key)
        with tempfile.TemporaryDirectory(prefix='cjj-q32-root-admission-negative-') as tmp:
            path=Path(tmp)/'pending.json';args=types.SimpleNamespace(commit='f'*40,preflight_sha256='e'*64,admission=path)
            receipt=f.admission_template(args.commit,args.preflight_sha256)
            for key,value in (('Root_numeric_approved',False),('Root_numeric_approved',1),('num_seqs',2),('timing_allowed',True)):
                body=dict(receipt,Root_numeric_approved=True);body[key]=value
                path.write_text(json.dumps(body));args.admission_sha256=s.sha(path.read_bytes())
                reject(lambda:f.admit(args,pre),'Root receipt rejects '+key+'/'+str(value))
        actual=dict(provenance=pre['provenance'],complete_typed_ABI_equal=True,
            metadata_difference=['hash','ir_override','num_stages','shared'],
            baseline_native_audit_replayed=True,hashes=f.HASHES,cubins=f.CUBINS,
            original_proposal_and_numeric_pending_labels_preserved=True,
            ELF_decoder_commit=decoder_commit,actual_ELF=binary,
            prepared_proof_sha256=s.sha(json.dumps(pre,sort_keys=True,indent=2,allow_nan=False)+'\n'))
    check(not {'torch','triton','numpy','vllm','vllm_fl'}.intersection(sys.modules),'stdlib only/no framework')
    print(json.dumps(dict(schema='cjj-q32-s2-numeric-cpu-v1',checks=checks,count=len(checks),
        driver_sha256=s.sha(driver),plan_sha256=s.sha(json.dumps(plan,sort_keys=True,indent=2,allow_nan=False)+'\n'),
        actual=actual,GPU_executed=False,numeric_verified=False,Graph_verified=False,precision_gate=False,
        performance_measured=False,service_verified=False),sort_keys=True,indent=2,allow_nan=False))


if __name__=='__main__':main()
