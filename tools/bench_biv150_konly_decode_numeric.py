#!/usr/bin/env python3
"""Complete private step falsifiers/costs. Default CPU plan; explicit --run-gpu only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import socket
import subprocess
import sys

from bench_biv150_konly_decode import (BASE_HASHES,CASES,JIT,JIT_SHA,ROOT,atomic,module,preserve)
from biv150_konly_decode_factory import MAIN,NAME,build,validate_binding
from biv150_konly_decode_ir import audit
from biv150_konly_decode_numeric_contract import (append_bf16_reference,codec_vectors,reference_gates,timing_policy)
from biv150_konly_decode_numeric_oracle import (attention_references,encoded_cache_bytes,quantize_cpu)
from biv150_konly_decode_quant_reference import FORMAT,quantize_v2
from biv150_konly_decode_reference import append_reference,cost_plan,encode_reference,valid_slots
from biv150_konly_decode_sdk import INSTALLED_MATH,MATH_SHA,check_frontend
from biv150_konly_i8_qk_reference import bf16,f32_bits

DECODER_SOURCE_BOUNDARY='ae00f58629ad85850fec1b8b8d2da28757af5f8b'
GUARD=64
CACHE_GUARD=256
HARD_NAMES=('finite','baseline_reference','codec','cache_bytes','sources_unchanged',
            'scratch_canaries','graph_eager','side_eager','actual_ir')


def host_bits(tensor,torch):
    host=tensor.detach().cpu().contiguous()
    return host.view({1:torch.int8,2:torch.int16,4:torch.int32}[host.element_size()])


def host_bytes(tensor,torch):
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def exact_bytes(actual,expected,label,np):
    if actual!=expected:
        offsets=[]
        for start in range(0,len(expected),1048576):
            a,b=actual[start:start+1048576],expected[start:start+1048576]
            if a!=b:
                offsets=[start+int(i) for i in np.flatnonzero(np.frombuffer(a,np.uint8)!=np.frombuffer(b,np.uint8))[:16]]
                break
        raise AssertionError((label,'byte mismatch',offsets))
    return dict(bytes=len(actual),sha256=hashlib.sha256(actual).hexdigest(),exact=True)


def loaded(kernel,out,label):
    record,strings=preserve(kernel,out,label)
    assert all(kind in record['files'] and record['files'][kind]['bytes']>0 for kind in
               ('source','ttir','ttgir','llir','asm','cubin')),'incomplete actual compiler artifacts'
    record['phase']='post_launch'
    record['loaded_resources_known']=all(type(v) is int for v in
            (record['n_regs'],record['n_spills'],record['metadata']['shared']))
    return record,strings


def resource_comparison(a,b):
    result={}
    for name in ('n_regs','n_spills','shared','num_warps','num_stages','warp_size','use_sme'):
        values=[r[name] if name in ('n_regs','n_spills') else r['metadata'].get(name) for r in (a,b)]
        known=all(type(v) is int for v in values)
        result[name]='unknown' if not known else ('known_equal' if values[0]==values[1] else 'known_changed')
    return result


def cache(pages,torch):
    backing=torch.full((pages*16384+2*CACHE_GUARD,),90,device='cuda',dtype=torch.uint8)
    raw=backing[CACHE_GUARD:-CACHE_GUARD]
    code=torch.as_strided(raw.view(torch.int8),(pages,16,2,128),(16384,256,128,1))
    wide=raw.view(torch.bfloat16)
    value=torch.as_strided(wide,(pages,16,2,128),(8192,256,128,1),wide.storage_offset()+4096)
    packed=raw.view(torch.bfloat16).reshape(pages,2,16,2,128)
    key_bf16,value_bf16=packed.unbind(1)
    assert value.data_ptr()-raw.data_ptr()==8192
    return dict(backing=backing,raw=raw,code=code,scale=raw.view(torch.float32),value=value,
                packed=packed,key_bf16=key_bf16,value_bf16=value_bf16)


def cache_guard(c,torch):
    return bool((c['backing'][:CACHE_GUARD].cpu()==90).all() and
                (c['backing'][-CACHE_GUARD:].cpu()==90).all())


def output_buffers(batch,segments,torch):
    result={};backings={}
    for key,shape,dtype in [('scratch',(batch,16,16,128),torch.float32),('maxima',(batch,16,16),torch.float32),
                            ('sums',(batch,16,16),torch.float32),('out',(batch,16,128),torch.bfloat16)]:
        n=math.prod(shape);full=torch.empty(n+2*GUARD,device='cuda',dtype=dtype)
        full[:GUARD].fill_(42.5);full[-GUARD:].fill_(42.5)
        tensor=full[GUARD:-GUARD].reshape(shape)
        result[key]=tensor if key=='out' else tensor.narrow(2,0,segments)
        backings[key]=(full,tensor)
    return result,backings


def poison_outputs(backings):
    for _,tensor in backings.values(): tensor.fill_(math.nan)


def output_check(c,backings,lengths,segments,torch):
    bits={};finite=True
    for key,(full,tensor) in backings.items():
        guard=torch.full((GUARD,),42.5,dtype=full.dtype)
        assert torch.equal(host_bits(full[:GUARD],torch),host_bits(guard,torch))
        assert torch.equal(host_bits(full[-GUARD:],torch),host_bits(guard,torch))
        host=tensor.detach().cpu();bits[key]=host_bits(c[key],torch)
        if key=='out':
            finite=finite and bool(torch.isfinite(host).all())
            for token,n in enumerate(lengths):
                if n==0: assert bool((host[token]==0).all()),'empty output not exactly zero'
        else:
            for token,n in enumerate(lengths):
                steps=(n+segments*16-1)//(segments*16)
                for segment in range(16):
                    active=segment<segments and n>0 and segment*steps*16<n
                    row=host[token,:,segment]
                    if active: finite=finite and bool(torch.isfinite(row).all())
                    else: assert bool(torch.isnan(row).all()),'inactive/unused scratch overwritten'
    assert finite,'nonfinite valid scratch or output'
    return bits,c['out'].detach().cpu().clone()


def qtemps(batch,torch):
    code=torch.full((batch*2048+2*GUARD,),85,device='cuda',dtype=torch.int8)
    scale=torch.full((batch*16+2*GUARD,),-98765.5,device='cuda',dtype=torch.float32)
    return code,scale,code[GUARD:-GUARD].view(batch,16,128),scale[GUARD:-GUARD].view(batch,16)


def qtemp_check(backing_code,backing_scale,code,scale,expected,mode,torch):
    assert bool((backing_code[:GUARD].cpu()==85).all() and (backing_code[-GUARD:].cpu()==85).all())
    assert bool((backing_scale[:GUARD].cpu()==-98765.5).all() and (backing_scale[-GUARD:].cpu()==-98765.5).all())
    if mode=='preq':
        assert torch.equal(code.cpu(),expected[0]),'Qcode differs from CPU RN32 codec'
        assert torch.equal(host_bits(scale,torch),host_bits(expected[1],torch)),'Qscale bits differ from CPU RN32 codec'
    else:
        assert bool((code.cpu()==-128).all() and torch.isnan(scale.cpu()).all()),'inline unexpectedly writes Qtemp'


def poison_slots(base,encoded,slots,pages,torch,label):
    live=valid_slots(slots,pages)
    if not live: return
    pp=torch.tensor([s//16 for s in live],device='cuda',dtype=torch.int64)
    ss=torch.tensor([s%16 for s in live],device='cuda',dtype=torch.int64)
    if label=='baseline':
        base['key_bf16'][pp,ss]=math.nan;base['value_bf16'][pp,ss]=math.nan
    else:
        encoded['code'][pp,ss]=-128;encoded['value'][pp,ss]=math.nan
        scales=torch.as_strided(encoded['scale'],(pages,16,2),(4096,2,1),encoded['scale'].storage_offset()+1024)
        scales[pp,ss]=math.nan


def codec_gate(support,out,torch,np):
    rows=codec_vectors();host=torch.tensor(rows,dtype=torch.bfloat16)
    expected=[quantize_v2(row) for row in rows]
    qc=torch.tensor([r[0] for r in expected],dtype=torch.int8)
    qs=torch.tensor([r[1] for r in expected],dtype=torch.float32)
    fastc,fasts,fast=quantize_cpu(host,torch,np)
    assert torch.equal(fastc,qc) and torch.equal(host_bits(fasts,torch),host_bits(qs,torch)),'CPU fast oracle differs from rational RN32'
    path=ROOT/'tools/kernels/biv150_konly_codec_falsifier.py'
    probe=module(path,'cjj_konly_codec_direct_falsifier')
    source=host.cuda();records=[];n=len(rows)
    for m in (1,8):
        code_guard=1024 # Contains a whole accidentally unmasked M8 tail for canary detection.
        cb=torch.full((n*128+2*code_guard,),85,device='cuda',dtype=torch.int8)
        sb=torch.full((n+2*GUARD,),-98765.5,device='cuda',dtype=torch.float32)
        c,s=cb[code_guard:-code_guard].view(n,128),sb[GUARD:-GUARD]
        c.fill_(-128);s.fill_(math.nan)
        kernel=probe.probe_codec[((n+m-1)//m,)](source,c,s,n,m,num_warps=1,num_stages=1)
        torch.cuda.synchronize()
        record,_=loaded(kernel,out,'codec-M'+str(m));records.append(dict(m=m,resources=record,passed=False))
        witness=out/('codec-M'+str(m)+'-witness.npz')
        np.savez_compressed(witness,source_bits=host.view(torch.int16).numpy().view(np.uint16),
            expected_codes=qc.numpy(),expected_scale_bits=host_bits(qs,torch).numpy().view(np.uint32),
            actual_codes=c.cpu().numpy(),actual_scale_bits=host_bits(s,torch).numpy().view(np.uint32))
        records[-1]['witness']=dict(path=witness.name,bytes=witness.stat().st_size,sha256=hashlib.sha256(witness.read_bytes()).hexdigest())
        atomic(out/'codec-progress.json',dict(records=records,passed=False))
        assert torch.equal(c.cpu(),qc) and torch.equal(host_bits(s,torch),host_bits(qs,torch)),'device codec differs from CPU RN32->RNE target'
        assert bool((cb[:code_guard].cpu()==85).all() and (cb[-code_guard:].cpu()==85).all())
        assert bool((sb[:GUARD].cpu()==-98765.5).all() and (sb[-GUARD:].cpu()==-98765.5).all())
        assert torch.equal(host_bits(source,torch),host_bits(host,torch))
        records[-1]['passed']=True
        atomic(out/'codec-progress.json',dict(records=records,passed=all(r['passed'] for r in records)))
    return dict(passed=True,vectors=n,shapes=[1,8],all_codes_and_scale_bits_exact=True,
                source_unchanged=True,canaries=True,cpu_oracle=fast,resources=records)


def writer_gate(support,out,torch,np):
    from check_biv150_konly_i8_qk_cpu import fixture
    keys,values,_=fixture(2)
    keys[1][7][0][:3]=[240.,120.,-120.]
    values[0][0][0][:4]=[-0.,2.**-133,-2.**-133,struct_max_bf16()]
    kh=torch.tensor(keys,dtype=torch.bfloat16);vh=torch.tensor(values,dtype=torch.bfloat16)
    source=torch.empty((3,2,16,2,128),device='cuda',dtype=torch.bfloat16)
    sk,sv=source.unbind(1);sk.copy_(kh);sv.copy_(vh)
    frozen_source=host_bytes(source,torch)
    base,encoded=cache(3,torch),cache(3,torch)
    base['packed'].copy_(source);encoded['raw'].fill_(0xa5)
    kernel=support.encode_initial[(96,)](sk,sv,encoded['code'],encoded['scale'],encoded['value'],3,num_warps=1,num_stages=1)
    torch.cuda.synchronize()
    resources={};resources['initial_encoder'],_=loaded(kernel,out,'writer-edge-initial')
    initial=exact_bytes(host_bytes(encoded['raw'],torch),encode_reference(keys,values),'edge encoder including original V bits',np)
    nk=torch.tensor([keys[t%3][t%16] for t in range(8)],device='cuda',dtype=torch.bfloat16)
    nv=torch.tensor([values[t%3][t%16] for t in range(8)],device='cuda',dtype=torch.bfloat16)
    nk[4:]=math.nan;nv[4:]=math.nan # All corresponding slots are invalid and must not write.
    nk_cpu,nv_cpu=nk.cpu(),nv.cpu()
    slots=[0,15,16,47,-1,48,2**63-1,-2**63]
    mapping=torch.tensor(slots,device='cuda',dtype=torch.int64)
    checks=[]
    for label in ('baseline','candidate'):
        poison_slots(base,encoded,slots,3,torch,label)
        before=host_bytes(base['raw'] if label=='baseline' else encoded['raw'],torch)
        args=(base['key_bf16'],encoded['scale'],base['value_bf16']) if label=='baseline' else (encoded['code'],encoded['scale'],encoded['value'])
        kernel=support.append_kv[(8,2)](nk,nv,mapping,*args,48,label!='baseline',num_warps=1,num_stages=1)
        torch.cuda.synchronize();resources[label+'_append'],_=loaded(kernel,out,'writer-edge-'+label)
        reference=(append_bf16_reference if label=='baseline' else append_reference)(before,slots,nk_cpu.tolist(),nv_cpu.tolist())
        checks.append(exact_bytes(host_bytes(base['raw'] if label=='baseline' else encoded['raw'],torch),reference,label+' edge append',np))
        assert cache_guard(base,torch) and cache_guard(encoded,torch)
        assert torch.equal(host_bits(nk,torch),host_bits(nk_cpu,torch)) and torch.equal(host_bits(nv,torch),host_bits(nv_cpu,torch))
        assert mapping.cpu().tolist()==slots
    assert host_bytes(source,torch)==frozen_source
    return dict(passed=True,initial_encoder=initial,append_checks=checks,invalid_slots=[-1,48,2**63-1,-2**63],
                source_unchanged=True,invalid_source_tokens_poisoned=True,unmapped_and_reserved_exact=True,original_v_signedzero_subnormal_max_bits_exact=True,
                resources=resources)


def struct_max_bf16():
    import struct
    return struct.unpack('<f',struct.pack('<I',0x7f7f0000))[0]


def source_controls(q0,table0,batch,length,iteration,torch):
    rng=random.Random(20261027+batch+length+iteration)
    new_k=torch.tensor([[[bf16(rng.uniform(-3,3)) for _ in range(128)] for _ in range(2)] for _ in range(batch)],dtype=torch.bfloat16)
    new_v=torch.tensor([[[bf16(rng.uniform(-3,3)) for _ in range(128)] for _ in range(2)] for _ in range(batch)],dtype=torch.bfloat16)
    q=q0.clone();table=table0.clone();lengths=[length]*batch
    if iteration:
        q=q.roll(iteration,1)
        if iteration%2: q=-q
        table=table.roll(iteration,0)
        for i,n in enumerate((0,1,15,16,17,63,100,length-1)): lengths[i]=n
    if iteration==2: q.zero_()
    slots=[int(table[t,(n-1)//16])*16+(n-1)%16 if n else (-1 if iteration%2 else 2**63-1)
           for t,n in enumerate(lengths)]
    return dict(q=q,table=table,lengths=lengths,slots=slots,new_k=new_k,new_v=new_v)


def run_gpu(args):
    assert args.commit and args.out,'explicit --run-gpu requires frozen --commit and fresh --out'
    assert subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==args.commit
    assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
    subprocess.run(['git','-C',str(ROOT),'diff','--exit-code',MAIN,'--','vllm_fl','csrc'],check=True)
    assert hashlib.sha256(JIT.read_bytes()).hexdigest()==JIT_SHA and hashlib.sha256(INSTALLED_MATH.read_bytes()).hexdigest()==MATH_SHA
    assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True),'GPU busy; no queue'
    with socket.socket() as sock: assert sock.connect_ex(('127.0.0.1',9031))!=0,'service9031 active'
    args.out.mkdir(parents=True,exist_ok=False)
    result=dict(source_commit=args.commit,decoder_source_boundary=DECODER_SOURCE_BOUNDARY,
                completed=False,passed=False,component_only=True,private_bf16_append_control=True,
                official_service_writer_bound=False,precision_gate=False,service_verified=False,
                performance_verified=False,capacity_reclaimed=False,mode=args.mode,
                numeric_diagnostic=args.numeric_diagnostic,records=[],errors=[])
    def save(): atomic(args.out/'result.json',result)
    save()
    try:
        import os
        os.environ['TRITON_CACHE_DIR']=str(args.out/'triton-cache')
        import numpy as np
        import torch
        import triton
        from types import SimpleNamespace
        from biv150_benchmark_utils import inputs,main_call,reduce_call,graph_time
        torch.set_flush_denormal(False) # CPU oracle only; does not configure GPU FTZ.
        torch.set_num_threads(4)
        assert sys.byteorder=='little'
        target=triton.runtime.driver.active.get_current_target()
        assert (target.backend,target.arch,target.warp_size)==('corex',71,64)
        base=module(ROOT/'vllm_fl/patches/biv150_gqa_decode.py','cjj_konly_numeric_main')
        support=module(ROOT/'tools/kernels/biv150_konly_decode_support.py','cjj_konly_decode_support')
        assert base.TARGET_PROGRAMS==256
        variants={}
        for mode in ('inline','preq'):
            generated,proof=build((ROOT/'vllm_fl/patches/biv150_gqa_decode.py').read_text(),mode)
            path=args.out/('generated-'+mode+'.py');path.write_text(generated)
            variants[mode]=(module(path,'cjj_konly_numeric_'+mode),generated,proof)
        (args.out/'support-source.py.txt').write_bytes((ROOT/'tools/kernels/biv150_konly_decode_support.py').read_bytes())
        result['frontend']=check_frontend(ROOT)
        result['gpu_work_attempted']=True
        result['stage']='codec_exact'
        result['codec']=codec_gate(support,args.out,torch,np);save()
        result['stage']='writer_edge_bytes'
        result['writer_edges']=writer_gate(support,args.out,torch,np);save()
        if args.codec_only:
            result.update(completed=True,passed=True,status='codec_and_writer_only',
                          full_decoder_numeric_verified=False,micro_cost_measured=False)
            save();return result
        for name in args.case or ['4k','16k']:
            batch,length=CASES[name];segments=base._pick_segments(batch,2,16)
            record=dict(case=name,batch=batch,kv=length,segments=segments,passed=False,timing_attempted=False,
                        numeric_checks=[],graph_checks=[],resources={},hard_checks={k:False for k in HARD_NAMES})
            result['stage']=name+'/encoder_exact'
            result['records'].append(record);save()
            c=inputs(batch,length)
            pages=c['k'].shape[0]
            cold=torch.empty((pages,2,16,2,128),device='cuda',dtype=torch.bfloat16)
            cold_k,cold_v=cold.unbind(1);cold_k.copy_(c['k']);cold_v.copy_(c['v'])
            frozen_k,frozen_v=cold_k.cpu(),cold_v.cpu()
            cold_fingerprint=hashlib.sha256(host_bytes(cold,torch)).hexdigest()
            record['original_inputs']=dict(seed=20261023+batch+length,cold_packed_bf16_sha256=cold_fingerprint,
                input_factory_sha256=hashlib.sha256((ROOT/'tools/biv150_benchmark_utils.py').read_bytes()).hexdigest())
            c['k'],c['v']=cold_k,cold_v
            expected,cold_codes,cold_scales,oracle=encoded_cache_bytes(frozen_k,frozen_v,torch,np)
            record['cpu_encoder_oracle']=oracle;save()
            q0=c['q'].cpu();table0=c['table'].cpu()
            baseline_cache,encoded=cache(pages,torch),cache(pages,torch)
            encoded['raw'].fill_(0xa5)
            init=support.encode_initial[(pages*32,)](cold_k,cold_v,encoded['code'],encoded['scale'],encoded['value'],pages,num_warps=1,num_stages=1)
            torch.cuda.synchronize()
            record['resources']['initial_encoder'],_=loaded(init,args.out,name+'-initial-encoder')
            save()
            record['initial_encoder_exact']=exact_bytes(host_bytes(encoded['raw'],torch),expected,'initial encoder',np)
            assert cache_guard(encoded,torch)
            q=c['q'];table=torch.full((batch,8192),-1,device='cuda',dtype=torch.int32)[:,:table0.shape[1]]
            lengths=torch.empty(batch,device='cuda',dtype=torch.int32);slots=torch.empty(batch,device='cuda',dtype=torch.int64)
            nk=torch.empty((batch,2,128),device='cuda',dtype=torch.bfloat16);nv=torch.empty_like(nk)
            qb,sb,qcode,qscale=qtemps(batch,torch)
            cases={};backs={}
            for label in ('baseline','inline','preq'):
                io,back=output_buffers(batch,segments,torch)
                cases[label]=dict(q=q,k=baseline_cache['key_bf16'] if label=='baseline' else encoded['code'],
                                 v=baseline_cache['value_bf16'] if label=='baseline' else encoded['value'],
                                 table=table,lengths=lengths,**io)
                backs[label]=back
            validate_binding(mode=args.mode,k_strides=encoded['code'].stride(),v_strides=encoded['value'].stride(),
                             scale_strides=(4096,2,1),q_strides=q.stride(),source_storage=cold.untyped_storage().data_ptr(),
                             destination_storage=encoded['raw'].untyped_storage().data_ptr(),format_id=FORMAT)
            proxies={'baseline':base}
            for mode,(own,_,_) in variants.items():
                class Producer:
                    def __init__(self,kernel): self.kernel=kernel
                    def __getitem__(self,grid):
                        def launch(**kw): return self.kernel[grid](**kw,query_code_ptr=qcode,query_scale_ptr=qscale,key_scale_ptr=encoded['scale'])
                        return launch
                proxies[mode]=SimpleNamespace(LOG2E=base.LOG2E,_gqa_decode_attention=Producer(getattr(own,NAME)))
            compiled={}
            def step(label):
                if label=='preq': compiled['qprep']=support.prepare_q[(batch,2)](q,qcode,qscale,batch,num_warps=1,num_stages=1)
                cache_args=(baseline_cache['key_bf16'],encoded['scale'],baseline_cache['value_bf16']) if label=='baseline' else (encoded['code'],encoded['scale'],encoded['value'])
                compiled[label+'-append']=support.append_kv[(batch,2)](nk,nv,slots,*cache_args,pages*16,label!='baseline',num_warps=1,num_stages=1)
                compiled[label+'-producer']=main_call(proxies[label],cases[label],segments)
                compiled[label+'-reducer']=reduce_call(base,cases[label],segments,4)
            def controls_load(control):
                q.copy_(control['q']);table.copy_(control['table']);lengths.copy_(torch.tensor(control['lengths'],dtype=torch.int32))
                slots.copy_(torch.tensor(control['slots'],dtype=torch.int64));nk.copy_(control['new_k']);nv.copy_(control['new_v'])
            def poison(label,control):
                poison_outputs(backs[label]);qcode.fill_(-128);qscale.fill_(math.nan)
                # Re-poison append destinations before every eager/Graph/side step.
                poison_slots(baseline_cache,encoded,control['slots'],pages,torch,label)
            def verify(label,control,qexpected):
                torch.cuda.synchronize()
                before=record['_before_base'] if label=='baseline' else record['_before_encoded']
                expected_append=(append_bf16_reference if label=='baseline' else append_reference)(
                    before,control['slots'],control['new_k'].tolist(),control['new_v'].tolist())
                actual=host_bytes(baseline_cache['raw'] if label=='baseline' else encoded['raw'],torch)
                exact_bytes(actual,expected_append,label+' complete cache after append/consume',np)
                assert cache_guard(baseline_cache,torch) and cache_guard(encoded,torch)
                assert torch.equal(host_bits(q,torch),host_bits(control['q'],torch))
                assert torch.equal(host_bits(nk,torch),host_bits(control['new_k'],torch)) and torch.equal(host_bits(nv,torch),host_bits(control['new_v'],torch))
                assert torch.equal(table.cpu(),control['table']) and lengths.cpu().tolist()==control['lengths'] and slots.cpu().tolist()==control['slots']
                if label!='baseline': qtemp_check(qb,sb,qcode,qscale,qexpected,label,torch)
                return output_check(cases[label],backs[label],control['lengths'],segments,torch)
            def before(label):
                # Keep large byte oracles out of JSON; ephemeral only.
                record['_before_base' if label=='baseline' else '_before_encoded']=host_bytes(
                    baseline_cache['raw'] if label=='baseline' else encoded['raw'],torch)
            def drop_before():
                record.pop('_before_base',None);record.pop('_before_encoded',None)
            controls=[];answers=[];eagers=[]
            selected=[0,1,batch-1]
            for iteration in range(3):
                result['stage']=name+'/eager-'+str(iteration)
                control=source_controls(q0,table0,batch,length,iteration,torch);controls.append(control)
                baseline_cache['packed'].copy_(cold)
                support.encode_initial[(pages*32,)](cold_k,cold_v,encoded['code'],encoded['scale'],encoded['value'],pages,num_warps=1,num_stages=1)
                controls_load(control)
                selected=([0,1,batch-1] if iteration==0 else list(range(min(8,batch)))+[batch-1])
                reference,qexpected=attention_references(frozen_k,frozen_v,cold_codes,cold_scales,control,selected,torch,np)
                answers.append((reference,qexpected));snapshots={};outputs={}
                for label in ('baseline','inline','preq'):
                    poison(label,control);before(label);step(label)
                    snapshots[label],outputs[label]=verify(label,control,qexpected);drop_before()
                checks={label:[dict(row=r['row'],length=r['length'],integer_bound=r['integer_bound'],
                            **reference_gates(outputs['baseline'][r['row']].float().reshape(-1).tolist(),
                                              outputs[label][r['row']].float().reshape(-1).tolist(),
                                              r['original'].reshape(-1).tolist(),r['quantized'].reshape(-1).tolist()))
                               for r in reference] for label in ('inline','preq')}
                assert all(x['baseline_hard_passed'] for rows in checks.values() for x in rows),'baseline independent reference failed'
                witness=args.out/(name+'-numeric-witness-'+str(iteration)+'.npz')
                np.savez_compressed(witness,
                    q_bits=control['q'].contiguous().view(torch.int16).numpy().view(np.uint16),
                    new_k_bits=control['new_k'].contiguous().view(torch.int16).numpy().view(np.uint16),
                    new_v_bits=control['new_v'].contiguous().view(torch.int16).numpy().view(np.uint16),
                    table=control['table'].numpy(),lengths=np.array(control['lengths'],dtype=np.int32),
                    slots=np.array(control['slots'],dtype=np.int64),selected_rows=np.array(selected,dtype=np.int32),
                    original_reference=np.stack([r['original'].numpy() for r in reference]),
                    quantized_reference=np.stack([r['quantized'].numpy() for r in reference]),
                    baseline_output_bits=outputs['baseline'].view(torch.int16).numpy().view(np.uint16),
                    inline_output_bits=outputs['inline'].view(torch.int16).numpy().view(np.uint16),
                    preq_output_bits=outputs['preq'].view(torch.int16).numpy().view(np.uint16))
                record['numeric_checks'].append(dict(iteration=iteration,checks=checks,
                     witness=dict(path=witness.name,bytes=witness.stat().st_size,sha256=hashlib.sha256(witness.read_bytes()).hexdigest()),
                     inline_preq_bits_equal={k:torch.equal(snapshots['inline'][k],snapshots['preq'][k]) for k in snapshots['inline']}))
                eagers.append(snapshots);save()
            for label,kernel in compiled.items():
                rec,strings=loaded(kernel,args.out,name+'-'+label)
                if label=='baseline-producer': assert str(kernel.hash)==BASE_HASHES[batch],'current main control hash changed'
                if label.endswith('producer'):
                    assert kernel.metadata.num_warps==1 and kernel.metadata.num_stages==1
                    if not label.startswith('baseline'):
                        mode=label.split('-')[0];rec['mechanism']=audit(strings,variants[mode][1]);assert rec['mechanism']['passed'],'actual full decoder IR gate failed'
                if label.endswith('reducer'): assert kernel.metadata.num_warps==4
                record['resources'][label]=rec
            record['resource_comparison']={mode:resource_comparison(record['resources']['baseline-producer'],record['resources'][mode+'-producer']) for mode in ('inline','preq')}
            record['hard_checks'].update(finite=True,baseline_reference=True,codec=True,cache_bytes=True,
                sources_unchanged=True,scratch_canaries=True,actual_ir=True)
            save()
            result['stage']=name+'/changed_graph'
            graphs={}
            baseline_cache['packed'].copy_(cold)
            support.encode_initial[(pages*32,)](cold_k,cold_v,encoded['code'],encoded['scale'],encoded['value'],pages,num_warps=1,num_stages=1)
            controls_load(controls[0])
            for label in ('baseline','inline','preq'):
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): step(label)
                graphs[label]=graph
            for iteration in (1,2,0,2,1):
                control=controls[iteration]
                baseline_cache['packed'].copy_(cold)
                support.encode_initial[(pages*32,)](cold_k,cold_v,encoded['code'],encoded['scale'],encoded['value'],pages,num_warps=1,num_stages=1)
                controls_load(control);qexpected=answers[iteration][1]
                for label in ('baseline','inline','preq'):
                    poison(label,control);before(label);graphs[label].replay()
                    actual,_=verify(label,control,qexpected);drop_before()
                    assert all(torch.equal(actual[k],eagers[iteration][label][k]) for k in actual),'changed-input poisoned Graph differs from eager'
                record['graph_checks'].append(dict(iteration=iteration,changed_q_k_v_slot_table_lengths=True,
                                                   scratch_cache_qtemp_poison=True,exact=True));save()
            result['stage']=name+'/side_stream'
            side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
            control=controls[1];qexpected=answers[1][1]
            with torch.cuda.stream(side):
                baseline_cache['packed'].copy_(cold)
                support.encode_initial[(pages*32,)](cold_k,cold_v,encoded['code'],encoded['scale'],encoded['value'],pages,num_warps=1,num_stages=1)
                controls_load(control)
                for label in ('baseline','inline','preq'):
                    poison(label,control);before(label);step(label)
                    side.synchronize();actual,_=verify(label,control,qexpected);drop_before()
                    assert all(torch.equal(actual[k],eagers[1][label][k]) for k in actual),'side stream differs from same-input eager'
            torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize()
            assert hashlib.sha256(host_bytes(cold,torch)).hexdigest()==cold_fingerprint,'initial BF16 source cache was modified'
            record['hard_checks'].update(graph_eager=True,side_eager=True)
            record['numeric_passed']={label:all(c['candidate_numeric_passed'] for item in record['numeric_checks'] for c in item['checks'][label]) for label in ('inline','preq')}
            record['timing_policy']=timing_policy(record['hard_checks'],record['numeric_passed'][args.mode],args.numeric_diagnostic)
            record['passed']=record['numeric_passed'][args.mode]
            save()
            if record['timing_policy']['allowed']:
                result['stage']=name+'/complete_cost'
                # Restore original inputs. The writer is inside every measured closure.
                controls_load(controls[0]);baseline_cache['packed'].copy_(cold)
                support.encode_initial[(pages*32,)](cold_k,cold_v,encoded['code'],encoded['scale'],encoded['value'],pages,num_warps=1,num_stages=1)
                record['timing_attempted']=True;record['timings']=[]
                for iteration in range(record['timing_policy']['timing_rounds']):
                    order=['baseline',args.mode] if iteration%2==0 else [args.mode,'baseline']
                    timing={}
                    for label in order:
                        median,low,high=graph_time(lambda label=label:step(label))
                        timing[label]=dict(complete_step_ms=median,min_ms=low,max_ms=high)
                    timing['baseline_over_candidate']=timing['baseline']['complete_step_ms']/timing[args.mode]['complete_step_ms']
                    record['timings'].append(timing);save()
                record['stage_ms']={}
                functions={
                    'baseline_append':lambda:support.append_kv[(batch,2)](nk,nv,slots,baseline_cache['key_bf16'],encoded['scale'],baseline_cache['value_bf16'],pages*16,False,num_warps=1,num_stages=1),
                    'candidate_append':lambda:support.append_kv[(batch,2)](nk,nv,slots,encoded['code'],encoded['scale'],encoded['value'],pages*16,True,num_warps=1,num_stages=1),
                    'baseline_producer':lambda:main_call(base,cases['baseline'],segments),
                    'candidate_producer':lambda:main_call(proxies[args.mode],cases[args.mode],segments),
                    'original_reducer':lambda:reduce_call(base,cases[args.mode],segments,4),
                    'initial_encoder_separate':lambda:support.encode_initial[(pages*32,)](cold_k,cold_v,encoded['code'],encoded['scale'],encoded['value'],pages,num_warps=1,num_stages=1)}
                if args.mode=='preq': functions['qprep']=lambda:support.prepare_q[(batch,2)](q,qcode,qscale,batch,num_warps=1,num_stages=1)
                for label,fn in functions.items(): record['stage_ms'][label]=graph_time(fn)[0]
                record['cost_plan']=cost_plan(args.mode);record['stage_sums_are_not_complete_step']=True
                record['initial_encoder_is_not_service_free']=True
                poison(args.mode,controls[0]);before(args.mode);step(args.mode)
                post,_=verify(args.mode,controls[0],answers[0][1]);drop_before()
                assert all(torch.equal(post[k],eagers[0][args.mode][k]) for k in post),'post-timing step differs from original eager'
                assert hashlib.sha256(host_bytes(cold,torch)).hexdigest()==cold_fingerprint
                record['post_timing_poisoned_step_exact']=True
            save()
        result['stage']='completed'
        result['completed']=True;result['passed']=all(r['passed'] for r in result['records'])
        result['micro_cost_measured']=any(r['timing_attempted'] for r in result['records'])
        result['performance_verified']=False # micro cost evidence is not a service acceptance
        save();return result
    except Exception as error:
        for record in result['records']:
            record.pop('_before_base',None);record.pop('_before_encoded',None)
        import traceback
        (args.out/'failure-traceback.txt').write_text(traceback.format_exc())
        result['errors'].append(dict(stage=result.get('stage'),type=type(error).__name__,message=str(error)))
        save();raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-gpu',action='store_true');parser.add_argument('--commit');parser.add_argument('--out',type=Path)
    parser.add_argument('--case',action='append',choices=CASES);parser.add_argument('--mode',choices=('inline','preq'),default='inline')
    parser.add_argument('--numeric-diagnostic',action='store_true')
    parser.add_argument('--codec-only',action='store_true',help='Same prefix hard gates only; no GQA or timing')
    args=parser.parse_args()
    result=run_gpu(args) if args.run_gpu else dict(cpu_only=True,decoder_source_boundary=DECODER_SOURCE_BOUNDARY,
        cost_plan=cost_plan(args.mode),cases=args.case or ['4k','16k'],complete_numeric_runner_prepared=True,
        component_device_validated=False,codec_only=args.codec_only,precision_gate=False,gpu_executed=False,
        numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules,triton_imported='triton' in sys.modules)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
