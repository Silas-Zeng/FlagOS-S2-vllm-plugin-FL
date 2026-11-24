#!/usr/bin/env python3
"""Default stdlib plan; explicit mixed-reader numeric falsifier, no service/timing."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

from bench_biv150_konly_mixed_decode import (ROOT, CASES, atomic, module, pick, preflight,
    preserve, producer_arguments, reducer_arguments)
from biv150_konly_mixed_factory import MAIN, NAMES, build
from biv150_konly_mixed_ir import mechanism, typed_abi
from biv150_konly_mixed_reference import cumulative, metadata, writer_partition, validate_binding, FORMAT, STOCK
from biv150_konly_decode_numeric_contract import reference_gates, metrics


def control_metadata(base, kv, iteration):
    """CPU scheduling fixtures; total query count/capacity stay fixed for Graph."""
    n=len(base);total=sum(base)
    if iteration==0:
        qlens=list(base)
    elif iteration==1:
        qlens=sorted(base,reverse=True)
    elif iteration==2:
        ordered=sorted(base,reverse=True);large=ordered.pop(0);qlens=ordered[:n//2]+[large]+ordered[n//2:]
    elif iteration==3:
        qlens=[0]*(n-1)+[total]
    else:
        raise ValueError('Unknown control')
    cu=cumulative(qlens);lengths=[]
    decode_rank=0
    for seq,qlen in enumerate(qlens):
        length=kv if qlen!=1 or iteration!=1 else (0,15,1,16,17,63,100,kv-1)[decode_rank%8]
        lengths.append(length)
        if qlen==1:decode_rank+=1
    metadata(cu,lengths,total+3)
    partition=writer_partition(cu,lengths,total+3,True)
    return dict(iteration=iteration,query_lengths=qlens,cu_q=cu,lengths=lengths,
        decode_sequences=[i for i,q in enumerate(qlens) if q==1],partition=partition,
        zero_q=iteration==2,table_roll=iteration,query_capacity=total+3)


def bits(tensor,torch):
    host=tensor.detach().cpu().contiguous()
    return host.view({1:torch.int8,2:torch.int16,4:torch.int32}[host.element_size()])


def raw(tensor,torch):
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def buffers(n,capacity,segments,torch):
    io={};backs={}
    for name,shape,dtype in [('scratch',(n,16,16,128),torch.float32),('maxima',(n,16,16),torch.float32),
                            ('sums',(n,16,16),torch.float32),('out',(capacity,16,128),torch.bfloat16)]:
        full=torch.empty(math.prod(shape)+128,device='cuda',dtype=dtype)
        full[:64].fill_(42.5);full[-64:].fill_(42.5);view=full[64:-64].reshape(shape)
        backs[name]=(full,view);io[name]=view if name=='out' else view.narrow(2,0,segments)
    return io,backs


def poison(backs,c):
    for name,(_,view) in backs.items():
        view.fill_(42.5 if name=='out' else math.nan)
    for seq in c['decode_sequences']:
        backs['out'][1][c['cu_q'][seq]].fill_(math.nan)


def output_check(io,backs,c,segments,torch):
    snapshots={}
    for name,(full,view) in backs.items():
        sentinel=torch.full((64,),42.5,dtype=full.dtype)
        assert torch.equal(bits(full[:64],torch),bits(sentinel,torch)) and torch.equal(bits(full[-64:],torch),bits(sentinel,torch))
        host=view.detach().cpu()
        if name=='out':
            decoded=set(c['partition']['decode_tokens'])
            for token in range(host.shape[0]):
                if token in decoded:
                    assert bool(torch.isfinite(host[token]).all()),'Decode output not overwritten/finiteness failed'
                else:
                    assert torch.equal(bits(host[token],torch),bits(torch.full_like(host[token],42.5),torch)),'Prefill/padded output overwritten'
            for seq in c['decode_sequences']:
                if c['lengths'][seq]==0:
                    assert bool((host[c['cu_q'][seq]]==0).all()),'Empty decode not zero'
        else:
            for seq,length in enumerate(c['lengths']):
                span=max(1,(length+segments*16-1)//(segments*16))
                active=(length+span*16-1)//(span*16) if seq in c['decode_sequences'] else 0
                for segment in range(16):
                    value=host[seq,:,segment]
                    if segment<active:
                        assert bool(torch.isfinite(value).all()),'Active compact partial not overwritten'
                    else:
                        assert bool(torch.isnan(value).all()),'Prefill/inactive/unused compact partial overwritten'
        snapshots[name]=bits(view,torch).clone()
    return snapshots,io['out'].detach().cpu().clone()


def run_device(args,sources,proof):
    preflight(args)
    result=dict(source_commit=args.commit,main=MAIN,case=args.case,arm=args.arm,source_proof=proof,
        completed=False,passed=False,compile_passed=False,gpu_executed=False,gpu_numeric_verified=False,
        gpu_implementation_numeric_verified=False,implementation_hard_gates_passed=False,
        original_bf16_proxy_passed=False,precision_gate=False,service_installed=False,full_service_verified=False,
        timing_attempted=False,capacity_reclaimed=False,controls=[],resources_pre_launch={},resources={},errors=[])
    def save():atomic(args.out/'result.json',result)
    save()
    try:
        import os
        os.environ['TRITON_CACHE_DIR']=str(args.out/'triton-cache')
        import numpy as np
        import torch
        import triton
        from biv150_konly_decode_numeric_oracle import quantize_cpu,pack_arrays,attention_references
        import biv150_prefill_probe_fixtures as fixtures
        torch.set_num_threads(4);torch.set_flush_denormal(False)
        target=triton.runtime.driver.active.get_current_target()
        assert (target.backend,target.arch,target.warp_size)==('corex',71,64)
        result['target']=[target.backend,target.arch,target.warp_size]
        qlens,kv=CASES[args.case];n=len(qlens);total=sum(qlens);capacity=total+3
        segments=pick((ROOT/'vllm_fl/patches/biv150_gqa_decode.py').read_text(),n)
        modes=('bf16','k8') if args.arm=='both' else (args.arm,)
        assert 'bf16' in modes,'Original BF16 control required for unchanged numeric gate'
        fixtures.torch=torch;seed=20261027
        cpu=fixtures.cpu_inputs(qlens,[kv]*n,seed,1)
        pages=cpu['packed'].shape[0]
        result.update(num_seqs=n,total_queries=total,query_capacity=capacity,segments=segments,input_seed=seed,
            input_factory='main biv150_prefill_probe_fixtures.cpu_inputs(pattern1)',packed_pages=pages,
            physical_page_bytes=16384,prefill_table_rows_poisoned_in_this_isolated_decode_reader=True)
        cold_k,cold_v=cpu['packed'].unbind(1)
        codes,scales,codec_info=quantize_cpu(cold_k,torch,np);result['cpu_codec']=codec_info
        source= torch.full((pages*16384+512,),90,device='cuda',dtype=torch.uint8)
        stock=source[256:-256].view(torch.bfloat16).reshape(pages,2,16,2,128);stock.copy_(cpu['packed'])
        key,value=stock.unbind(1)
        packed_back=torch.full_like(source,165)
        assert source.untyped_storage().data_ptr()!=packed_back.untyped_storage().data_ptr(),'Initial encoder source/destination alias'
        packed=packed_back[256:-256].reshape(pages,16384)
        packed_value=packed.view(torch.bfloat16).reshape(pages,2,16,2,128)[:,1]
        q_back=torch.full((capacity,20,128),37.,device='cuda',dtype=torch.bfloat16)
        q=q_back[:,:16]  # Actual fused-QKV-style token stride2560, heads128/dim1.
        q0=torch.full((capacity,16,128),-31.,dtype=torch.bfloat16);q0[:total].copy_(cpu['q'])
        table_back=torch.full((n,8192),-2,device='cuda',dtype=torch.int32);table=table_back[:,:((kv+15)//16)]
        common=dict(q=q,cu_q=torch.empty(n+1,device='cuda',dtype=torch.int32),lengths=torch.empty(n,device='cuda',dtype=torch.int32),
            table=table,qcode_dummy=torch.empty(1,device='cuda',dtype=torch.int8),qscale_dummy=torch.empty(1,device='cuda',dtype=torch.float32))
        support=module(ROOT/'tools/kernels/biv150_konly_decode_support.py','cjj_konly_decode_support')
        if 'k8' in modes:
            encoder=support.encode_initial.warmup(key,value,packed.view(torch.int8),packed.view(torch.float32),packed_value,pages,
                grid=(pages*32,),num_warps=1,num_stages=1)
            result['resources_pre_launch']['encoder'],_=preserve(encoder,args.out,'encoder','pre_launch');save()
        ios={};backs={};compiled={};handles={}
        for mode in modes:
            allocation,backs[mode]=buffers(n,capacity,segments,torch)
            validate_binding(mode=mode,format_id=STOCK if mode=='bf16' else FORMAT,num_seqs=n,total_queries=capacity,
                query_strides=q.stride(),output_strides=allocation['out'].stride(),key_strides=(8192,256,128,1) if mode=='bf16' else (16384,256,128,1),
                value_strides=value.stride(),scale_strides=None if mode=='bf16' else (4096,2,1))
            path=args.out/('generated-'+mode+'.py');path.write_text(sources[mode]);compiled[mode]=module(path,'cjj_mixed_numeric_'+mode)
            ios[mode]=dict(common,key=key if mode=='bf16' else packed.view(torch.int8),value=value if mode=='bf16' else packed_value,
                key_scale=packed.view(torch.float32),**allocation)
            for role,fn,kwargs,grid,warps in (
                ('producer',getattr(compiled[mode],NAMES[mode]),producer_arguments(ios[mode],mode,n,capacity,segments),(n,2,segments),1),
                ('reducer',compiled[mode]._reduce_mixed_partials,reducer_arguments(ios[mode],n,capacity,segments),(n,16),4)):
                label=mode+'-'+role;handle=fn.warmup(**kwargs,grid=grid,num_warps=warps,num_stages=1)
                resource,arts=preserve(handle,args.out,label,'pre_launch');result['resources_pre_launch'][label]=resource;save()
                resource['actual_typed_abi']=typed_abi(arts,sources[mode],NAMES[mode] if role=='producer' else '_reduce_mixed_partials',mode)
                if role=='producer':
                    resource['actual_mechanism']=mechanism(arts,sources[mode],mode)
                    assert resource['actual_mechanism']['passed'],'Actual producer IR mechanism rejected'
        result['compile_passed']=True;result['stage']='numeric_after_all_compile_and_ir_gates';save()
        source_hash=hashlib.sha256(raw(stock,torch)).hexdigest()
        assert source_hash==hashlib.sha256(raw(cpu['packed'],torch)).hexdigest(),'Stock BF16 fixture upload changed bits'
        result.update(stock_input_bits_exact=True,query_stride=list(q.stride()),output_stride=list(ios['bf16']['out'].stride()))
        if 'k8' in modes:
            encoder=support.encode_initial[(pages*32,)](key,value,packed.view(torch.int8),packed.view(torch.float32),packed_value,pages,num_warps=1,num_stages=1)
            torch.cuda.synchronize();result['gpu_executed']=True
            result['resources']['encoder'],_=preserve(encoder,args.out,'encoder','post_launch')
            expected=pack_arrays(codes.numpy(),scales.numpy(),cold_v.contiguous().view(torch.int16).numpy().view(np.uint16),np)
            assert hashlib.sha256(raw(packed,torch)).hexdigest()==hashlib.sha256(expected).hexdigest(),'Initial K8 code/scale/V/reserved byte oracle failed'
            assert hashlib.sha256(raw(stock,torch)).hexdigest()==source_hash,'Initial encoder changed its BF16 source'
            result['encoder_bytes_exact']=True
        packed_hash=hashlib.sha256(raw(packed,torch)).hexdigest()
        result.update(stock_cache_sha256=source_hash,packed_cache_sha256=packed_hash);save()
        def upload(c):
            q.copy_(c['q']);common['cu_q'].copy_(torch.tensor(c['cu_q'],dtype=torch.int32))
            common['lengths'].copy_(torch.tensor(c['lengths'],dtype=torch.int32));table.copy_(c['table'][:,:table.shape[1]])
        def step(mode):
            producer=getattr(compiled[mode],NAMES[mode])[(n,2,segments)](**producer_arguments(ios[mode],mode,n,capacity,segments),num_warps=1,num_stages=1)
            reducer=compiled[mode]._reduce_mixed_partials[(n,16)](**reducer_arguments(ios[mode],n,capacity,segments),num_warps=4,num_stages=1)
            handles[mode]=(producer,reducer)
        def verify(mode,c):
            torch.cuda.synchronize()
            assert torch.equal(bits(q,torch),bits(c['q'],torch)) and common['cu_q'].cpu().tolist()==c['cu_q']
            assert bool((q_back[:,16:].cpu()==37.).all()),'Unused fused-QKV query backing overwritten'
            assert common['lengths'].cpu().tolist()==c['lengths'] and torch.equal(table.cpu(),c['table'][:,:table.shape[1]])
            assert bool((table_back[:,table.shape[1]:].cpu()==-2).all()),'Unused block table backing changed'
            for full,sentinel in ((source,90),(packed_back,165)):
                assert bool((full[:256].cpu()==sentinel).all() and (full[-256:].cpu()==sentinel).all())
            assert hashlib.sha256(raw(stock,torch)).hexdigest()==source_hash and hashlib.sha256(raw(packed,torch)).hexdigest()==packed_hash,'Readonly cache modified'
            return output_check(ios[mode],backs[mode],c,segments,torch)
        controls=[];eager={mode:[] for mode in modes}
        numeric_original=True
        for iteration in range(4):
            c=control_metadata(qlens,kv,iteration);c['q']=q0.clone()
            if iteration==1:c['q']=-c['q'].roll(1,1)
            if c['zero_q']:c['q'].zero_()
            c['table']=cpu['table'].roll(c['table_roll'],0).clone()
            for seq,qlen in enumerate(c['query_lengths']):
                if qlen!=1:c['table'][seq].fill_(-1)
            controls.append(c);upload(c);outputs={}
            for mode in modes:
                poison(backs[mode],c);step(mode);snap,out=verify(mode,c);eager[mode].append(snap);outputs[mode]=out
                if mode+'-producer' not in result['resources']:
                    for role,handle in zip(('producer','reducer'),handles[mode]):
                        record,arts=preserve(handle,args.out,mode+'-'+role,'post_launch')
                        record['actual_typed_abi']=typed_abi(arts,sources[mode],NAMES[mode] if role=='producer' else '_reduce_mixed_partials',mode)
                        if role=='producer':record['actual_mechanism']=mechanism(arts,sources[mode],mode)
                        result['resources'][mode+'-'+role]=record
            result['gpu_executed']=True
            decoded=c['decode_sequences']
            selected=(sorted(set(decoded[:8])|{decoded[-1]}) if iteration==1 else sorted({decoded[0],decoded[min(1,len(decoded)-1)],decoded[-1]})) if decoded else []
            # Gather Q by independently computed sequence offsets for the CPU oracle.
            perseq=c['q'][torch.tensor(c['cu_q'][:-1],dtype=torch.int64).clamp_max(capacity-1)]
            oracle=dict(q=perseq,table=c['table'],lengths=c['lengths'],slots=[-1]*n,
                new_k=torch.zeros((n,2,128),dtype=torch.bfloat16),new_v=torch.zeros((n,2,128),dtype=torch.bfloat16))
            refs,_=attention_references(cold_k,cold_v,codes,scales,oracle,selected,torch,np)
            checks=[]
            for r in refs:
                token=c['cu_q'][r['row']];original=r['original'].reshape(-1).tolist();quantized=r['quantized'].reshape(-1).tolist()
                baseline=outputs['bf16'][token].float().reshape(-1).tolist()
                if 'k8' in modes:
                    check=reference_gates(baseline,outputs['k8'][token].float().reshape(-1).tolist(),original,quantized)
                    assert check['baseline_hard_passed'] and check['quantized_operand']['pointwise_passed'],'FP64 implementation hard gate failed'
                    numeric_original &= check['candidate_numeric_passed']
                else:
                    check=dict(baseline=metrics(baseline,original));assert check['baseline']['pointwise_passed']
                checks.append(dict(sequence=r['row'],q_start=token,length=r['length'],integer_bound=r['integer_bound'],**check))
            witness=dict(q_bits=c['q'].view(torch.int16).numpy().view(np.uint16),cu_q=np.array(c['cu_q'],dtype=np.int32),
                lengths=np.array(c['lengths'],dtype=np.int32),table=c['table'].numpy(),selected_sequences=np.array(selected,dtype=np.int32),
                original_reference=np.stack([r['original'].numpy() for r in refs]) if refs else np.empty((0,16,128),dtype=np.float64),
                quantized_reference=np.stack([r['quantized'].numpy() for r in refs]) if refs else np.empty((0,16,128),dtype=np.float64),
                **{mode+'_output_bits':out.view(torch.int16).numpy().view(np.uint16) for mode,out in outputs.items()})
            np.savez_compressed(args.out/('numeric-control%d.npz'%iteration),**witness)
            result['controls'].append(dict(iteration=iteration,query_lengths=c['query_lengths'],lengths=c['lengths'],checks=checks,
                prefill_and_padding_untouched=True,compact_partials_active_and_poison_verified=True,cache_readonly=True));save()
        for mode in modes:
            upload(controls[0]);poison(backs[mode],controls[0]);graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):step(mode)
            for index in (1,2,3,0):
                upload(controls[index]);poison(backs[mode],controls[index]);graph.replay();got,_=verify(mode,controls[index])
                assert all(torch.equal(got[k],eager[mode][index][k]) for k in got),'Changed cu_q/Q/table/length Graph mismatch'
            side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):poison(backs[mode],controls[0]);step(mode)
            torch.cuda.current_stream().wait_stream(side);got,_=verify(mode,controls[0])
            assert all(torch.equal(got[k],eager[mode][0][k]) for k in got),'Side/eager mismatch'
        result.update(completed=True,passed=bool(numeric_original),implementation_hard_gates_passed=True,
            original_bf16_proxy_passed=bool(numeric_original),gpu_numeric_verified=bool(numeric_original),
            gpu_implementation_numeric_verified=True,
            graph_changed_fields=['Q','cu_q','block_table','lengths','decode_membership'],graph_side_exact=True,
            status='private_reader_implementation_only_no_service_or_timing')
        save()
    except Exception as error:
        result['errors'].append(dict(type=type(error).__name__,message=str(error)));(args.out/'error-traceback.txt').write_text(traceback.format_exc());save();raise
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-gpu',action='store_true')
    parser.add_argument('--case',choices=CASES,default='mixed4k');parser.add_argument('--arm',choices=('both','bf16'),default='both')
    parser.add_argument('--commit');parser.add_argument('--out',type=Path);args=parser.parse_args()
    source=(ROOT/'vllm_fl/patches/biv150_gqa_decode.py').read_text();sources,proof=build(source)
    if args.run_gpu:print(json.dumps(run_device(args,sources,proof),indent=2));return
    print(json.dumps(dict(cpu_only=True,numeric_entry_ready=True,source_proof=proof,
        controls={name:[control_metadata(*value,i) for i in range(4)] for name,value in CASES.items()},
        gpu_executed=False,gpu_numeric_verified=False,service_installed=False,timing_prepared=False,
        torch_imported='torch' in sys.modules,triton_imported='triton' in sys.modules,numpy_imported='numpy' in sys.modules),indent=2))


if __name__=='__main__':main()
