#!/usr/bin/env python3
"""Causal enqueue association for v1 MM/prefill/GQA; CPU/read-only SQLite only."""
import argparse,ast,bisect,collections,csv,hashlib,json,math,sqlite3,statistics
from pathlib import Path

TARGETS={'mm_kernel':'mm','kernel_biv150_prefill_normalized':'prefill_attention','_gqa_decode_attention':'gqa_decode'}

def corr(x,y):
    if len(x)<3:return None
    mx=statistics.mean(x);my=statistics.mean(y)
    a=sum((v-mx)**2 for v in x);b=sum((v-my)**2 for v in y)
    return None if a==0 or b==0 else sum((u-mx)*(v-my) for u,v in zip(x,y))/math.sqrt(a*b)

def bucket(v,edges):
    if v is None:return 'none'
    for lo,hi in edges:
        if lo<=v<=hi:return f'{lo}-{hi}'
    return 'outside'

def stats(group):
    result={'steps':len(group)}
    for name in TARGETS.values():
        count=sum(s[name+'_count'] for s in group);total=sum(s[name+'_ns'] for s in group)
        result[name]={'calls':count,'gpu_sum_s':total/1e9,'gpu_mean_kernel_ms':total/count/1e6 if count else None,
                      'gpu_mean_step_ms':total/len(group)/1e6,
                      'steps_with_kernel':sum(s[name+'_count']>0 for s in group)}
    result['scheduled_tokens_min']=min(s['M'] for s in group);result['scheduled_tokens_max']=max(s['M'] for s in group)
    result['prefill_tokens_min']=min(s['prefill_tokens'] for s in group);result['prefill_tokens_max']=max(s['prefill_tokens'] for s in group)
    result['decode_request_count_min']=min(s['decode_request_count'] for s in group);result['decode_request_count_max']=max(s['decode_request_count'] for s in group)
    return result

def grouped(steps,key):
    groups=collections.defaultdict(list)
    for s in steps:groups[str(key(s))].append(s)
    return {k:stats(v) for k,v in sorted(groups.items())}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(exist_ok=False)
    worker_file=args.root/'observations/worker-282216-window-1.json';worker=json.loads(worker_file.read_text());pid=worker['pid']
    c=sqlite3.connect((args.root/'baseline.ixsys').as_uri()+'?mode=ro',uri=True)
    nvs=list(c.execute("SELECT ts,dur,thread_id FROM cuda_nvtx WHERE process_id=? AND name='worker.execute_model' ORDER BY ts",(pid,)))
    assert len(nvs)==len(worker['scheduled_steps'])
    starts=[x[0] for x in nvs];steps=[];prompts={}
    for index,(meta,nvtx) in enumerate(zip(worker['scheduled_steps'],nvs)):
        assert nvtx[0]-10000<=meta['monotonic_ns']<=nvtx[0]+nvtx[1]+10000
        before={};pre=[];dec=[]
        for req in meta['new_requests']:prompts[req['req_id']]=req['prompt_token_count'];before[req['req_id']]=req['num_computed_tokens']
        for req in meta['cached_requests']:before[req['req_id']]=req['num_computed_tokens']
        for req,n in meta['num_scheduled_tokens'].items():
            assert req in before and req in prompts
            context=before[req];pcount=min(n,max(0,prompts[req]-context));dcount=n-pcount
            if pcount:pre.append((context,pcount))
            if dcount:dec.append((context+pcount,dcount))
        ptotal=sum(q for _,q in pre);dtotal=sum(q for _,q in dec)
        assert ptotal+dtotal==meta['scheduled_token_sum']
        value=dict(step=meta['step'],phase='mixed' if pre and dec else 'prefill' if pre else 'decode' if dec else 'empty',
                   M=meta['scheduled_token_sum'],worker_start_ns=nvtx[0],worker_end_ns=nvtx[0]+nvtx[1],worker_tid=nvtx[2],
                   prefill_tokens=ptotal,decode_tokens=dtotal,prefill_request_count=len(pre),decode_request_count=len(dec),
                   prefill_context_before_min=min((a for a,q in pre),default=None),prefill_context_before_max=max((a for a,q in pre),default=None),
                   prefill_context_after_min=min((a+q for a,q in pre),default=None),prefill_context_after_max=max((a+q for a,q in pre),default=None),
                   decode_context_before_min=min((a for a,q in dec),default=None),decode_context_before_max=max((a for a,q in dec),default=None),
                   decode_context_after_min=min((a+q for a,q in dec),default=None),decode_context_after_max=max((a+q for a,q in dec),default=None),
                   prefill_causal_attention_pairs=sum(q*a+q*(q+1)//2 for a,q in pre),
                   decode_causal_attention_pairs=sum(q*a+q*(q+1)//2 for a,q in dec),
                   decode_context_after_mean=(sum(a+q for a,q in dec)/len(dec) if dec else None))
        value['gqa_grid_seq_count']=None
        value['gqa_grid_segments']=None
        for name in TARGETS.values():value[name+'_ns']=0;value[name+'_count']=0
        steps.append(value)
    kernels=list(c.execute("SELECT id,name,ts,dur,correlationId,grid,block,dynamicSharedMemory FROM cuda_kernel WHERE processId=? AND name IN ('mm_kernel','kernel_biv150_prefill_normalized','_gqa_decode_attention')",(pid,)))
    wanted={r[4] for r in kernels};apis=collections.defaultdict(list)
    ids=[r[0] for r in c.execute("SELECT id FROM function_name_map WHERE name IN ('cuLaunchKernel','cudaLaunchKernel')")]
    for row in c.execute('SELECT correlationId,ts,dur,threadId FROM cuda_api_data WHERE processId=? AND name_id IN ('+','.join('?'*len(ids))+')',[pid,*ids]):
        if row[0] in wanted:apis[row[0]].append(row)
    counts=collections.Counter();errors=[];samples=[];mm_by_step=collections.defaultdict(list);families=collections.defaultdict(list)
    for kid,name,ts,dur,cid,grid,block,shared in kernels:
        if len(apis[cid])!=1:errors.append({'kernel_id':kid,'correlationId':cid,'api_matches':len(apis[cid])});continue
        _,enqueue,api_dur,tid=apis[cid][0];index=bisect.bisect_right(starts,enqueue)-1
        if index<0 or enqueue+api_dur>steps[index]['worker_end_ns'] or tid!=steps[index]['worker_tid']:
            errors.append({'kernel_id':kid,'reason':'enqueue not fully inside same-thread worker NVTX'});continue
        step=steps[index];group=TARGETS[name];step[group+'_count']+=1;step[group+'_ns']+=dur;counts['mapped']+=1
        assert ts>=enqueue
        if ts>step['worker_end_ns']:counts['gpu_start_after_origin_worker_return']+=1
        if ts+dur>step['worker_end_ns']:counts['gpu_end_after_origin_worker_return']+=1
        if name=='mm_kernel':
            record=dict(kernel_id=kid,correlationId=cid,enqueue_ns=enqueue,gpu_ns=ts,duration_ns=dur,grid=grid,block=block,dynamic_shared=shared,M=step['M'])
            mm_by_step[index].append(record);families[(step['M'],grid,block,shared)].append(dur)
        elif name=='_gqa_decode_attention':
            gx,heads,segments=ast.literal_eval(grid)
            assert heads==2 and gx==step['decode_request_count']
            assert step['gqa_grid_seq_count'] in (None,gx)
            assert step['gqa_grid_segments'] in (None,segments)
            step['gqa_grid_seq_count']=gx;step['gqa_grid_segments']=segments
        if len(samples)<12 or (step['M']!=2048 and name=='mm_kernel' and len(samples)<24):
            samples.append(dict(kernel_id=kid,kernel=name,correlationId=cid,enqueue_ns=enqueue,enqueue_end_ns=enqueue+api_dur,
                                gpu_start_ns=ts,gpu_end_ns=ts+dur,worker_step=step['step'],worker_start_ns=step['worker_start_ns'],worker_end_ns=step['worker_end_ns']))
    assert not errors,errors[:5]
    active=[s for s in steps if s['mm_count'] or s['prefill_attention_count'] or s['gqa_decode_count']]
    assert all(s['mm_count']==134 and s['prefill_attention_count']==42 and s['gqa_decode_count'] in (0,42) for s in active)
    assert len(active)==526 and counts['mapped']==len(kernels)
    # Secondary, explicit model-order inference; runtime API arguments were not captured.
    model_order=[]
    for layer in range(42):
        model_order.extend([(layer,'qkv',2560,2048),(layer,'o',2048,2048)])
        if layer<4 or layer>=38:model_order.append((layer,'bf16_edge_gate_up',12288,2048))
        model_order.append((layer,'down',2048,6144))
    assert len(model_order)==134
    shapes=collections.defaultdict(list)
    for index,events in mm_by_step.items():
        events.sort(key=lambda r:r['enqueue_ns']);assert len(events)==134
        for event,(layer,role,N,K) in zip(events,model_order):
            bn=256 if role=='bf16_edge_gate_up' else 128
            expected_grid=((event['M']+255)//256*((N+bn-1)//bn),1,1)
            assert ast.literal_eval(event['grid'])==expected_grid
            assert ast.literal_eval(event['block'])==((1024 if role=='bf16_edge_gate_up' else 512),1,1)
            assert event['dynamic_shared']==(65536 if role=='bf16_edge_gate_up' else 49152)
            shapes[(event['M'],N,K,role)].append(event['duration_ns'])
    def duration_group(key,values):return dict(key=key,calls=len(values),gpu_sum_s=sum(values)/1e9,gpu_mean_kernel_ms=statistics.mean(values)/1e6)
    shape_rows=[duration_group(dict(M=M,N=N,K=K,role=role,evidence='source-order inference checked against 134 calls/step and grid/block/shared memory'),values)
                for (M,N,K,role),values in sorted(shapes.items())]
    family_rows=[duration_group(dict(M=M,grid=grid,block=block,dynamic_shared_bytes=shared,evidence='runtime scheduler plus verified eager padding; runtime trace geometry'),values)
                for (M,grid,block,shared),values in sorted(families.items())]
    dimensions={
        'phase':grouped(active,lambda s:s['phase']),
        'scheduled_M':grouped(active,lambda s:s['M']),
        'prefill_tokens_exact':grouped(active,lambda s:s['prefill_tokens']),
        'decode_request_count_exact':grouped(active,lambda s:s['decode_request_count']),
        'decode_request_count_bucket':grouped(active,lambda s:bucket(s['decode_request_count'],[(0,0),(1,4),(5,16),(17,32),(33,48),(49,63)])),
        'prefill_context_before_max_bucket':grouped(active,lambda s:bucket(s['prefill_context_before_max'],[(0,0),(1,1024),(1025,2048),(2049,3072),(3073,4095)])),
        'decode_context_after_max_bucket':grouped(active,lambda s:bucket(s['decode_context_after_max'],[(4097,4352),(4353,4608),(4609,4864),(4865,5120)])),
        'decode_context_after_mean_bucket':grouped(active,lambda s:bucket(s['decode_context_after_mean'],[(4097,4352),(4352.000001,4608),(4608.000001,4864),(4864.000001,5120)])),
        'prefill_attention_pairs_bucket':grouped(active,lambda s:bucket(s['prefill_causal_attention_pairs'],[(0,2500000),(2500001,3500000),(3500001,4500000),(4500001,5500000),(5500001,6500000)])),
        'prefill_request_count':grouped(active,lambda s:s['prefill_request_count']),
        'mixed_gqa_present':grouped([s for s in active if s['phase']=='mixed'],lambda s:bool(s['gqa_decode_count'])),
        'gqa_runtime_grid':grouped([s for s in active if s['gqa_decode_count']],lambda s:f"{s['gqa_grid_seq_count']},2,{s['gqa_grid_segments']}"),
        'M2048_decode61_context_mean':grouped([s for s in active if s['M']==2048 and s['decode_request_count']==61 and s['gqa_decode_count']],lambda s:bucket(s['decode_context_after_mean'],[(4097,4352),(4352.000001,4608),(4608.000001,4864),(4864.000001,5120)])),
    }
    correlations={}
    m2048=[s for s in active if s['M']==2048]
    for subset_name,subset in [('all_526',active),('M2048_only',m2048),('M2048_gqa_present',[s for s in m2048 if s['gqa_decode_count']])]:
        correlations[subset_name]={}
        for kernel in TARGETS.values():
            y=[s[kernel+'_ns'] for s in subset]
            correlations[subset_name][kernel]={x:corr([s[x] for s in subset],y) for x in ('M','prefill_tokens','decode_request_count','prefill_causal_attention_pairs','decode_causal_attention_pairs')}
    mm2048=sum(s['mm_ns'] for s in m2048);mmtotal=sum(s['mm_ns'] for s in active)
    summary=dict(diagnostic_only=True,official_score=False,trace=str(args.root/'baseline.ixsys'),worker_pid=pid,
                 association=dict(method='same PID + unique GPU correlationId to cuLaunchKernel/cudaLaunchKernel; enqueue full interval contained in same-thread worker NVTX',
                                  selected_gpu_kernels=len(kernels),unique_api_correlations=len(apis),errors=errors,**dict(counts)),
                 baseline_step_counts=dict(all_worker_steps=len(steps),non_graph_target_steps=len(active)),
                 gqa_runtime_grid_checks=dict(launches=sum(s['gqa_decode_count'] for s in active),steps=sum(bool(s['gqa_decode_count']) for s in active),grid_x_matches_full_scheduled_decode_count=True,mismatches=0),
                 totals=stats(active),M2048=dict(steps=len(m2048),steps_fraction=len(m2048)/len(active),mm_calls=sum(s['mm_count'] for s in m2048),
                                             mm_call_fraction=sum(s['mm_count'] for s in m2048)/sum(s['mm_count'] for s in active),mm_gpu_s=mm2048/1e9,mm_gpu_fraction=mm2048/mmtotal),
                 M_evidence='scheduler runtime metadata + source-verified eager no-padding path: TP=DP=1, enable_sp=false, all M>max_capture512, dispatcher returns BatchDescriptor(num_tokens), model input slice uses this size; not captured MM argument registers',
                 shape_evidence='Grid/block/shared memory is directly traced. N/K/role split is a separately labelled source-order inference; O and down share geometry and cannot be distinguished from grid alone.',
                 attention_scope_evidence='On all 516 GQA-launching steps every runtime grid.x equals the full scheduled decode-request count; source grid=(num_seqs,2,segments) establishes the count directly. Eight mixed steps have no GQA launch: the exact failed split-eligibility predicate was not recorded. For those eight, prefill-only causal pairs omit decode queries handled by the normalized kernel.',
                 correlation_caveat='Pearson relationships are descriptive and confounded by context/batch composition; they are not causal speedup estimates.',
                 correlations=correlations)
    for name,value in [('summary',summary),('grouped_dimensions',dimensions),('mm_runtime_geometry',family_rows),('mm_model_order_shapes',shape_rows),('association_samples',samples)]:
        (args.out/(name+'.json')).write_text(json.dumps(value,indent=2))
    with (args.out/'per_step_kernel_times.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(active[0]));writer.writeheader();writer.writerows(active)
    print(json.dumps(summary,indent=2))
    print('PHASE',json.dumps(dimensions['phase']))
    print('DECODE_COUNT',json.dumps(dimensions['decode_request_count_bucket']))
    print('PREFILL_CONTEXT',json.dumps(dimensions['prefill_context_before_max_bucket']))

if __name__=='__main__':main()
