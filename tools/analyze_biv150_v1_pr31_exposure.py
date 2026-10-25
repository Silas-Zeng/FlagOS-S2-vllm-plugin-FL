#!/usr/bin/env python3
"""Read-only v1 enqueue association for already-explored PR31 GQA batches."""
import argparse,bisect,collections,csv,hashlib,json,sqlite3
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--steps',type=Path,required=True)
    parser.add_argument('--timeline-summary',type=Path,required=True)
    parser.add_argument('--pr31-snapshot',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    steps=list(csv.DictReader(args.steps.open()))
    starts=[int(s['worker_start_ns']) for s in steps]
    assert starts==sorted(starts)
    c=sqlite3.connect(args.trace.as_uri()+'?mode=ro',uri=True)
    pid=282216
    kernels=list(c.execute("SELECT name,correlationId,dur FROM cuda_kernel WHERE processId=? AND name IN ('_gqa_decode_attention','_reduce_partials')",(pid,)))
    wanted={r[1] for r in kernels};apis=collections.defaultdict(list)
    ids=[r[0] for r in c.execute("SELECT id FROM function_name_map WHERE name IN ('cuLaunchKernel','cudaLaunchKernel')")]
    for row in c.execute('SELECT correlationId,ts,dur,threadId FROM cuda_api_data WHERE processId=? AND name_id IN ('+','.join('?'*len(ids))+')',[pid,*ids]):
        if row[0] in wanted:apis[row[0]].append(row)
    associated=collections.defaultdict(list)
    for name,cid,dur in kernels:
        assert len(apis[cid])==1,(name,cid,len(apis[cid]))
        _,ts,host_dur,tid=apis[cid][0]
        index=bisect.bisect_right(starts,ts)-1
        assert index>=0
        step=steps[index]
        assert ts+host_dur<=int(step['worker_end_ns']) and tid==int(step['worker_tid'])
        associated[index].append((ts,name,dur))
    per_batch=collections.defaultdict(lambda:dict(steps=0,producer_calls=0,reducer_calls=0,producer_ns=0,reducer_ns=0,baseline_segments=set()))
    pairs=0
    for index,events in associated.items():
        step=steps[index];events.sort()
        assert len(events)==2*int(step['gqa_decode_count'])
        for a,b in zip(events[::2],events[1::2]):
            assert a[1]=='_gqa_decode_attention' and b[1]=='_reduce_partials'
            pairs+=1
        batch=int(step['gqa_grid_seq_count']);value=per_batch[batch]
        assert batch==int(step['decode_request_count'])
        value['steps']+=1;value['baseline_segments'].add(int(step['gqa_grid_segments']))
        for _,name,dur in events:
            prefix='producer' if name=='_gqa_decode_attention' else 'reducer'
            value[prefix+'_calls']+=1;value[prefix+'_ns']+=dur
    selected=[29,30,60,61,62,63]
    timeline=json.loads(args.timeline_summary.read_text());wall=timeline['active_window']['duration_s']
    for batch,value in per_batch.items():
        value['baseline_segments']=sorted(value['baseline_segments'])
        value['producer_gpu_s']=value.pop('producer_ns')/1e9
        value['reducer_gpu_s']=value.pop('reducer_ns')/1e9
        value['combined_gpu_s']=value['producer_gpu_s']+value['reducer_gpu_s']
        value['fraction_of_active_window']=value['combined_gpu_s']/wall
    total=dict(steps=sum(per_batch[b]['steps'] for b in selected),producer_calls=sum(per_batch[b]['producer_calls'] for b in selected),reducer_calls=sum(per_batch[b]['reducer_calls'] for b in selected))
    for key in ['producer_gpu_s','reducer_gpu_s','combined_gpu_s']:
        total[key]=sum(per_batch[b][key] for b in selected)
    total['fraction_of_active_window']=total['combined_gpu_s']/wall
    total['fraction_of_non_graph_gqa_producer_reducer']=total['combined_gpu_s']/sum(v['combined_gpu_s'] for v in per_batch.values())
    pr=json.loads(args.pr31_snapshot.read_text())
    result=dict(diagnostic_only=True,official_score=False,method='same PID and unique enqueue correlation; complete same-thread worker NVTX containment; all producer/reducer pairs alternate by enqueue order',verified_pairs=pairs,association_errors=0,active_window_s=wall,selected_batches=selected,selected={str(b):per_batch[b] for b in selected},selected_total=total,all_non_graph_gqa={str(b):v for b,v in sorted(per_batch.items())},pr31=dict(url=pr['html_url'],head_sha=pr['head']['sha'],snapshot_sha256=hashlib.sha256(args.pr31_snapshot.read_bytes()).hexdigest(),body=pr['body']),limitations=['This is baseline v1 producer plus reducer device duration, not an incremental speedup measurement.','Other wrapper, metadata, launch, and downstream costs are excluded; this is not complete service cost.','Opaque Graph internals are excluded and cannot be assigned to these batch values from v1.','PR31 already explores B29/30 -> 4 segments and B60..63 -> 2; do not re-propose these as a new direction.'])
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'verified_pairs':pairs,'selected_total':total,'selected':result['selected']},ensure_ascii=False))

if __name__=='__main__':main()
