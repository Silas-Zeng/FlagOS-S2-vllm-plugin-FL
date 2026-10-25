#!/usr/bin/env python3
"""Read-only analysis of the completed v1 4K ixsys trace; no GPU imports."""
import argparse,bisect,collections,json,math,sqlite3,statistics
from pathlib import Path


def merge(intervals):
    out=[]
    for a,b in sorted(intervals):
        if b<=a:continue
        if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
        else:out.append([a,b])
    return out


class Coverage:
    def __init__(self,intervals):
        self.intervals=merge(intervals)
        self.starts=[a for a,b in self.intervals]
        self.ends=[b for a,b in self.intervals]
        self.prefix=[0]
        for a,b in self.intervals:self.prefix.append(self.prefix[-1]+b-a)
    def measure(self,a,b):
        if b<=a:return 0
        lo=bisect.bisect_right(self.ends,a)
        hi=bisect.bisect_left(self.starts,b)
        if lo>=hi:return 0
        return self.prefix[hi]-self.prefix[lo]-max(0,a-self.starts[lo])-max(0,self.ends[hi-1]-b)


def quantile(values,q):
    values=sorted(values)
    if not values:return None
    index=(len(values)-1)*q
    a=math.floor(index);b=math.ceil(index)
    return values[a]+(values[b]-values[a])*(index-a)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(exist_ok=False)
    c=sqlite3.connect((args.root/'baseline.ixsys').as_uri()+'?mode=ro',uri=True)
    worker=json.loads((args.root/'observations/worker-282216-window-1.json').read_text())
    pid=worker['pid']
    rows=[dict(id=i,tid=t,start=s,end=s+d,name=n,depth=dep,dur=d) for i,t,s,d,n,dep in
          c.execute('SELECT id,thread_id,ts,dur,name,depth FROM cuda_nvtx WHERE process_id=? AND dur>=0',(pid,))]
    diag=merge([(r['start'],r['end']) for r in rows if r['name'].startswith('biv150_diag')])
    # V1 did not enter the faulty graph probe, so no diagnostic duplicate work exists.
    if diag:raise RuntimeError('This v1 analysis requires explicit subtraction before handling diagnostic ranges')
    steps_outer=[r for r in rows if r['name']=='engine.step_with_batch_queue']
    start=min(r['start'] for r in steps_outer)
    end=max(r['end'] for r in rows)
    rows=[r for r in rows if r['start']>=start and r['end']<=end]
    gpu={};all_intervals=[];tables=('cuda_graph','cuda_kernel','cuda_memcpy','cuda_memset')
    for table in tables:
        events=[(max(start,s),min(end,s+d),stream) for s,d,stream in
                c.execute('SELECT ts,dur,streamId FROM '+table+' WHERE processId=? AND dur>=0 AND ts<? AND ts+dur>?',(pid,end,start))]
        intervals=[(a,b) for a,b,_ in events]
        cover=Coverage(intervals)
        gpu[table]=dict(count=len(events),sum_s=sum(b-a for a,b in intervals)/1e9,
                        union_s=cover.measure(start,end)/1e9,streams=sorted({x[2] for x in events}))
        all_intervals.extend(intervals)
    cover=Coverage(all_intervals)
    busy=cover.measure(start,end)
    # Strictly nested NVTX stacks provide direct-child subtraction, independently per CPU thread.
    stack=[];last_tid=None;crossings=[]
    for r in sorted(rows,key=lambda r:(r['tid'],r['start'],-r['end'],r['depth'],r['id'])):
        if last_tid!=r['tid']:stack=[];last_tid=r['tid']
        while stack and stack[-1]['end']<=r['start']:stack.pop()
        r['exclusive']=r['dur'];r['gpu_overlap']=cover.measure(r['start'],r['end'])
        r['exclusive_gpu_overlap']=r['gpu_overlap']
        if stack:
            parent=stack[-1]
            if r['end']>parent['end']:crossings.append([parent['id'],r['id']])
            else:
                parent['exclusive']-=r['dur']
                parent['exclusive_gpu_overlap']-=r['gpu_overlap']
        stack.append(r)
    if crossings:raise RuntimeError(f'Crossing NVTX ranges: {crossings[:8]}')
    if min(r['exclusive'] for r in rows)<0:raise RuntimeError('Negative exclusive duration')
    grouped=collections.defaultdict(list)
    for r in rows:grouped[r['name']].append(r)
    scope=[]
    for name,group in grouped.items():
        durations=[r['dur']/1e6 for r in group]
        scope.append(dict(name=name,count=len(group),inclusive_s=sum(r['dur'] for r in group)/1e9,
                          exclusive_s=sum(r['exclusive'] for r in group)/1e9,
                          inclusive_gpu_overlap_s=sum(r['gpu_overlap'] for r in group)/1e9,
                          exclusive_gpu_uncovered_s=sum(r['exclusive']-r['exclusive_gpu_overlap'] for r in group)/1e9,
                          p50_ms=quantile(durations,.5),p95_ms=quantile(durations,.95),p99_ms=quantile(durations,.99)))
    scope.sort(key=lambda r:r['inclusive_s'],reverse=True)
    (args.out/'cpu_scopes.json').write_text(json.dumps(scope,indent=2))
    kernel_top=[dict(name=n,count=count,gpu_sum_s=total/1e9,mean_ms=mean/1e6,max_ms=maximum/1e6)
        for n,count,total,mean,maximum in c.execute('SELECT name,COUNT(*),SUM(dur),AVG(dur),MAX(dur) FROM cuda_kernel WHERE processId=? AND ts>=? AND ts+dur<=? GROUP BY name ORDER BY SUM(dur) DESC',(pid,start,end))]
    (args.out/'non_graph_kernel_hotspots.json').write_text(json.dumps(kernel_top,indent=2))
    api_top=[dict(name=n,count=count,host_sum_s=total/1e9,mean_us=mean/1e3)
        for n,count,total,mean in c.execute('SELECT name,COUNT(*),SUM(dur),AVG(dur) FROM cuda_api WHERE processId=? AND ts>=? AND ts+dur<=? GROUP BY name ORDER BY SUM(dur) DESC LIMIT 30',(pid,start,end))]
    (args.out/'cuda_api_host_durations.json').write_text(json.dumps(api_top,indent=2))
    # Phase labels use request prompt length and computed-token count, never cached/new alone.
    execution=sorted(grouped['worker.execute_model'],key=lambda r:r['start'])
    metadata=worker['scheduled_steps']
    if len(execution)!=len(metadata):raise RuntimeError('Worker NVTX / metadata count mismatch')
    prompts={};phases=collections.defaultdict(lambda:dict(count=0,scheduled_tokens=0,prefill_tokens=0,decode_tokens=0,worker_host_s=0))
    phase_records=[];unknown=[];clock_fail=[]
    for event,step in zip(execution,metadata):
        if not event['start']-10000<=step['monotonic_ns']<=event['end']+10000:clock_fail.append(step['step'])
        before={}
        for req in step['new_requests']:
            prompts[req['req_id']]=req['prompt_token_count'];before[req['req_id']]=req['num_computed_tokens']
        for req in step['cached_requests']:before[req['req_id']]=req['num_computed_tokens']
        prefill=decode=0
        for req,n in step['num_scheduled_tokens'].items():
            if req not in prompts or req not in before:unknown.append([step['step'],req]);continue
            pre=min(n,max(0,prompts[req]-before[req]));prefill+=pre;decode+=n-pre
        label='mixed' if prefill and decode else 'prefill' if prefill else 'decode' if decode else 'empty'
        v=phases[label];v['count']+=1;v['scheduled_tokens']+=step['scheduled_token_sum'];v['prefill_tokens']+=prefill;v['decode_tokens']+=decode;v['worker_host_s']+=event['dur']/1e9
        phase_records.append(dict(step=step['step'],phase=label,start=event['start'],end=event['end'],scheduled_tokens=step['scheduled_token_sum'],prefill_tokens=prefill,decode_tokens=decode))
    if unknown or clock_fail:raise RuntimeError(f'Unverified phase mapping: unknown={unknown[:5]},clock={clock_fail[:5]}')
    (args.out/'worker_phase_steps.json').write_text(json.dumps(phase_records,indent=2))
    gaps=[];cursor=start
    for a,b in cover.intervals:
        if a>cursor:gaps.append([cursor,a])
        cursor=max(cursor,b)
    if cursor<end:gaps.append([cursor,end])
    summary=dict(source_trace=str(args.root/'baseline.ixsys'),pid=pid,diagnostic_only=True,official_score=False,
                 active_window=dict(start_ns=start,end_ns=end,duration_s=(end-start)/1e9,
                                    definition='first active EngineCore step through final active CPU NVTX range'),
                 worker_profile_envelope_monotonic_s=(worker['finished_monotonic_ns']-worker['started_monotonic_ns'])/1e9,
                 worker_profile_envelope_gpu_coverage_fraction=busy/(worker['finished_monotonic_ns']-worker['started_monotonic_ns']),
                 worker_profile_envelope_uncovered_s=(worker['finished_monotonic_ns']-worker['started_monotonic_ns']-busy)/1e9,
                 no_startup_or_warmup_ranges=True,excluded_diagnostic_ranges=len(diag),
                 graph_probe_samples=len(worker['samples']),graph_internal_kernel_visibility=False,
                 scope_errors=worker['scope_diagnostics']['error_total'],cpu_threads=sorted({r['tid'] for r in rows}),
                 gpu=gpu,gpu_union_s=busy/1e9,gpu_union_fraction=busy/(end-start),
                 gpu_uncovered_s=(end-start-busy)/1e9,
                 gpu_uncovered_intervals=dict(count=len(gaps),max_ms=max((b-a for a,b in gaps),default=0)/1e6,
                                              p99_ms=quantile([(b-a)/1e6 for a,b in gaps],.99)),
                 worker_phases=phases,phase_mapping_clock_matches=len(metadata),
                 numeric_graph_clone_validation_passed=False,
                 caveats=['Host NVTX widths are wall time, not CPU execution cycles.',
                          'GPU union includes complete opaque CUDA Graph intervals; internal missing nodes are never gaps.',
                          'Uncovered time is absence of recorded GPU activity, not a claim that CPU caused it.',
                          'Captured kernel hotspots are non-Graph operations only; no model-wide kernel ranking is possible from v1.',
                          'Profiler/metadata overhead is present; these are diagnostic measurements, not formal score results.'])
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    print('CPU_EXCLUSIVE_TOP',json.dumps([v for v in scope if not v['name'].startswith('execute_new')][:23],indent=2))

if __name__=='__main__':main()
