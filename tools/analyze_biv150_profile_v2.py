#!/usr/bin/env python3
"""Read-only completed v2 ixsys analysis, with one shared CPU/GPU exclusion mask.

No GPU imports. The --completion JSON must document Report saved and collector
exit before this program opens the immutable local SQLite mirror.
"""
import argparse,bisect,collections,hashlib,json,math,sqlite3
from pathlib import Path


def merged(intervals):
    out=[]
    for a,b in sorted(tuple(x) for x in intervals):
        if b<=a:continue
        if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
        else:out.append([a,b])
    return out


class Intervals:
    def __init__(self,intervals=()):
        self.parts=merged(intervals)
        self.starts=[a for a,b in self.parts];self.ends=[b for a,b in self.parts]
        self.prefix=[0]
        for a,b in self.parts:self.prefix.append(self.prefix[-1]+b-a)
    @property
    def total(self):return self.prefix[-1]
    def measure(self,a,b):
        if b<=a:return 0
        lo=bisect.bisect_right(self.ends,a);hi=bisect.bisect_left(self.starts,b)
        if lo>=hi:return 0
        return self.prefix[hi]-self.prefix[lo]-max(0,a-self.starts[lo])-max(0,self.ends[hi-1]-b)
    def clip(self,a,b):
        if b<=a:return []
        lo=bisect.bisect_right(self.ends,a);hi=bisect.bisect_left(self.starts,b)
        return [(max(a,x),min(b,y)) for x,y in self.parts[lo:hi]]
    def subtract(self,other):
        out=[]
        for a,b in self.parts:
            cursor=a
            for x,y in other.clip(a,b):
                if x>cursor:out.append((cursor,x))
                cursor=max(cursor,y)
            if cursor<b:out.append((cursor,b))
        return Intervals(out)


def quantile(values,q):
    values=sorted(values)
    if not values:return None
    pos=(len(values)-1)*q;lo=math.floor(pos);hi=math.ceil(pos)
    return values[lo]+(values[hi]-values[lo])*(pos-lo)


def scope_records(rows,keep,gpu):
    stack=[];last_tid=None;out=[]
    for original in sorted(rows,key=lambda r:(r['tid'],r['start'],-r['end'],r['depth'],r['id'])):
        r=dict(original)
        if last_tid!=r['tid']:stack=[];last_tid=r['tid']
        while stack and stack[-1]['end']<=r['start']:stack.pop()
        r['kept_ns']=keep.measure(r['start'],r['end'])
        r['exclusive_ns']=r['kept_ns'];r['overlap_ns']=gpu.measure(r['start'],r['end']);r['exclusive_overlap_ns']=r['overlap_ns']
        if stack:
            parent=stack[-1]
            if r['end']>parent['end']:raise ValueError(f"Crossing NVTX ranges {parent['id']}/{r['id']}")
            parent['exclusive_ns']-=r['kept_ns'];parent['exclusive_overlap_ns']-=r['overlap_ns']
        stack.append(r);out.append(r)
    assert all(0<=r['exclusive_overlap_ns']<=r['exclusive_ns']<=r['kept_ns'] for r in out)
    return out


def self_test():
    # Union, touching boundaries, crossing exclusion, and independently separated windows.
    outer=Intervals([(0,100)]);diag=Intervals([(30,50),(45,60)])
    keep=outer.subtract(diag);assert keep.parts==[[0,30],[60,100]] and keep.total==70
    assert keep.measure(20,70)==20 and keep.clip(30,60)==[]
    assert Intervals([(0,50),(40,70)]).total==70
    assert Intervals([(0,100),(200,300)]).subtract(Intervals([(90,210)])).parts==[[0,90],[210,300]]
    gpu=Intervals([(20,30),(60,80)])
    rows=[dict(id=1,tid=1,start=0,end=100,depth=0,name='parent'),dict(id=2,tid=1,start=20,end=80,depth=1,name='child'),dict(id=3,tid=1,start=30,end=60,depth=2,name='biv150_diag.sample')]
    r=scope_records(rows,keep,gpu)
    assert [(x['kept_ns'],x['exclusive_ns'],x['exclusive_overlap_ns']) for x in r]==[(70,40,0),(30,30,30),(0,0,0)]
    # Asynchronous diagnostic work can extend the same mask for BOTH CPU and GPU.
    expanded=Intervals(diag.parts+[(55,65)]);same=outer.subtract(expanded)
    assert same.total==65 and same.measure(20,70)==15
    # Independent discrete set oracle checks clipping/union/subtraction at all edges.
    for ranges in [[(1,8),(3,4),(8,12)],[(0,3),(7,15)],[(4,4),(5,10)]]:
        real=Intervals(ranges);oracle={i for a,b in ranges for i in range(a,b)}
        assert real.total==len(oracle)
        for a in range(16):
            for b in range(a,17):
                assert real.measure(a,b)==len(oracle & set(range(a,b)))
                assert sum(y-x for x,y in real.clip(a,b))==real.measure(a,b)
        cut=Intervals([(2,6),(10,13)]);remaining=real.subtract(cut)
        assert remaining.total==len(oracle-{i for a,b in cut.parts for i in range(a,b)})
    try:scope_records([rows[0],dict(id=4,tid=1,start=90,end=110,depth=1,name='bad')],keep,gpu)
    except ValueError:pass
    else:raise AssertionError('Crossing scopes must fail')
    print('CPU interval/union/nesting/window/async-mask checks passed')


def write_json(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def phases_for(worker,rows,keep):
    execution=sorted((r for r in rows if r['name']=='worker.execute_model'),key=lambda r:r['start'])
    metadata=worker['scheduled_steps'];assert len(execution)==len(metadata),(len(execution),len(metadata))
    prompts={};phases=collections.defaultdict(lambda:dict(count=0,scheduled_tokens=0,prefill_tokens=0,decode_tokens=0,worker_host_retained_ns=0));records=[]
    for event,step in zip(execution,metadata):
        assert event['start']-10000<=step['monotonic_ns']<=event['end']+10000
        before={}
        for req in step['new_requests']:prompts[req['req_id']]=req['prompt_token_count'];before[req['req_id']]=req['num_computed_tokens']
        for req in step['cached_requests']:before[req['req_id']]=req['num_computed_tokens']
        prefill=decode=0
        for req,n in step['num_scheduled_tokens'].items():
            assert req in prompts and req in before
            pre=min(n,max(0,prompts[req]-before[req]));prefill+=pre;decode+=n-pre
        assert prefill+decode==step['scheduled_token_sum']
        phase='mixed' if prefill and decode else 'prefill' if prefill else 'decode' if decode else 'empty'
        value=phases[phase];value['count']+=1;value['scheduled_tokens']+=step['scheduled_token_sum'];value['prefill_tokens']+=prefill;value['decode_tokens']+=decode
        value['worker_host_retained_ns']+=keep.measure(event['start'],event['end'])
        records.append(dict(step=step['step'],phase=phase,start_ns=event['start'],end_ns=event['end'],scheduled_tokens=step['scheduled_token_sum'],prefill_tokens=prefill,decode_tokens=decode,worker_host_retained_ns=keep.measure(event['start'],event['end'])))
    return phases,records


def analyze_window(c,worker,folder):
    folder.mkdir();pid=worker['pid'];epoch=worker['epoch'];ws=worker['started_monotonic_ns'];we=worker['finished_monotonic_ns']
    assert not worker['errors'] and worker['scope_diagnostics']['error_total']==0
    rows=[dict(id=i,tid=tid,start=ts,end=ts+dur,name=name,depth=depth) for i,tid,ts,dur,name,depth in c.execute('SELECT id,thread_id,ts,dur,name,depth FROM cuda_nvtx WHERE process_id=? AND dur>=0 AND ts<? AND ts+dur>?',(pid,we,ws))]
    engine=[r for r in rows if r['name']=='engine.step_with_batch_queue']
    assert engine
    start=max(ws,min(r['start'] for r in engine));end=min(we,max(r['end'] for r in rows))
    envelope=Intervals([(start,end)])
    diagnostics=[r for r in rows if r['name']=='biv150_diag.sample']
    assert len(diagnostics)==len(worker['samples']) and len(diagnostics)==4
    diag=Intervals((r['start'],r['end']) for r in diagnostics)
    assert len(diag.parts)==len(diagnostics),'Overlapping outer diagnostic samples'
    for r in rows:
        if r['name'].startswith('biv150_diag.'):
            assert diag.measure(r['start'],r['end'])==r['end']-r['start']
    sample_steps={s['scheduler']['step'] for s in worker['samples']}
    assert len(sample_steps)==len(diagnostics)
    for sample in worker['samples']:
        assert sample['original_control']['numeric_exact']
        assert all(x['numeric_exact'] and x['all_api_calls_succeeded'] for x in sample['observations'])
    apis=list(c.execute('SELECT name,ts,dur,threadId,correlationId FROM cuda_api WHERE processId=? AND dur>=0 AND ts<? AND ts+dur>?',(pid,we,ws)))
    diag_cids=set();diag_api_count=0
    for name,ts,dur,tid,cid in apis:
        owners=[r for r in diagnostics if r['tid']==tid and r['start']<=ts<r['end']]
        if owners:
            assert len(owners)==1 and ts+dur<=owners[0]['end']
            if cid is not None and cid>0:diag_cids.add(cid)
            diag_api_count+=1
    raw={};diag_events=collections.defaultdict(list);extensions=[]
    tables=('cuda_graph','cuda_kernel','cuda_memcpy','cuda_memset')
    for table in tables:
        raw[table]=[dict(id=i,name=n,start=ts,end=ts+dur,stream=stream,cid=cid) for i,n,ts,dur,stream,cid in c.execute('SELECT id,name,ts,dur,streamId,correlationId FROM '+table+' WHERE processId=? AND dur>=0 AND ts<? AND ts+dur>?',(pid,we,ws))]
        for event in raw[table]:
            if event['cid'] in diag_cids:
                diag_events[table].append(event)
                if diag.measure(event['start'],event['end'])<event['end']-event['start']:extensions.append(dict(table=table,**event))
    expected_diag_graphs=sum(2+len(s['observations']) for s in worker['samples'])
    assert len(diag_events['cuda_graph'])==expected_diag_graphs, ('Missing diagnostic Graph correlation',len(diag_events['cuda_graph']),expected_diag_graphs)
    expected_raw_graphs=sum(worker['graph_replay_counts'].values())+expected_diag_graphs-len(worker['samples'])
    assert len(raw['cuda_graph'])==expected_raw_graphs, ('Graph replay accounting mismatch',len(raw['cuda_graph']),expected_raw_graphs)
    # Conservatively remove any correlated diagnostic GPU tail from every domain.
    excluded=Intervals(diag.parts+[(e['start'],e['end']) for values in diag_events.values() for e in values])
    keep=envelope.subtract(excluded);window_keep=Intervals([(ws,we)]).subtract(excluded)
    assert keep.total>0
    gpu={};table_parts={};all_parts=[];normal_removed=collections.defaultdict(lambda:dict(events=0,masked_ns=0,partially_retained_events=0))
    for table,events in raw.items():
        fragments=[];retained=partial=0;sum_ns=0
        for event in events:
            pieces=keep.clip(event['start'],event['end']);ns=sum(b-a for a,b in pieces)
            in_envelope=envelope.measure(event['start'],event['end'])
            if ns:retained+=1;fragments.extend(pieces);sum_ns+=ns
            if 0<ns<in_envelope:partial+=1
            if event['cid'] in diag_cids:assert ns==0,'Diagnostic GPU work survived common mask'
            elif in_envelope>ns:
                value=normal_removed[table];value['events']+=1;value['masked_ns']+=in_envelope-ns;value['partially_retained_events']+=int(ns>0)
        union=Intervals(fragments);table_parts[table]=union;all_parts.extend(fragments)
        gpu[table]=dict(raw_events_overlapping_profile=len(events),retained_event_count=retained,partially_retained_events=partial,sum_s=sum_ns/1e9,union_s=union.total/1e9,diagnostic_enqueued_events=len(diag_events[table]))
    coverage=Intervals(all_parts);uncovered=keep.subtract(coverage)
    raw_window_coverage=Intervals((a,b) for events in raw.values() for e in events for a,b in window_keep.clip(e['start'],e['end']))
    scopes=scope_records(rows,keep,coverage);groups=collections.defaultdict(list)
    for r in scopes:
        if r['kept_ns']:groups[r['name']].append(r)
    cpu=[]
    for name,group in groups.items():
        durations=[r['kept_ns']/1e6 for r in group]
        cpu.append(dict(name=name,retained_count=len(group),inclusive_host_s=sum(r['kept_ns'] for r in group)/1e9,exclusive_host_s=sum(r['exclusive_ns'] for r in group)/1e9,inclusive_gpu_overlap_s=sum(r['overlap_ns'] for r in group)/1e9,exclusive_gpu_overlap_s=sum(r['exclusive_overlap_ns'] for r in group)/1e9,exclusive_gpu_uncovered_s=sum(r['exclusive_ns']-r['exclusive_overlap_ns'] for r in group)/1e9,p50_retained_ms=quantile(durations,.5),p95_retained_ms=quantile(durations,.95),p99_retained_ms=quantile(durations,.99)))
    cpu.sort(key=lambda x:x['inclusive_host_s'],reverse=True);write_json(folder/'cpu_scopes.json',cpu)
    api_group=collections.defaultdict(list)
    for name,ts,dur,tid,cid in apis:
        ns=keep.measure(ts,ts+dur)
        if ns:api_group[name].append(ns)
    api_summary=[dict(name=name,retained_count=len(values),retained_host_s=sum(values)/1e9,mean_retained_us=sum(values)/len(values)/1e3) for name,values in api_group.items()]
    api_summary.sort(key=lambda x:x['retained_host_s'],reverse=True);write_json(folder/'cuda_api_host_durations.json',api_summary)
    # Kernel rows overlapping opaque Graph spans cannot be double-counted as non-Graph.
    outside_graph=keep.subtract(table_parts['cuda_graph']);hotspots=collections.defaultdict(list)
    for event in raw['cuda_kernel']:
        ns=outside_graph.measure(event['start'],event['end'])
        if ns:hotspots[event['name']].append(ns)
    top=[dict(name=name,retained_count=len(values),gpu_sum_s=sum(values)/1e9,mean_retained_ms=sum(values)/len(values)/1e6,max_retained_ms=max(values)/1e6) for name,values in hotspots.items()]
    top.sort(key=lambda x:x['gpu_sum_s'],reverse=True);write_json(folder/'non_graph_kernel_hotspots.json',top)
    phases,phase_rows=phases_for(worker,rows,keep);write_json(folder/'worker_phase_steps.json',phase_rows)
    for r in diagnostics:
        matched=[x for x in phase_rows if x['start_ns']<=r['start'] and r['end']<=x['end_ns']]
        assert len(matched)==1 and matched[0]['step'] in sample_steps
    prompt_sizes=sorted({r['prompt_token_count'] for s in worker['scheduled_steps'] for r in s['new_requests']});assert len(prompt_sizes)==1
    exclusion=dict(outer_sample_ranges=diagnostics,outer_sample_union_s=diag.total/1e9,common_excluded_ranges_ns=excluded.parts,excluded_from_active_window_s=((end-start)-keep.total)/1e9,diagnostic_api_count=diag_api_count,diagnostic_gpu_events=diag_events,diagnostic_gpu_extensions=extensions,normal_gpu_work_overlapping_exclusion=dict(normal_removed),retained_ranges_ns=keep.parts)
    write_json(folder/'diagnostic_exclusions.json',exclusion)
    summary=dict(epoch=epoch,pid=pid,prompt_tokens=prompt_sizes[0],diagnostic_only=True,official_score=False,active_window=dict(start_ns=start,end_ns=end,raw_duration_s=(end-start)/1e9,retained_duration_s=keep.total/1e9,definition='first EngineCore step through final scoped host return, clipped by the same diagnostic exclusion mask in every domain'),profile_envelope=dict(start_ns=ws,end_ns=we,raw_duration_s=(we-ws)/1e9,retained_duration_s=window_keep.total/1e9,gpu_union_s=raw_window_coverage.total/1e9,gpu_coverage_fraction=raw_window_coverage.total/window_keep.total),diagnostic_samples=len(diagnostics),diagnostic_gpu_extension_count=len(extensions),numeric_samples_exact=True,scope_errors=0,worker_steps=len(worker['scheduled_steps']),worker_phase_counts=dict(phases),gpu=gpu,gpu_union_s=coverage.total/1e9,gpu_union_fraction=coverage.total/keep.total,gpu_uncovered_s=uncovered.total/1e9,gpu_uncovered_intervals=dict(count=len(uncovered.parts),max_ms=max((b-a for a,b in uncovered.parts),default=0)/1e6,p99_ms=quantile([(b-a)/1e6 for a,b in uncovered.parts],.99)),graph_replays_metadata=sum(worker['graph_replay_counts'].values()),retained_kernel_graph_overlap_s=(table_parts['cuda_kernel'].total-table_parts['cuda_kernel'].subtract(table_parts['cuda_graph']).total)/1e9,caveats=['Host scope/API durations are elapsed host time, including GPU waiting, not CPU utilization or CPU overhead.','Uncovered means no recorded GPU activity; it does not establish a CPU cause.','The common mask also clips normal asynchronous work overlapping diagnostic sampling; it is not an uninstrumented counterfactual.','Counts and token totals describe all original scheduled steps; retained host times exclude diagnostic intervals.','Graph intervals retain complete opaque spans; missing internal kernel rows are never treated as GPU idle.','Graph node event samples are analyzed separately and are not multiplied into whole-round kernel totals.','All results are diagnostic; profiler and metadata overhead remain.'])
    write_json(folder/'summary.json',summary)
    return summary,cpu,top


def main():
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--root',type=Path);p.add_argument('--out',type=Path);p.add_argument('--completion',type=Path)
    args=p.parse_args()
    if args.self_test:self_test();return
    if not args.root or not args.out or not args.completion:p.error('--root, --out and --completion required')
    completion=json.loads(args.completion.read_text())
    assert completion.get('report_saved') is True and completion.get('collector_exited') is True,'Collector completion evidence required'
    trace=args.root/'baseline.ixsys';before=trace.stat();args.out.mkdir(exist_ok=False)
    c=sqlite3.connect(trace.as_uri()+'?mode=ro&immutable=1',uri=True)
    assert c.execute('PRAGMA quick_check').fetchone()==('ok',)
    windows=[(path,json.loads(path.read_text())) for path in sorted((args.root/'observations').glob('worker-*-window-*.json'))]
    assert len(windows)==2 and len({w['pid'] for _,w in windows})==1
    windows.sort(key=lambda x:x[1]['epoch'])
    assert windows[0][1]['finished_monotonic_ns']<windows[1][1]['started_monotonic_ns']
    summaries=[];report=['# v2 whole-round CPU/GPU diagnostic analysis','','Identical exclusion intervals are applied to GPU events, NVTX scopes, host APIs and the elapsed-time denominator. Host waiting is not reported as CPU overhead.','']
    for path,worker in windows:
        summary,cpu,top=analyze_window(c,worker,args.out/f"window-{worker['epoch']}");summaries.append(summary)
        wait=next((r for r in cpu if r['name']=='engine.wait_for_batch_queue'),None)
        report += [f"## {summary['prompt_tokens']} tokens / window {worker['epoch']}",'',f"Original active envelope {summary['active_window']['raw_duration_s']:.6f}s; retained {summary['active_window']['retained_duration_s']:.6f}s after {summary['diagnostic_samples']} diagnostic samples. GPU union {summary['gpu_union_s']:.6f}s ({summary['gpu_union_fraction']*100:.4f}%); no recorded GPU activity {summary['gpu_uncovered_s']:.6f}s.",'','| GPU event type | Retained events | Duration sum (s) | Union (s) |','|---|---:|---:|---:|']
        report += [f"| {name} | {v['retained_event_count']} | {v['sum_s']:.6f} | {v['union_s']:.6f} |" for name,v in summary['gpu'].items()]
        report+=['','| Host scope | Inclusive (s) | Exclusive (s) | Exclusive with no recorded GPU activity (s) |','|---|---:|---:|---:|']
        report += [f"| {v['name']} | {v['inclusive_host_s']:.6f} | {v['exclusive_host_s']:.6f} | {v['exclusive_gpu_uncovered_s']:.6f} |" for v in cpu[:18]]
        report+=['','| Non-Graph kernel | Retained events | GPU duration (s) | Mean retained (ms) |','|---|---:|---:|---:|']
        report += [f"| {v['name']} | {v['retained_count']} | {v['gpu_sum_s']:.6f} | {v['mean_retained_ms']:.6f} |" for v in top[:12]]
        report+=['']
    assert sum(s['diagnostic_samples'] for s in summaries)==8
    c.close();after=trace.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Source trace changed'
    report+=['## Limits','',*['- '+x for x in summaries[0]['caveats']],'','The raw trace remains unchanged; per-window JSON retains clipping evidence, CPU scopes, API durations, phase metadata and non-Graph kernel distributions.']
    (args.out/'report.md').write_text('\n'.join(report)+'\n');write_json(args.out/'summary.json',summaries)
    write_json(args.out/'identity.json',dict(trace=str(trace),trace_size=before.st_size,trace_sha256=file_hash(trace),script=str(Path(__file__).resolve()),script_sha256=file_hash(Path(__file__)),completion=completion,worker_files=[dict(path=str(path),sha256=file_hash(path)) for path,_ in windows]))
    print(json.dumps([dict(window=s['epoch'],prompt_tokens=s['prompt_tokens'],retained_s=s['active_window']['retained_duration_s'],gpu_union_s=s['gpu_union_s'],gpu_fraction=s['gpu_union_fraction']) for s in summaries],indent=2))

if __name__=='__main__':main()
