#!/usr/bin/env python3
"""Owner-run M8/O v2: nondefault capture, explicit dependencies, selected-stream replay."""
import argparse,collections,hashlib,inspect,json,os,socket,statistics,subprocess,sys,time,traceback
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(a):
    if not a.run_gpu:
        print(json.dumps(dict(routes=a.routes,streams=a.streams,inputs=2,eager_repeats=a.repeats,graph_repeats=a.graph_repeats,timing=dict(rounds=3,graph_unroll=16,replays=16),capture_stream='dedicated nondefault',replay_stream='selected default or side',rows=8,n=2048,k=2048)));return
    for key in ('source','commit','library','library_sha256','fixture','out'):
        if getattr(a,key) is None:raise ValueError('--'+key.replace('_','-')+' required')
    assert subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()==a.commit
    assert sha(a.library)==a.library_sha256
    with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',9031))!=0,'service running'
    assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True),'GPU busy'
    a.out.mkdir(parents=True,exist_ok=False)
    sys.path[:0]=[str(a.source.resolve()),str(a.source.resolve()/'tools')]
    import torch
    import flag_gems
    from safetensors import safe_open
    from vllm_fl.utils import use_flaggems,get_flag_gems_whitelist_blacklist
    from torch.torch_version import TorchVersion
    wl,bl=get_flag_gems_whitelist_blacklist();assert use_flaggems() and not wl and not bl
    baseline_file=Path(inspect.getsourcefile(inspect.unwrap(flag_gems.mm_out)))
    assert sha(baseline_file)=='4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090'
    with torch.serialization.safe_globals([TorchVersion]):fixture=torch.load(a.fixture,map_location='cpu',weights_only=True)
    torch.set_num_threads(2);flag_gems.enable(record=False,once=True)
    os.environ['BIV150_NATIVE_BF16_LIBRARY']=str(a.library.resolve())
    from vllm_fl.patches.biv150_native_bf16_prefill import load_library
    load_library();functional=torch.ops.biv150_native_bf16_probe.linear;native_out=torch.ops.biv150_native_bf16_probe.linear_out
    weight=None
    for path in sorted(a.model.glob('*.safetensors')):
        with safe_open(str(path),framework='pt',device='cpu') as h:
            if fixture['weight_key'] in h.keys():weight=h.get_tensor(fixture['weight_key']);break
    assert weight is not None and tuple(weight.shape)==(2048,2048)
    assert hashlib.sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest()==fixture['weight_sha256']
    result=dict(diagnostic_only=True,official_score=False,source_commit=a.commit,library_sha256=sha(a.library),script_sha256=sha(__file__),fixture_sha256=sha(a.fixture),routes=a.routes,streams=a.streams,records=[],completed=False,errors=[],rng_reconstruction_caveat=fixture['reconstruction']['caveat'])
    def save():
        tmp=a.out/'result.json.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(a.out/'result.json')
    def compare(value,reference):
        value=value.double();reference=reference.double();d=(value-reference).abs();bad=(d>0.005+0.005*reference.abs())|~torch.isfinite(value)
        return dict(finite=bool(torch.isfinite(value).all()),exact=bool(torch.equal(value,reference)),differences=int((value!=reference).sum()),violations=int(bad.sum()),max_abs=float(d.max()))
    def tensor_sha(value):return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
    save();started=time.monotonic()
    with torch.inference_mode():
        w=weight.cuda();x=torch.empty((8,2048),device='cuda',dtype=torch.bfloat16);out=torch.empty_like(x)
        torch.cuda.synchronize();default=torch.cuda.default_stream();side=torch.cuda.Stream();capture_stream=torch.cuda.Stream()
        assert capture_stream.cuda_stream!=default.cuda_stream
        side.wait_stream(default);capture_stream.wait_stream(default)
        for stream_name in a.streams:
            stream=default if stream_name=='default' else side
            for route in a.routes:
                record=dict(route=route,stream=stream_name,capture_replay=dict(default_stream_handle=int(default.cuda_stream),capture_stream_handle=int(capture_stream.cuda_stream),replay_stream_handle=int(stream.cuda_stream),capture_is_nondefault=True,dependency_policy='replay stream synchronize; capture waits replay; capture synchronize; replay waits capture'),eager=[],graph=[],timing_ms=[],errors=[]);result['records'].append(record);raw=[]
                def call():
                    if route=='baseline':return torch.mm(x,w.t(),out=out)
                    if route=='functional':return functional(x,w)
                    return native_out(x,w,out)
                try:
                    for pattern,host in enumerate(fixture['inputs']):
                        reference=fixture['oracles_fp64'][pattern]
                        for index in range(a.repeats):
                            with torch.cuda.stream(stream):
                                x.copy_(host,non_blocking=False);out.fill_(float('nan'));value=call()
                            stream.synchronize();actual=value.cpu().clone()
                            record['eager'].append(dict(input=pattern,index=index,cpu_fp64=compare(actual,reference),cpu_rounded=compare(actual,reference.to(torch.bfloat16)),output_sha256=tensor_sha(actual)))
                            raw.append(dict(stage='eager',input=pattern,index=index,output=actual))
                        save()
                    stream.synchronize();capture_stream.wait_stream(stream)
                    with torch.cuda.stream(capture_stream):
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph,stream=capture_stream):graph_out=call()
                    capture_stream.synchronize();stream.wait_stream(capture_stream)
                    for pattern,host in enumerate(fixture['inputs']):
                        reference=fixture['oracles_fp64'][pattern]
                        for index in range(a.graph_repeats):
                            with torch.cuda.stream(stream):
                                x.copy_(host,non_blocking=False);graph_out.fill_(float('nan'));graph.replay()
                            stream.synchronize();actual=graph_out.cpu().clone()
                            with torch.cuda.stream(stream):eager=call()
                            stream.synchronize();eager_cpu=eager.cpu().clone()
                            record['graph'].append(dict(input=pattern,index=index,cpu_fp64=compare(actual,reference),graph_eager=compare(actual,eager_cpu),output_sha256=tensor_sha(actual)))
                            raw.append(dict(stage='graph',input=pattern,index=index,output=actual,eager=eager_cpu))
                    # Sixteen operations per captured graph avoid Python replay gaps dominating tiny M8 kernels.
                    with torch.cuda.stream(stream):
                        x.copy_(fixture['inputs'][0],non_blocking=False)
                    stream.synchronize();capture_stream.wait_stream(stream)
                    with torch.cuda.stream(capture_stream):
                        timing_graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(timing_graph,stream=capture_stream):
                            for _ in range(16):timing_out=call()
                    capture_stream.synchronize();stream.wait_stream(capture_stream)
                    with torch.cuda.stream(stream):
                        timing_graph.replay();timing_graph.replay()
                    stream.synchronize()
                    for _ in range(3):
                        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                        with torch.cuda.stream(stream):
                            begin.record(stream)
                            for _ in range(16):timing_graph.replay()
                            end.record(stream)
                        end.synchronize();record['timing_ms'].append(begin.elapsed_time(end)/256)
                    torch.save(raw,a.out/f'outputs-{route}-{stream_name}.pt')
                    record['raw_outputs_sha256']=sha(a.out/f'outputs-{route}-{stream_name}.pt')
                    record['all_finite']=all(row['cpu_fp64']['finite'] for row in record['eager']+record['graph'])
                    record['cpu_violation_calls']=sum(row['cpu_fp64']['violations']>0 for row in record['eager']+record['graph'])
                    record['all_graph_eager_exact']=all(row['graph_eager']['exact'] for row in record['graph'])
                    record['unique_outputs_per_input']={str(pattern):len({row['output_sha256'] for row in record['eager']+record['graph'] if row['input']==pattern}) for pattern in (0,1)}
                    record['median_kernel_ms']=statistics.median(record['timing_ms'])
                    record['timing_note']='Warm repeated component Graph timing; not cold-weight or model speedup. Baseline/native_out use the same caller output; functional allocates inside capture.'
                    del graph,graph_out,timing_graph,timing_out,value,eager
                except Exception as error:record['errors'].append(dict(error=repr(error),traceback=traceback.format_exc()))
                if stream.cuda_stream!=default.cuda_stream:default.wait_stream(stream)
                save()
                print(json.dumps({k:record.get(k) for k in ['route','stream','all_finite','cpu_violation_calls','all_graph_eager_exact','unique_outputs_per_input','median_kernel_ms','errors']}),flush=True)
    result['completed']=True;result['elapsed_seconds']=time.monotonic()-started;save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-gpu',action='store_true');p.add_argument('--source',type=Path);p.add_argument('--commit');p.add_argument('--library',type=Path);p.add_argument('--library-sha256');p.add_argument('--fixture',type=Path);p.add_argument('--out',type=Path);p.add_argument('--model',type=Path,default=Path('/workspace/MiniCPM5-2B'))
    p.add_argument('--routes',nargs='+',choices=['baseline','functional','native_out'],default=['native_out','baseline','functional']);p.add_argument('--streams',nargs='+',choices=['default','side'],default=['default','side']);p.add_argument('--repeats',type=int,default=32);p.add_argument('--graph-repeats',type=int,default=8)
    a=p.parse_args()
    if not 1<=a.repeats<=256 or not 1<=a.graph_repeats<=64:p.error('repeats must be 1..256 and graph-repeats 1..64')
    if len(set(a.routes))!=len(a.routes) or len(set(a.streams))!=len(a.streams):p.error('routes and streams must be distinct')
    main(a)
