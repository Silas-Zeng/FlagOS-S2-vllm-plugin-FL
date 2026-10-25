#!/usr/bin/env python3
"""Bounded functional BF16 / dynamic-M / real-weight-ring validation.

Import and --dry-run use only the standard library. GPU execution requires the
owner's explicit --run-gpu flag, an idle device, exact source and SO identities.
Numerical discrepancies are retained independently of diagnostic ring timing.
"""
import argparse,ast,hashlib,importlib.util,inspect,json,math,os,socket,statistics,subprocess,sys,time,traceback
from pathlib import Path

GEOMETRIES={'qkv':(2560,2048),'o':(2048,2048),'down':(2048,6144),'gate_up':(12288,2048)}
LAYERS=(4,20,37,0,41)
EDGES=(0,1,2,3,38,39,40,41)
ROWS=(1,8,16,32,40,64,2048)
DYNAMIC_ROWS=(32,64,2048,2030,8)
BASELINE_SHA='4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090'


def ring_keys():
    return [(layer,kind) for layer in range(42) for kind in ('qkv','o','gate_up','down') if kind!='gate_up' or layer in EDGES]


def plan(layers=LAYERS,gate_layers=(0,41)):
    keys=ring_keys();representative=[(layer,kind) for layer in layers for kind in ('qkv','o','down')]+[(layer,'gate_up') for layer in gate_layers]
    return dict(representative_weights=representative,numeric_cases=len(representative)*len(ROWS),rows=list(ROWS),dynamic_rows=list(DYNAMIC_ROWS),ring_weights=len(keys),ring_order=keys,ring_weight_bytes=sum(2*math.prod(GEOMETRIES[kind]) for _,kind in keys),cpu_fp64_products=2*len(representative),cpu_fp64_flops=2*sum(2*2048*math.prod(GEOMETRIES[kind]) for _,kind in representative),ring_round_orders=[['baseline','native'],['native','baseline'],['baseline','native']],default_replays_per_round=10)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def safe_number(value):
    value=float(value)
    return value if math.isfinite(value) else None


def save_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def difference(actual,reference,path,*,save_exact=False):
    """All threshold violations (and optional exact differences) retain raw values."""
    a=actual.detach().cpu().double();r=reference.detach().cpu().double()
    assert a.shape==r.shape
    delta=(a-r).abs();finite=torch.isfinite(a);ref_finite=torch.isfinite(r)
    unequal=(a!=r)|~finite|~ref_finite
    bad=(delta>0.005+0.005*r.abs())|~finite|~ref_finite
    result=dict(finite=bool(finite.all()),reference_finite=bool(ref_finite.all()),elements=a.numel(),numeric_exact=bool(torch.equal(a,r)),numeric_differences=int(unequal.sum()),tolerance_passed=not bool(bad.any()),violations=int(bad.sum()),atol=0.005,rtol=0.005,max_abs=safe_number(delta.max()),rms_abs=safe_number(delta.square().mean().sqrt()),relative_l2=safe_number(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(r).clamp_min(1e-30)))
    retain=unequal if save_exact else bad
    if bool(retain.any()):
        indices=retain.flatten().nonzero().flatten()
        torch.save(dict(shape=tuple(a.shape),flat_indices=indices,actual=a.flatten()[indices],reference=r.flatten()[indices],selection='all unequal' if save_exact else 'all tolerance violations/nonfinite'),path)
        result['raw_differences']=dict(file='differences/'+path.name,sha256=sha(path),count=indices.numel())
    return result


def capture_pair(x,weight,baseline_out,functional,stream):
    with torch.cuda.stream(stream):
        torch.mm(x,weight.t(),out=baseline_out);functional(x,weight)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            torch.mm(x,weight.t(),out=baseline_out)
            candidate_out=functional(x,weight)
    stream.synchronize()
    return graph,candidate_out


def numeric_case(entry,m,functional,native_out,folder):
    layer,kind=entry['key'];weight=entry['device'];host=entry['host'];n,k=host.shape
    record=dict(layer=layer,geometry=kind,m=m,n=n,k=k,patterns=[],graph_checks=[],errors=[])
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    host_inputs=[torch.randn((m,k),dtype=torch.bfloat16),torch.randn((m,k),dtype=torch.bfloat16).mul_(-0.75)]
    host_inputs[1][:,k//2]=3
    references=[None,None]
    if m==2048:
        w64=host.double()
        references=[x.double()@w64.t() for x in host_inputs]
        del w64
    x=torch.empty((m,k),dtype=torch.bfloat16,device='cuda');baseline_out=torch.empty((m,n),dtype=torch.bfloat16,device='cuda')
    stem=f'l{layer}-{kind}-m{m}'
    for index,source in enumerate(host_inputs):
        with torch.cuda.stream(stream):
            x.copy_(source);baseline_out.fill_(float('nan'))
            torch.mm(x,weight.t(),out=baseline_out);candidate=functional(x,weight)
        stream.synchronize();b=baseline_out.cpu().clone();c=candidate.cpu()
        item=dict(input=index,baseline_vs_functional=difference(c,b,folder/f'{stem}-a{index}-direct.pt',save_exact=True))
        if references[index] is not None:
            reference=references[index]
            item['baseline_cpu_fp64']=difference(b,reference,folder/f'{stem}-a{index}-baseline-fp64.pt')
            item['functional_cpu_fp64']=difference(c,reference,folder/f'{stem}-a{index}-functional-fp64.pt')
            item['functional_cpu_fp64_rounded_bf16']=difference(c,reference.to(torch.bfloat16),folder/f'{stem}-a{index}-functional-rounded.pt')
            with torch.cuda.stream(stream):native_out(x,weight,baseline_out)
            stream.synchronize()
            item['functional_vs_native_out_exact']=difference(baseline_out.cpu(),c,folder/f'{stem}-a{index}-out-vs-functional.pt',save_exact=True)
        record['patterns'].append(item)
        del candidate,b,c
    graph,graph_out=capture_pair(x,weight,baseline_out,functional,stream)
    for index,source in enumerate(host_inputs):
        with torch.cuda.stream(stream):
            x.copy_(source);baseline_out.fill_(float('nan'));graph_out.fill_(float('nan'));graph.replay()
        stream.synchronize();gb=baseline_out.cpu().clone();gc=graph_out.cpu().clone()
        with torch.cuda.stream(stream):
            baseline_out.fill_(float('nan'));torch.mm(x,weight.t(),out=baseline_out);eager=functional(x,weight)
        stream.synchronize()
        record['graph_checks'].append(dict(input=index,baseline_graph_vs_eager=difference(gb,baseline_out.cpu(),folder/f'{stem}-a{index}-baseline-graph.pt',save_exact=True),functional_graph_vs_eager=difference(gc,eager.cpu(),folder/f'{stem}-a{index}-functional-graph.pt',save_exact=True),graph_vs_baseline=difference(gc,gb,folder/f'{stem}-a{index}-graph-direct.pt',save_exact=True)))
        del eager,gb,gc
    record['graph_eager_exact']=all(c['baseline_graph_vs_eager']['numeric_exact'] and c['functional_graph_vs_eager']['numeric_exact'] for c in record['graph_checks'])
    record['non2048_fallback_exact']=(all(p['baseline_vs_functional']['numeric_exact'] for p in record['patterns']) if m!=2048 else None)
    record['cpu_fp64_complete']=all('functional_cpu_fp64' in p for p in record['patterns']) if m==2048 else None
    torch.cuda.current_stream().wait_stream(stream)
    return record


def dynamic_case(entry,functional,folder):
    # The closed-over nn.Parameter mirrors the service's static projection weight.
    weight=entry['device'];assert isinstance(weight,torch.nn.Parameter) and not weight.is_inference()
    n,k=weight.shape;compiles=[];calls=[]
    from torch._dynamo.backends.registry import lookup_backend
    inductor=lookup_backend('inductor')
    def backend(gm,example_inputs):
        compiles.append(dict(index=len(compiles)+1,fx_code=gm.code,input_types=[type(v).__name__ for v in example_inputs]))
        return inductor(gm,example_inputs)
    def projection(x):
        return functional(x,weight)
    torch._dynamo.reset()
    compiled=torch.compile(projection,backend=backend,dynamic=True,fullgraph=True)
    for m in DYNAMIC_ROWS:
        x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)
        torch._dynamo.mark_dynamic(x,0)
        actual=compiled(x);expected=functional(x,weight);torch.cuda.synchronize()
        calls.append(dict(m=m,compile_count=len(compiles),comparison=difference(actual,expected,folder/f'dynamic-{entry["key"][1]}-m{m}.pt',save_exact=True)))
    return dict(geometry=entry['key'][1],weight_is_parameter=True,weight_shape=list(weight.shape),weight_closed_over=True,mark_dynamic_dimension=0,dynamic=True,fullgraph=True,sequence=list(DYNAMIC_ROWS),compiles=compiles,calls=calls,single_graph_all_rows=len(compiles)==1 and all(c['compile_count']==1 for c in calls),numeric_exact=all(c['comparison']['numeric_exact'] for c in calls))


def ring_check(entries,functional,native_out,folder,replays,save_progress):
    # Exactly the same static input/output addresses are used by both graphs.
    inputs={k:torch.randn((2048,k),device='cuda',dtype=torch.bfloat16) for k in {GEOMETRIES[kind][1] for _,kind in ring_keys()}}
    outputs={n:torch.empty((2048,n),device='cuda',dtype=torch.bfloat16) for n in {GEOMETRIES[kind][0] for _,kind in ring_keys()}}
    result=dict(weight_count=len(entries),weight_bytes=sum(e['device'].numel()*e['device'].element_size() for e in entries),rows=2048,replays_per_round=replays,rounds=[],per_weight=[],changed_input_endpoints=[],errors=[],timing_is_model_speedup=False,baseline='FlagGems mm.out',candidate='native linear_out; only M2048',reused_input_bytes=sum(t.numel()*t.element_size() for t in inputs.values()),reused_output_bytes=sum(t.numel()*t.element_size() for t in outputs.values()),output_addresses={str(n):t.data_ptr() for n,t in outputs.items()})
    assert result['weight_count']==134 and result['weight_bytes']==plan()['ring_weight_bytes']
    for entry in entries:
        layer,kind=entry['key'];weight=entry['device'];n,k=weight.shape;x=inputs[k];out=outputs[n]
        out.fill_(float('nan'));torch.mm(x,weight.t(),out=out);torch.cuda.synchronize();baseline=out.cpu().clone()
        out.fill_(float('nan'));native_out(x,weight,out);torch.cuda.synchronize();candidate=out.cpu().clone()
        functional_output=functional(x,weight);torch.cuda.synchronize()
        result['per_weight'].append(dict(layer=layer,geometry=kind,baseline_native=difference(candidate,baseline,folder/f'ring-l{layer}-{kind}-direct.pt',save_exact=True),functional_out=difference(candidate,functional_output.cpu(),folder/f'ring-l{layer}-{kind}-functional-out.pt',save_exact=True)))
        del baseline,candidate,functional_output
        save_progress(result)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    def execute(native):
        for entry in entries:
            weight=entry['device'];n,k=weight.shape
            if native:native_out(inputs[k],weight,outputs[n])
            else:torch.mm(inputs[k],weight.t(),out=outputs[n])
    graphs={}
    with torch.cuda.stream(stream):
        for label,native in [('baseline',False),('native',True)]:
            execute(native);graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):execute(native)
            graphs[label]=graph
    stream.synchronize()
    for index,factor in enumerate((0.5,-1.0)):
        with torch.cuda.stream(stream):
            for x in inputs.values():x.normal_().mul_(factor)
            for out in outputs.values():out.fill_(float('nan'))
            graphs['baseline'].replay()
        stream.synchronize();expected={n:x.cpu().clone() for n,x in outputs.items()}
        with torch.cuda.stream(stream):
            for out in outputs.values():out.fill_(float('nan'))
            graphs['native'].replay()
        stream.synchronize()
        result['changed_input_endpoints'].append(dict(input=index,scope='last writer for each reused output width; individual weights were checked eagerly',comparisons={str(n):difference(out.cpu(),expected[n],folder/f'ring-endpoint-a{index}-n{n}.pt',save_exact=True) for n,out in outputs.items()}))
    with torch.cuda.stream(stream):
        for x in inputs.values():x.normal_()
        for graph in graphs.values():
            graph.replay();graph.replay()
    stream.synchronize()
    for round_index,order in enumerate(plan()['ring_round_orders']):
        record=dict(index=round_index,order=order)
        for label in order:
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                start.record(stream)
                for _ in range(replays):graphs[label].replay()
                end.record(stream)
            end.synchronize();record[label+'_total_ms']=start.elapsed_time(end);record[label+'_ring_ms']=record[label+'_total_ms']/replays
        record['baseline_over_native_ring_ratio']=record['baseline_ring_ms']/record['native_ring_ms'];result['rounds'].append(record);save_progress(result)
    result['median_ring_ratio']=statistics.median(r['baseline_over_native_ring_ratio'] for r in result['rounds'])
    result['timing_completed']=True
    torch.cuda.current_stream().wait_stream(stream)
    return result


def main(args):
    specification=plan(args.layers,args.gate_layers)
    if args.dry_run or not args.run_gpu:
        assert specification['numeric_cases']==(3*len(args.layers)+len(args.gate_layers))*len(ROWS) and specification['ring_weights']==134 and specification['ring_weight_bytes']==2252341248
        assert len(set(ring_keys()))==134 and all(layer in EDGES for layer,kind in ring_keys() if kind=='gate_up')
        assert specification['ring_round_orders']==[['baseline','native'],['native','baseline'],['baseline','native']]
        print(json.dumps(specification,indent=2));return
    for name in ('source','commit','library','library_sha256','out'):
        if getattr(args,name) is None:raise ValueError('--'+name.replace('_','-')+' required for GPU run')
    source=args.source.resolve();library=args.library.resolve()
    assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==args.commit
    assert sha(library)==args.library_sha256
    with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',9031))!=0,'Service 9031 is running'
    idle=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
    assert 'No running processes found' in idle,'GPU is busy; owner must schedule this probe'
    args.out.mkdir(parents=True,exist_ok=False);differences=args.out/'differences';differences.mkdir()
    sys.path[:0]=[str(source),str(source/'tools')]
    global torch
    import torch
    import flag_gems
    from vllm_fl.utils import use_flaggems,get_flag_gems_whitelist_blacklist
    whitelist,blacklist=get_flag_gems_whitelist_blacklist()
    assert use_flaggems() and not whitelist and not blacklist,'Reaudit worker FlagGems policy'
    import vllm_fl
    assert Path(vllm_fl.__file__).resolve().is_relative_to(source)
    baseline_file=Path(inspect.getsourcefile(inspect.unwrap(flag_gems.mm_out)));assert sha(baseline_file)==BASELINE_SHA
    torch.set_num_threads(args.cpu_threads);torch.manual_seed(20261025)
    flag_gems.enable(record=False,once=True)
    os.environ['BIV150_NATIVE_BF16_LIBRARY']=str(library)
    from vllm_fl.patches.biv150_native_bf16_prefill import load_library
    load_library()  # Reuse the exact production fake/meta implementation.
    functional=torch.ops.biv150_native_bf16_probe.linear;native_out=torch.ops.biv150_native_bf16_probe.linear_out
    loader_file=source/'tools/bench_biv150_native_bf16_prefill.py'
    spec=importlib.util.spec_from_file_location('_native_bf16_weight_loader',loader_file);loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader);loader.torch=torch
    source_hashes={name:sha(source/name) for name in ['csrc/iluvatar/native_bf16_probe.cpp','vllm_fl/patches/biv150_native_bf16_prefill.py','vllm_fl/compile_semantics.py','vllm_fl/utils.py']}
    result=dict(diagnostic_only=True,official_score=False,precision_gate=False,source_hashes=source_hashes,source=str(source),source_commit=args.commit,library=str(library),library_sha256=sha(library),script_sha256=sha(__file__),loader_sha256=sha(loader_file),flaggems_mm_sha256=sha(baseline_file),dispatch_table=torch._C._dispatch_dump_table('aten::mm.out'),plan=specification,requested_stages=args.stages,precision_policy='All discrepancies are retained. Tolerance failures do not automatically suppress ring timing or establish service acceptance.',oracle='M2048: both inputs, full CPU FP64 and rounded BF16; other M: exact actual FlagGems fallback',numeric=[],dynamic=[],ring=None,weights=[],stage_seconds={},errors=[],completed=False)
    save=lambda:save_json(args.out/'result.json',result)
    save();started=time.monotonic();entries={}
    try:
        with torch.inference_mode():
            for layer,kind,host,identities in loader.weights(args.model,list(range(42)),list(EDGES)):
                key=(layer,kind);assert key not in entries and tuple(host.shape)==GEOMETRIES[kind]
                with torch.inference_mode(False):
                    parameter=torch.nn.Parameter(host.cuda(),requires_grad=False)
                entries[key]=dict(key=key,host=host,device=parameter,identities=identities)
                result['weights'].append(dict(layer=layer,geometry=kind,shape=list(host.shape),identities=identities))
            assert set(entries)==set(ring_keys());result['stage_seconds']['load']=time.monotonic()-started;save()
            stage=time.monotonic()
            for layer,kind in (specification['representative_weights'] if 'numeric' in args.stages else []):
                for m in ROWS:
                    try:record=numeric_case(entries[(layer,kind)],m,functional,native_out,differences)
                    except Exception as error:record=dict(layer=layer,geometry=kind,m=m,error=repr(error),traceback=traceback.format_exc())
                    result['numeric'].append(record);save()
                    print(json.dumps(dict(stage='numeric',layer=layer,geometry=kind,m=m,error=record.get('error'),graph_eager_exact=record.get('graph_eager_exact'),fallback_exact=record.get('non2048_fallback_exact'))),flush=True)
            result['stage_seconds']['numeric']=time.monotonic()-stage;stage=time.monotonic()
            for kind,layer in ([('qkv',20),('o',20),('down',20),('gate_up',0)] if 'dynamic' in args.stages else []):
                try:record=dynamic_case(entries[(layer,kind)],functional,differences)
                except Exception as error:record=dict(geometry=kind,error=repr(error),traceback=traceback.format_exc())
                result['dynamic'].append(record);save();print(json.dumps(dict(stage='dynamic',geometry=kind,single_graph=record.get('single_graph_all_rows'),error=record.get('error'))),flush=True)
            result['stage_seconds']['dynamic']=time.monotonic()-stage;stage=time.monotonic()
            def save_ring(value):result['ring']=value;save()
            if 'ring' in args.stages:
                try:result['ring']=ring_check([entries[key] for key in ring_keys()],functional,native_out,differences,args.ring_replays,save_ring)
                except Exception as error:
                    result['errors'].append(dict(stage='ring',error=repr(error),traceback=traceback.format_exc()))
            result['stage_seconds']['ring']=time.monotonic()-stage
            result['max_cuda_allocated_bytes']=torch.cuda.max_memory_allocated();result['max_cuda_reserved_bytes']=torch.cuda.max_memory_reserved()
    except Exception as error:result['errors'].append(dict(stage='outer',error=repr(error),traceback=traceback.format_exc()))
    result['total_seconds']=time.monotonic()-started;result['completed']=True
    result['numeric_cases_completed']=len(result['numeric'])
    result['numeric_execution_errors']=sum('error' in r for r in result['numeric'])
    result['dynamic_passed']=(len(result['dynamic'])==4 and all(r.get('single_graph_all_rows') and r.get('numeric_exact') for r in result['dynamic'])) if 'dynamic' in args.stages else None
    save();print(json.dumps(dict(completed=True,numeric_cases=len(result['numeric']),numeric_execution_errors=result['numeric_execution_errors'],dynamic_passed=result['dynamic_passed'],ring_ratio=(result['ring'] or {}).get('median_ring_ratio'),errors=result['errors'],seconds=result['total_seconds'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true');parser.add_argument('--run-gpu',action='store_true')
    parser.add_argument('--source',type=Path);parser.add_argument('--commit');parser.add_argument('--library',type=Path);parser.add_argument('--library-sha256');parser.add_argument('--out',type=Path)
    parser.add_argument('--model',type=Path,default=Path('/workspace/MiniCPM5-2B'));parser.add_argument('--cpu-threads',type=int,default=2);parser.add_argument('--ring-replays',type=int,default=10)
    parser.add_argument('--layers',type=int,nargs='+',default=list(LAYERS));parser.add_argument('--gate-layers',type=int,nargs='*',default=[0,41])
    parser.add_argument('--stages',nargs='+',choices=['numeric','dynamic','ring'],default=['numeric','dynamic','ring'])
    args=parser.parse_args()
    if len(set(args.layers))!=len(args.layers) or not all(0<=x<42 for x in args.layers):parser.error('layers must be distinct values in 0..41')
    if len(set(args.gate_layers))!=len(args.gate_layers) or not all(x in EDGES for x in args.gate_layers):parser.error('gate-layers must be distinct BF16 edge layers')
    if args.cpu_threads<1 or not 1<=args.ring_replays<=20:parser.error('cpu-threads >= 1 and ring-replays in 1..20 required')
    main(args)
