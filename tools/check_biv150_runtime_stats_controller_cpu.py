"""Pinned controller inverse plus opaque HTTP/owner/benchmark CPU seam checks."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from unittest.mock import patch

import biv150_runtime_stats_controller as c
import build_biv150_runtime_stats_controller as b

ROOT=Path(__file__).resolve().parents[1]
PARENT=Path('/private/tmp/cjj-native117-parent107-controller-20261029.py')
GENERATED=ROOT/'tools'/b.OUTPUT
checks={}
def ok(name,value=True):assert value,name;checks[name]=True
def refuse(name,fn):
    try:fn()
    except (ValueError,RuntimeError,FileExistsError,KeyError):ok(name);return
    raise AssertionError(name+' did not reject')
def load(path):
    spec=importlib.util.spec_from_file_location('cpu_controller',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def arguments(out,ref):
    return types.SimpleNamespace(runtime_stats_bootstrap=True,throughput=True,full_workload_warmup=True,
        warmup_plan='full-only',throughput_case='both',baseline=False,q32_hoist=True,native_k8_library=Path('/actual/8b.so'),
        native_k8_sha256=c.NATIVE_SHA,math105=False,supplemental_native_math105_64=False,
        graph_coverage_bootstrap_dir=None,source=ROOT,out=out,commit=ref)

def main(ref,arm):
    global PARENT,GENERATED
    parent_sha=b.PARENT_SHA
    if arm=='B':
        PARENT=Path('/private/tmp/cjj-agent-native117-service-controller-v1-20261029.py');GENERATED=ROOT/'tools'/b.B_OUTPUT;parent_sha=b.B_PARENT_SHA
    entry=str(GENERATED.relative_to(ROOT));binding=c.CONTROLLERS[entry]
    previous='d9c91e011915396754efe8023c817831c733bdae'
    for name in ('tools/biv150_runtime_stats_snapshot.py','tools/runtime_stats_bootstrap/sitecustomize.py','tools/run_biv150_parent107_stats_service.py'):
        ok('historical_bytes_preserved_'+Path(name).name,(ROOT/name).read_bytes()==subprocess.check_output(['git','-C',str(ROOT),'show',previous+':'+name]))
    prior_builder=subprocess.check_output(['git','-C',str(ROOT),'show',previous+':tools/build_biv150_runtime_stats_controller.py'])
    original_patches=next(n.value for n in ast.parse(prior_builder).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='PATCHES')
    ok('same_fourteen_additive_seams',len(b.PATCHES)==14 and b.PATCHES==ast.literal_eval(original_patches))
    command=[sys.executable,str(GENERATED),'--runtime-stats-bootstrap','--throughput','--full-workload-warmup',
        '--warmup-plan','full-only','--q32-hoist','--native-k8-library','/cpu-fixture/native.so','--native-k8-sha256',binding['library_sha256']]
    valid=json.loads(subprocess.check_output(command,text=True))
    ok('actual_parser_own_arm_CPU_plan',valid['runtime_stats_snapshot']['enabled'] and valid['native_library_sha256']==binding['library_sha256'] and valid['gpu_executed'] is False)
    command[-1]=next(sha for sha in c.NATIVE_SHAS if sha!=binding['library_sha256'])
    crossed=subprocess.run(command,text=True,capture_output=True)
    ok('actual_parser_rejects_other_allowed_arm',crossed.returncode!=0 and 'Native K8 requires' in crossed.stderr)
    raw=PARENT.read_bytes();made=GENERATED.read_bytes();ok('parent_SHA',hashlib.sha256(raw).hexdigest()==parent_sha)
    ok('generated_exact',made==b.build(raw));ok('inverse_byte_exact',b.inverse(made)==raw)
    refuse('wrong_parent_rejected',lambda:b.build(raw+b'\n'))
    def top(data):
        text=data.decode();lines=text.splitlines(keepends=True)
        return {n.name:''.join(lines[n.lineno-1:n.end_lineno]) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    old,new=top(raw),top(made)
    unchanged=sorted(set(old)-{'run','full_workload_only_throughput'})
    ok('all_other_functions_and_classes_unchanged',all(old[k]==new[k] for k in unchanged))
    def region(data,kind):
        text=data.decode();run=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='run')
        outer=next(n for n in run.body if isinstance(n,ast.Try));lines=text.splitlines(keepends=True)
        nodes=outer.finalbody if kind=='cleanup' else next(n for n in outer.body if isinstance(n,ast.If) and isinstance(n.test,ast.Attribute) and n.test.attr==kind).body
        return ''.join(lines[nodes[0].lineno-1:nodes[-1].end_lineno])
    for kind in ('cleanup','math105','supplemental_native_math105_64'):ok(kind+'_source_exact',region(raw,kind)==region(made,kind))
    oldplan=json.loads(subprocess.check_output([sys.executable,str(PARENT)],text=True))
    newplan=json.loads(subprocess.check_output([sys.executable,str(GENERATED)],text=True))
    ok('default_OFF_plan_only_adds_disabled_observer',newplan.pop('runtime_stats_snapshot')=={'enabled':False} and newplan==oldplan)
    source=made.decode()
    ok('token_added_after_CPU_preflights_only_inside_original_try',source.index('env=stats_session.service_environment(env)')>source.index('expected = m.expected_fingerprint') and source.index('env=stats_session.service_environment(env)')>source.index('state = dict(completed=False'))
    ok('client_scrub_precedes_original_adapter',source.index('client=stats_hook.clean_environment(client,source)')<source.index('namespace = dict(json=json, subprocess=OwnedBenchmarkAdapter'))
    module=load(GENERATED)
    with tempfile.TemporaryDirectory(prefix='cjj-stats-controller-cpu-') as tmp:
        out=Path(tmp)/'cpu-stats-run';out.mkdir();a=arguments(out,ref);a.native_k8_sha256=binding['library_sha256']
        for key,value in (('throughput',False),('full_workload_warmup',False),('warmup_plan','legacy'),('throughput_case','4096'),('baseline',True),('q32_hoist',False),('native_k8_sha256','0'*64),('math105',True),('supplemental_native_math105_64',True),('graph_coverage_bootstrap_dir',Path('/graph'))):
            bad=copy.copy(a);setattr(bad,key,value);refuse('reject_mode_'+key,lambda:c.plan(bad))
        expected={'konly_encoded':dict(q32_hoist_enabled=True,implementation_sha256='f'*64,native_decode=dict(enabled=True,library_sha256=binding['library_sha256'],operator=binding['operator']))}
        # This calls actual git show on the given frozen commit/tree, not a
        # canned success. It does not replace the original service HEAD/clean gate.
        def identity_refusal(label,arg,semantics,needle,controller=GENERATED):
            try:c.Session(arg,controller,semantics)
            except ValueError as error:ok(label,needle in str(error));return
            raise AssertionError(label+' did not reject')
        other_sha=next(sha for sha in c.NATIVE_SHAS if sha!=a.native_k8_sha256)
        crossed=copy.copy(a);crossed.native_k8_sha256=other_sha
        identity_refusal('controller_rejects_other_allowed_library_arg',crossed,expected,'controller/library argument mismatch')
        bad_c3=copy.deepcopy(expected);bad_c3['konly_encoded']['native_decode']['library_sha256']=other_sha
        identity_refusal('C3_rejects_other_allowed_library',a,bad_c3,'expected active semantics differ')
        bad_op=copy.deepcopy(expected);bad_op['konly_encoded']['native_decode']['operator']='foreign::produce'
        identity_refusal('C3_rejects_foreign_operator',a,bad_op,'expected active semantics differ')
        identity_refusal('unknown_same_source_entry_rejected',a,expected,'unknown controller',ROOT/'tools/unknown_controller.py')
        session=c.Session(a,GENERATED,expected)
        ok('actual_git_object_binding',set(session.hashes)==set(c.FILES))
        refuse('foreign_controller_path',lambda:c.Session(a,Path('/foreign/controller.py'),expected))
        base={'PYTHONPATH':str(ROOT),'KEEP':'original','VLLM_FL_BIV150_KONLY_NATIVE_DECODE':'1',
              'VLLM_FL_BIV150_KONLY_ENCODED_SERVICE':'1','VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST':'1'}
        token='CPU_PRIVATE_TOKEN_'+'x'*32
        with patch.object(c.secrets,'token_urlsafe',return_value=token):service=session.service_environment(base)
        client=c.clean_environment(service,ROOT)
        ok('client_env_exact_original_and_no_token',client==base and token not in json.dumps(client))
        ok('token_only_in_service_env',service[c.snapshot.TOKEN]==token and not any(k in base for k in (c.snapshot.TOKEN,c.snapshot.ENABLE,c.snapshot.RUN_ID)))
        refuse('double_activation',lambda:session.service_environment(base))
        class Owner:
            members={22:dict(pid=22,start_ticks=222,sid=11,state='S')}
            def remember(self):pass
            def read(self,pid):return {11:dict(pid=11,start_ticks=111,sid=11,state='S'),**self.members}.get(pid)
        owner=Owner();server=dict(pid=11,start_ticks=111);events=[];counter=[0];fault=[None]
        def payload(phase):
            return dict(schema=c.snapshot.SCHEMA,phase=phase,run_id=session.run_id,rpc_rank=0,pid=22,start_ticks=222,
              tool_source_sha256=session.hashes[c.FILES[1]],sdk_pins=c.snapshot.SDK_PINS,implementation_sha256=session.implementation,
              service_module=str(ROOT/'vllm_fl/patches/biv150_konly_encoded_service.py'),worker_type='vllm_fl.worker.worker.WorkerFL',runner_type='vllm_fl.worker.model_runner.ModelRunnerFL',
              graph_replays_counted=False,client_invocation_including_preflight=True,monotonic_ns=counter[0],shape_records={'k':{'host_calls':counter[0]}},shape_overflow=0,
              stats=dict(selected={'writer83':counter[0]},native_decode=dict(enabled=True,library=session.library,records=[],selected_host_calls=0,private_Triton_fallbacks={},overflow=0),q32_hoist=dict(enabled=True,admitted_host_calls=0)),q32_geometry_records={})
        class Response:
            status=200
            def __init__(self,request):
                self.url=request.full_url;phase=json.loads(request.data)['phase'];value=payload(phase)
                if fault[0]=='wrong_source':value['implementation_sha256']='0'*64
                if fault[0]=='wrong_library':value['stats']['native_decode']['library']=dict(session.library,sha256=other_sha)
                self.raw=json.dumps(value,sort_keys=True).encode();self.headers={'x-cjj-stats-sha256':hashlib.sha256(self.raw).hexdigest() if fault[0]!='bad_SHA' else '0'*64}
                self.phase=phase
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,limit):events.append('ACK '+self.phase);return self.raw[:limit]
            def geturl(self):return self.url
        class Opener:
            def open(self,request,timeout):
                ok('request_fixed_endpoint_header_and_JSON',request.full_url=='http://127.0.0.1:9031'+c.snapshot.ROUTE and request.get_method()=='POST' and set(json.loads(request.data))=={'phase','run_id'} and token in request.headers.values() and token.encode() not in request.data and timeout==60)
                if fault[0]=='timeout':raise TimeoutError('opaque fixture')
                return Response(request)
        def opener(*handlers):
            ok('proxy_and_redirect_disabled',handlers[0].proxies=={} and handlers[1].redirect_request(None,None,None,None,None,None) is None)
            return Opener()
        def benchmark(folder,label,length,output,count,seed):
            expected_ack=f'warm-before-{length}' if label.startswith('full-warm-') else f'warm-after-{length}'
            ok('benchmark_waited_for_'+expected_ack,events[-1]=='ACK '+expected_ack)
            events.append('BENCH '+label);counter[0]+=7
            return dict(completed=count,duration=10.0,total_token_throughput=count*(length+output)/10,
                output_throughput=count*output/10,mean_ttft_ms=1.0,mean_tpot_ms=1.0,total_input_tokens=count*length,total_output_tokens=count*output)
        # Owned registry, HTTP replies, and benchmark calls are opaque fixtures;
        # actual full-only control flow, score validation, snapshots and deltas run.
        with patch.object(c.urllib.request,'build_opener',opener):
            result=module.full_workload_only_throughput(out,benchmark,'both',snapshot=lambda phase:session.capture(phase,owner,server))
        expected_events=[item for length in (4096,16384) for item in (f'ACK warm-before-{length}',f'BENCH full-warm-{length}',f'ACK warm-after-{length}',f'BENCH score-{length}',f'ACK score-after-{length}')]
        ok('six_ACKs_and_four_original_benchmark_calls',events==expected_events and set(result)=={'4096','16384'})
        receipt=session.finish();ok('four_real_counter_delta_files',len(list(session.directory.glob('*-delta.json')))==4)
        ok('six_raw_snapshots_no_private_token',len(session.rows)==6 and all(token.encode() not in p.read_bytes() for p in session.directory.iterdir()))
        ok('public_receipt_no_token',token not in json.dumps(receipt))
        # Restore only the opaque test session state to replay rejection cases;
        # the production observer never resets counters or truncates real files.
        saved_rows=session.rows;session.rows=[];session.previous=None
        for failure in ('timeout','bad_SHA','wrong_source','wrong_library'):
            fault[0]=failure
            with patch.object(c.urllib.request,'build_opener',opener):refuse('RPC_'+failure,lambda:session.capture(c.PHASES[0],owner,server))
            ok('no_progress_after_'+failure,session.rows==[])
        fault[0]=None;old_owner=owner.members;owner.members={}
        with patch.object(c.urllib.request,'build_opener',opener):refuse('unowned_worker',lambda:session.capture(c.PHASES[0],owner,server))
        owner.members=old_owner
        with patch.object(c.urllib.request,'build_opener',opener):refuse('existing_snapshot_not_overwritten',lambda:session.capture(c.PHASES[0],owner,server))
        session.rows=saved_rows
        stopped=[]
        def no_ack(phase):stopped.append(phase);raise RuntimeError('no ACK')
        refuse('missing_ACK_stops_before_benchmark',lambda:module.full_workload_only_throughput(out,lambda *args:(_ for _ in ()).throw(AssertionError('benchmark must not start')),'both',snapshot=no_ack))
        ok('only_initial_missing_ACK_attempt',stopped==['warm-before-4096'])
    ok('no_GPU_modules_imported',not {'torch','triton'}&set(sys.modules))
    report=dict(passed=True,checks=checks,unchanged_functions_classes=unchanged,parent_sha256=parent_sha,arm=arm,library_sha256=binding['library_sha256'],
        generated_sha256=hashlib.sha256(made).hexdigest(),source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),ROOT/'tools/biv150_runtime_stats_controller.py',ROOT/'tools/build_biv150_runtime_stats_controller.py')},
        actual_HTTP_GPU_worker_or_service_verified=False,Math105_or_throughput_result_produced=False,
        CPU_limitations=['Opaque HTTP, owner and benchmark fixtures; actual existing control-flow/validation/delta code executed.',
          'Frozen Git objects checked; actual service retains original HEAD/clean checks.',
          'No real HTTP/FastAPI/ZMQ/Msgpack/CUDA path or latency measured.'])
    print(json.dumps(report,sort_keys=True,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze-ref',default='HEAD');p.add_argument('--arm',choices=('A','B'),default='A');a=p.parse_args();main(a.freeze_ref,a.arm)
