"""Narrow production/controller inverse and existing opaque routing; no device run."""
import argparse
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import threading
from unittest.mock import patch
import biv150_native118_service_bridge as b

ROOT=Path(__file__).resolve().parents[1]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent-controller','parent-library','mixed-library','helper','compile-summary','consumer-review','step-review','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--production-ref',required=True);a=p.parse_args();checks=[]
    def reject(name,fn):
        try:fn()
        except (ValueError,RuntimeError,AssertionError):checks.append(name)
        else:raise AssertionError('accepted negative: '+name)
    source=(ROOT/'tools/run_biv150_native118_service.py').read_text()
    edits=json.loads((ROOT/'tools/biv150_native118_controller_edits.json').read_bytes())
    inverse=source
    for e in reversed(edits['edits']):
        assert inverse.count(e['new'])==e['count']==1;inverse=inverse.replace(e['new'],e['old'])
    assert inverse.encode()==a.parent_controller.read_bytes() and b.sha(a.parent_controller)==edits['parent_sha256']
    parent=load('cpu_parent117',a.parent_controller);child=load('cpu_child118',ROOT/'tools/run_biv150_native118_service.py')
    record=b.prepare(ROOT,a.production_ref,a.compile_summary,a.consumer_review,a.step_review)
    assert b.sha(a.parent_library)==b.PARENT and b.sha(a.mixed_library)==b.MIXED
    with tempfile.TemporaryDirectory(prefix='cjj-native118-controller-CPU-') as folder:
        folder=Path(folder);bridge=folder/'pending.json'
        def write(value):bridge.write_text(json.dumps(value,sort_keys=True)+'\n');return b.sha(bridge)
        pending=write(record)
        reject('actual_pending_bridge_refused',lambda:b.validate(ROOT,a.production_ref,bridge,pending,a.mixed_library))
        # Deliberately ephemeral approval flags only to exercise CPU gates.
        fixture=copy.deepcopy(record);fixture.update(source_bridge_Root_approved=True,service_admission_authorized=True)
        approved=write(fixture);assert b.validate(ROOT,a.production_ref,bridge,approved,a.mixed_library)==fixture
        for key,value in [('source_bridge_Root_approved',False),('service_admission_authorized',None),('actual_decode_step_review_sha256','0'*64),
                          ('actual_consumer_review_sha256','0'*64),('actual_SO_sha256','0'*64),('decode_step_scope',{}),('qualification',{}),
                          ('production',{}),('original_consumer_admission_modified',True),('service_verified',True)]:
            altered=copy.deepcopy(fixture);altered[key]=value;h=write(altered)
            reject('bridge_'+key,lambda h=h:b.validate(ROOT,a.production_ref,bridge,h,a.mixed_library))
        h=write(fixture)
        reject('bridge_content_SHA',lambda:b.validate(ROOT,a.production_ref,bridge,'0'*64,a.mixed_library))
        reject('mixed_SO_bytes',lambda:b.validate(ROOT,a.production_ref,bridge,h,a.parent_library))
        env={child.ENABLE:'1',b.ENV:'1',b.PATH_ENV:'/stale',b.SHA_ENV:'f'*64}
        native=child.native_env(env,a.parent_library);assert all(k not in native for k in (b.ENV,b.PATH_ENV,b.SHA_ENV))
        service=b.service_environment(native,a.mixed_library)
        assert service[b.SHA_ENV]==b.MIXED and service[child.K8_HASH]==b.PARENT
        off=child.native_env(dict(service,**{child.ENABLE:'0'}))
        assert all(k not in off for k in (b.ENV,b.PATH_ENV,b.SHA_ENV)) and off[child.K8_ENABLE]=='0'
        helper=load('cpu_helper94db',a.helper);assert b.sha(a.helper)==child.HELPER_SHA
        fingerprint=helper.expected_fingerprint(ROOT,dict(service,VLLM_FL_BIV150_NATIVE_W8A8='0',VLLM_FL_BIV150_NATIVE_GQA='0'),{'torch':'2.10.0','cxx11_abi':False})
        b.fingerprint(fingerprint)
        no=helper.expected_fingerprint(ROOT,dict(off,VLLM_FL_BIV150_NATIVE_W8A8='0',VLLM_FL_BIV150_NATIVE_GQA='0'),{'torch':'2.10.0','cxx11_abi':False})
        assert 'konly_encoded' not in no
        options=NS(baseline=False,native_k8_library=a.parent_library,native_k8_sha256=b.PARENT,graph_coverage_bootstrap_dir=None,
                   mixed_native_library=a.mixed_library,mixed_native_sha256=b.MIXED,mixed_source_bridge=bridge,mixed_source_bridge_sha256=h)
        b.options(options)
        for key,value in [('baseline',True),('native_k8_sha256','0'*64),('mixed_native_sha256','0'*64),
                          ('graph_coverage_bootstrap_dir',folder),('mixed_source_bridge',None)]:
            changed=copy.copy(options);setattr(changed,key,value);reject('options_'+key,lambda changed=changed:b.options(changed))
        marker='BIV150_KONLY_MIXED_NATIVE_SELECTED '
        event=dict(rows=61,segments=2,library_sha256=b.MIXED,host_selection_not_replay_count=True)
        log=folder/'server.log';before='(EngineCore_DP0 pid=100) '+marker+json.dumps(event)+'\n';log.write_text(before)
        start=log.stat().st_size
        assert not b.smoke_window(log,start,start)['mixed_events']
        with log.open('a') as f:f.write(before)
        window=b.smoke_window(log,start,log.stat().st_size);assert window['mixed_events'][0]['worker_pid']==100
        for key,value in [('rows',True),('segments',4),('library_sha256',b.PARENT),('host_selection_not_replay_count',False)]:
            changed=dict(event,**{key:value});reject('event_'+key,lambda changed=changed:b.mixed_events(marker+json.dumps(changed)))
        # Same-worker receipt parser; original117 detailed geometry parser is
        # exercised by the unchanged parent CPU suite/routing below, no GPU claims.
        data=dict(schema='cjj-konly-encoded-final-stats-v1',pid=100,start_ticks=200,implementation_sha256='x',stats={
            'mixed_native':dict(enabled=True,library=dict(path=str(a.mixed_library.resolve()),sha256=b.MIXED),rows_segments=b.PAIRS,service_verified=False,host_calls_not_graph_replays=3),
            'native_decode':dict(enabled=True,library=dict(path=str(a.parent_library.resolve()),sha256=b.PARENT),records=[dict(shape=32,capture=True,host_calls=1),dict(shape=64,capture=False,host_calls=1)])})
        stats=folder/'konly-encoded-final-100.json';stats.write_text(json.dumps(data))
        final='BIV150_KONLY_ENCODED_FINAL_STATS '+json.dumps(dict(pid=100,path=str(stats),sha256=b.sha(stats),bytes=stats.stat().st_size))
        owned=NS(members={100:dict(start_ticks=200,sid=90)})
        def fake_parser(text,library):return [dict(event=json.loads(line.split(' ',1)[1])) for line in text.splitlines()]
        def finish(o=owned,w=window):return b.final_evidence(final,folder,{'konly_encoded':{'implementation_sha256':'x'},'_mixed_library_path':str(a.mixed_library.resolve())},o,{'sid':90},w,a.parent_library,fake_parser)
        assert finish()['mixed_same_worker_smoke_verified']
        reject('final_wrong_owner_epoch',lambda:finish(NS(members={100:dict(start_ticks=201,sid=90)})))
        assert not finish(w=dict(window,mixed_events=[]))['mixed_same_worker_smoke_verified']
        # Threaded request assembly only; all HTTP replaced, no network calls.
        releases={};release_lock=threading.Lock()
        def request(i,payload,on_first):
            tag=payload['user'].rsplit('-',1)[0]
            with release_lock:release=releases.setdefault(tag,threading.Event())
            if payload.get('stream'):
                on_first();assert release.wait(10)
                return dict(usage={'completion_tokens':512})
            release.set();return dict(choices=[{}],usage={'completion_tokens':16})
        smoke=b.mixed_smoke_requests(request)
        assert [(x['keepers'],x['new_long_prompts']) for x in smoke['groups']]==[(61,3),(30,2)]
        assert all(len(x['records'])==x['keepers']+x['new_long_prompts'] for x in smoke['groups'])
        early=b.mixed_smoke_requests(lambda i,payload,on_first:dict(usage={'completion_tokens':512}))
        assert all(x['coverage_gaps'] and x['new_requests_submitted']==0 for x in early['groups'])
    # Reuse prior opaque routing/negative/fallback cases, only stale identity and
    # pending-qualification assertions adapted; no new numeric implementation.
    import check_biv150_native118_main117_merge_cpu as old
    service,sp=old.adapt('check_biv150_native_k8_service_cpu', [('cjj_native_k8_v_overlap_probe','cjj_native_k8_klookahead_probe'),('launch_cjj_native_k8_v_overlap_','launch_cjj_native_k8_klookahead_')]);service.SO=a.parent_library
    fp,fp_proof=old.adapt('check_biv150_native_k8_fingerprint_cpu', [('cjj_native_k8_v_overlap_probe','cjj_native_k8_klookahead_probe'),(old.OLD,old.NEW),('22e4a5e2a0b9d9a36ea390a11126417e1bf58e3c27f6b96ba0913b27690f09dd',old.OLD)])
    fp.LIBRARY,fp.HELPER=a.parent_library,a.helper
    mixed,mp=old.adapt('check_biv150_native_k8_mixed_cpu',[("assert all(v is None for k,v in pending.QUALIFICATION.items() if k.endswith('sha256'))","assert pending.QUALIFICATION == "+repr(b.QUALIFICATION)),('mixed_pending_fingerprint_rejected','mixed_missing_library_fingerprint_rejected'),('default_OFF_and_pending_rejection','default_OFF_and_missing_library_rejection'),('production qualification remains pending','production qualification pins independently checked; no service validation')])
    with contextlib.redirect_stdout(io.StringIO()):routing=mixed.run()
    assert routing['passed'] and not {'torch','triton','vllm','vllm_fl'}.intersection(sys.modules)
    report=dict(passed=True,CPU_only=True,GPU_executed=False,network_executed=False,
        controller_inverse_exact=True,controller_parent_sha256=edits['parent_sha256'],controller_edit_count=len(edits['edits']),
        production_four_pin_inverse_exact=True,retained5898_C3_exact=True,manifest_two_distinct_definitions=True,
        actual_final_source_bridge_pending=True,actual_bridge_sha256=b.digest(record),bridge_approval_fixture_ephemeral_only=True,
        actual_fingerprint=fingerprint,off_fingerprint=no,negative_checks=checks,
        mixed_window_and_same_worker_checks=True,smoke_request_assembly_fake_HTTP_only=True,
        inherited_routing=routing,adapters=[sp,fp_proof,mp],
        source_hashes={str(p.relative_to(ROOT)):b.sha(p) for p in [ROOT/'tools/run_biv150_native118_service.py',ROOT/'tools/biv150_native118_service_bridge.py',ROOT/'tools/biv150_native118_controller_edits.json',Path(__file__)]},
        service_verified=False,model_precision_verified=False)
    a.out.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');print(b.sha(a.out))

if __name__=='__main__':main()
