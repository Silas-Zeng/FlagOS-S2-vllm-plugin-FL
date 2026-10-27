"""CPU-only fixtures for native host PID transition and exact host AST inverses."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import biv150_native_pid_admission as native
import biv150_v8_bias_byte_stage as contract
import bench_biv150_v8_bias_byte_stage as driver


def check():
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    calibration=contract.ROOT/'docs/evidence/v8-bias-decode-20261027/native-pid-admission-v1/root-controlled-calibration.json'
    actual=json.loads(calibration.read_text());assert actual['command']==native.QUERY
    container=actual['context_owner_pid'];host=native.parse_native_pids(actual['owned_context_stdout'])[0]
    assert container==623393 and host==1362794 and container!=host
    assert native.parse_native_pids(actual['idle_stdout'])==[]
    rejected=[]
    def reject(name,fn):
        try:fn()
        except (AssertionError,RuntimeError,KeyError):rejected.append(name)
        else:raise AssertionError('Admission fault accepted: '+name)
    for text in ('0','-1','+123','pid\n123','N/A','123,456','１２３','123\nunknown'):
        reject('bad_native_stdout_'+repr(text),lambda s=text:native.parse_native_pids(s))
    original_run=native.subprocess.run
    try:
        for code,stderr in ((1,''),(0,'unknown warning')):
            native.subprocess.run=lambda *a,c=code,e=stderr,**k:SimpleNamespace(returncode=c,stderr=e,stdout='123\n')
            reject('query_failure_'+str(code)+'_'+stderr,native.query_native_pids)
    finally:native.subprocess.run=original_run
    class Fixture:
        def __init__(self,values=None):
            self.values=copy.deepcopy(values if values is not None else [[],[host],[host],[host]])
            self.who=dict(pid=container,start_ticks=123456);self.roots=dict(validator='new',source='a698')
            self.initializations=0;self.events=[];self.closed=True;self.preflight_ok=True
            def preflight():self.events.append('original_preflight');assert self.preflight_ok
            def closed():self.events.append('port_closed');return self.closed
            def query():
                self.events.append('native_query');value=self.values.pop(0)
                if isinstance(value,Exception):raise value
                return value
            def identity():self.events.append('identity');return dict(self.who)
            def binding():self.events.append('binding');return dict(self.roots)
            def initialize():
                assert self.guard.owner==self.who and self.guard.roots==self.roots
                assert self.events[:3]==['original_preflight','port_closed','native_query']
                self.events.append('tiny_initialize_and_sync');self.initializations+=1;return object()
            self.guard=native.NativeAdmission(preflight,binding,query=query,identity=identity,
                closed=closed,initialize=initialize,nonce=lambda:'CPU synthetic token')
        def begin(self):return self.guard.begin()
        def phase(self,variant,token):return self.guard.check(SimpleNamespace(case=['steady4k'],variant=variant),token)
    good=Fixture();token=good.begin();good.phase('old',token);good.phase('A',token);good.guard.complete()
    receipt=good.guard.receipt();assert receipt['state']=='completed' and good.initializations==1
    assert receipt['container_identity']['pid']!=receipt['native_host_pid']
    assert [x['pids'] for x in receipt['queries']]==[[],[host],[host],[host]]
    for name,values,expected_inits in [('initial_foreign',[[999]],0),('transition_missing',[[],[]],1),
        ('transition_multiple',[[],[host,999]],1),('transition_duplicates',[[],[host,host]],1),
        ('initial_badquery',[None],0),('initial_query_error',[RuntimeError('query failed')],0),
        ('transition_query_error',[[],RuntimeError('query failed')],1)]:
        fixture=Fixture(values);reject(name,fixture.begin)
        assert fixture.initializations==expected_inits and fixture.guard.state=='failed'
    for name in ('service_active','original_preflight_rejected'):
        fixture=Fixture()
        if name=='service_active':fixture.closed=False
        else:fixture.preflight_ok=False
        reject(name,fixture.begin);assert fixture.initializations==0
    fixture=Fixture()
    def initialization_error():raise RuntimeError('synthetic initialization failure')
    fixture.guard.initialize=initialization_error
    reject('initialization_failure_not_reported_as_zero_gpu',fixture.begin)
    assert fixture.guard.receipt()['cuda_work_may_have_started'] and not fixture.guard.initialized
    for key,value in [('pid',container+1),('start_ticks',123457)]:
        fixture=Fixture();original_initialize=fixture.guard.initialize
        def changed_identity(f=fixture,k=key,v=value,init=original_initialize):
            result=init();f.who[k]=v;return result
        fixture.guard.initialize=changed_identity
        reject('identity_changed_during_initialization_'+key,fixture.begin)
        assert fixture.guard.initialized and fixture.guard.state=='failed'
    for name,pids in [('foreign_added',[host,999]),('native_disappeared',[]),('host_changed',[999]),
                     ('namespace_container_pid_is_not_host',[container]),('duplicate_native',[host,host])]:
        fixture=Fixture([[],[host],pids]);token=fixture.begin()
        reject(name,lambda f=fixture,t=token:f.phase('old',t));assert fixture.guard.state=='failed'
    for name,change in [('stale_container_pid',lambda f:f.who.update(pid=container+1)),
                        ('stale_start',lambda f:f.who.update(start_ticks=123457)),
                        ('validator_or_old_root_changed',lambda f:f.roots.update(source='new-root'))]:
        fixture=Fixture();token=fixture.begin();change(fixture)
        reject(name,lambda f=fixture,t=token:f.phase('old',t))
    for name,variant,badtoken in [('missing_token','old',None),('stale_token','old','other'),('phase_reordered','A','CPU synthetic token')]:
        fixture=Fixture();fixture.begin();reject(name,lambda f=fixture,v=variant,t=badtoken:f.phase(v,t))
    fixture=Fixture([[],[host],[999],[host]]);token=fixture.begin()
    reject('foreign_causes_permanent_rejection',lambda:fixture.phase('old',token))
    reject('token_cannot_recover_after_foreign_disappears',lambda:fixture.phase('old',token))
    reject('transition_token_cannot_be_reissued',good.begin);assert good.initializations==1
    ns,_=contract.frozen_namespace(contract.ROOT)
    original,_=ns['adapted_probe']();compile(driver.native_phase_tree(original),'native_phase_inverse','exec')
    compile(driver.adapted_run_tree(contract.ROOT),'native_host_inverse','exec')
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    return dict(calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),rejected_faults=rejected,
        synthetic_transition_fixture_only=True,container_and_driver_pid_distinguished=True,
        hardware_assert_ast_inverse=True,host_run_ast_inverse=True,
        original_numeric_bitwise_timing_unchanged=True,phase_requires_live_singleton_not_token_alone=True,
        fixture_tiny_calls=1,fixture_phases=['old','A'],gpu_executed=False,jit_executed=False,
        precision_gate=False,official_score=False)


if __name__=='__main__':print(json.dumps(check(),indent=2))
