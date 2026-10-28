"""Default-OFF source-pinned CPU observer; no framework imports/device operations."""
import builtins
from contextlib import contextmanager
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import types
import weakref

PINS={
'vllm_fl.compilation.graph':'ba78ed0a736bb21a4f2c307fc9775f15e3742b94533eff333b0c364eb8213fc2',
'vllm.compilation.cuda_graph':'cbc474f9098386d2eef2e3ff61364c611fbc1338d9c259ad5a24439aa7f06412',
'vllm.forward_context':'1c6fa51e9a02a120fed5e7357afb1190516cbdfefd1c644dfcd860603f1527c5',
'vllm.v1.cudagraph_dispatcher':'51f632753af729aeee6ce4a802d9d74b64f22eef02a524f36e2a24b07ef0cf63',
'vllm.v1.worker.gpu_input_batch':'de1ea4fc9d43fc6079dd375e407c2b6d93f5e1a11f49732d13654c41eaf30a65',
'vllm_fl.worker.model_runner':'6e86acf3ee8329468278b0f708465ddf503eb7937a5cf24cbe43dcefd71e5517',
'vllm_fl.patches.biv150_konly_encoded_native':'165d54a29d878b91532076b1a28c77ff5e43ff1b8772ab440b0fe7c54f2873b3',
'vllm_fl.patches.biv150_konly_encoded_runtime':'e51b8d86c54ec39f5aa2b5d465aa5bf2971fd335278c775504606ad40833bfb8',
'vllm_fl.patches.biv150_konly_encoded_views':'018c993858803df250ec1fa089778dbed5627e2c1720c3ccd12aad4e40344ede'}
SHA='22e4a5e2a0b9d9a36ea390a11126417e1bf58e3c27f6b96ba0913b27690f09dd'
OP='cjj_native_k8_qk_probe::produce'
FIELDS=('co_argcount','co_posonlyargcount','co_kwonlyargcount','co_nlocals','co_stacksize','co_flags','co_code','co_names','co_varnames','co_freevars','co_cellvars','co_filename','co_firstlineno','co_name','co_qualname','co_linetable','co_exceptiontable')
def code_at(code,*names):
    for name in names:code=next(x for x in code.co_consts if isinstance(x,types.CodeType) and x.co_name==name)
    return code

def same(a,b):
    return all(getattr(a,k)==getattr(b,k) for k in FIELDS) and len(a.co_consts)==len(b.co_consts) and all(same(x,y) if isinstance(x,types.CodeType) and isinstance(y,types.CodeType) else type(x) is type(y) and x==y for x,y in zip(a.co_consts,b.co_consts))

def require(value,message):
    if not value:raise ValueError(message)

def pinned(module):
    raw=Path(module.__file__).read_bytes();require(hashlib.sha256(raw).hexdigest()==PINS[module.__name__],'source SHA '+module.__name__)
    return compile(raw,str(module.__file__),'exec',dont_inherit=True,optimize=sys.flags.optimize)

def method(fn,code,module):require(isinstance(fn,types.FunctionType) and fn.__globals__ is vars(module) and fn.__closure__ is None and same(fn.__code__,code),'full-module function/global/closure identity')

def descriptor(d):
    names=('num_tokens','num_reqs','uniform','has_lora','num_active_loras')
    return {n:getattr(d,n) for n in names}

class Probe:
    def __init__(self,modules,*,emit=None,limit=8192):
        self.modules=modules;self.emit=emit;self.limit=limit;self.local=threading.local();self.certificates={};self.events=[];self.errors=[];self.error_count=0;self.overflow=0;self.restores=[];self.graph_calls=0;self.supported_frames=0;self.native_capture_prints=0
        codes={name:pinned(modules[name]) for name in PINS}
        self.graph=modules['vllm_fl.compilation.graph'];self.fc=modules['vllm.forward_context'];self.runner=modules['vllm_fl.worker.model_runner'];self.native=modules['vllm_fl.patches.biv150_konly_encoded_native'];self.inputs=modules['vllm.v1.worker.gpu_input_batch'];self.runtime=modules['vllm_fl.patches.biv150_konly_encoded_runtime'];self.views=modules['vllm_fl.patches.biv150_konly_encoded_views']
        self.original_forward=self.runner.ModelRunnerFL.__dict__['_model_forward'];self.original_graph=self.graph.GraphWrapper.__dict__['__call__'];self.native_code=self.native.Adapter.try_produce.__code__
        require(self.original_forward.__defaults__==(None,None,None,None) and self.original_forward.__kwdefaults__ is None and self.original_graph.__defaults__ is None and self.original_graph.__kwdefaults__ is None,'original wrapper argument defaults')
        method(self.original_forward,code_at(codes[self.runner.__name__],'ModelRunnerFL','_model_forward'),self.runner)
        method(self.original_graph,code_at(codes[self.graph.__name__],'GraphWrapper','__call__'),self.graph)
        
        for name in ('try_produce','validate','verify'):method(getattr(self.native.Adapter,name),code_at(codes[self.native.__name__],'Adapter',name),self.native)
        for name in ('get_forward_context','is_forward_context_available'):method(getattr(self.fc,name),code_at(codes[self.fc.__name__],name),self.fc)
        method(self.runtime.Registry.observe,code_at(codes[self.runtime.__name__],'Registry','observe'),self.runtime);method(self.runtime.descriptor,code_at(codes[self.runtime.__name__],'descriptor'),self.runtime)
        for name in ('req_ids','num_reqs'):method(self.inputs.InputBatch.__dict__[name].fget,code_at(codes[self.inputs.__name__],'InputBatch',name),self.inputs)
        require(self.runner.GraphWrapper is self.graph.GraphWrapper,'worker GraphWrapper binding')
        self.graph_type=self.graph.Graph.graph;self.observe_method=self.runtime.Registry.observe;self.descriptor_function=self.runtime.descriptor
        self.original_print=vars(self.native).get('print',builtins.print);require(self.original_print is builtins.print,'preexisting Native print observer')
    def record(self,value):
        value=dict(pid=os.getpid(),wall_ns=time.time_ns(),monotonic_ns=time.monotonic_ns(),diagnostic_only=True,formal_score_eligible=False,quality_covered=False,**value)
        if len(self.events)>=self.limit:self.overflow+=1;return
        self.events.append(value)
        if self.emit:self.emit(value)
    def attempt(self,fn,*args):
        try:return fn(*args)
        except BaseException as error:
            self.error_count+=1
            message=type(error).__name__+':'+str(error)
            if len(self.errors)<64:self.errors.append(message)
            if self.error_count==1 and self.emit:
                try:self.emit(dict(kind='observer_error',pid=os.getpid(),wall_ns=time.time_ns(),function=getattr(fn,'__name__','unknown'),error=message,diagnostic_only=True,formal_score_eligible=False,quality_covered=False))
                except BaseException:pass
            return None
    def origin(self,obj):
        require(type(obj) is self.runner.ModelRunnerFL,'unknown runner')
        r=vars(obj)
        if r.get('speculative_config') is not None or r.get('use_async_spec_decode',False) is not False:return None
        require(r.get('uniform_decode_query_len')==1,'query length unsupported')
        batch=r.get('input_batch');require(type(batch) is self.inputs.InputBatch,'unknown InputBatch')
        b=vars(batch);ids=b.get('_req_ids');mapping=b.get('req_id_to_index');states=r.get('requests')
        require(type(ids) is list and type(mapping) is dict and type(states) is dict,'CPU active request container')
        if not 0<len(ids)<=64:return None
        require(all(type(x) is str for x in ids) and len(set(ids))==len(ids) and all(type(k) is str and type(v) is int for k,v in mapping.items()),'CPU request map types')
        require(mapping=={x:i for i,x in enumerate(ids)},'active/nonpadding request identity')
        rows=[]
        for rid in ids:
            state=states.get(rid);require(type(state) is self.inputs.CachedRequestState,'unknown request state');s=vars(state)
            tokens=s.get('prompt_token_ids');computed=s.get('num_computed_tokens');n=s.get('num_prompt_tokens')
            require(type(tokens) is list and type(n) is int and n==len(tokens) and type(computed) is int,'unknown CPU prompt state')
            if computed<n or s.get('prompt_embeds') is not None:return None
            rows.append(dict(request_id=rid,prompt_tokens=n,computed_tokens=computed))
        return dict(valid_requests=len(ids),requests=rows,runner_id=id(obj))
    def forward(self,obj,*args,**kwargs):
        previous=getattr(self.local,'request',None);nested=previous is not None
        if nested:previous['nested_origin_ambiguous']=True
        self.local.request=None if nested else self.attempt(self.origin,obj)
        try:return self.original_forward(obj,*args,**kwargs)
        finally:self.local.request=previous
    def before(self,obj):
        require(type(obj) is self.graph.GraphWrapper,'unknown FL GraphWrapper')
        if not self.fc.is_forward_context_available():return None
        ctx=self.fc.get_forward_context();mode=self.graph.CUDAGraphMode
        if ctx.cudagraph_runtime_mode is not mode.FULL or obj.runtime_mode is not mode.FULL:return None
        metadata=ctx.attn_metadata
        if type(metadata) is not dict or not metadata or any(type(vars(x).get('max_query_len')) is not int or vars(x)['max_query_len']!=1 for x in metadata.values()):return None
        d=ctx.batch_descriptor
        if type(d) is not self.fc.BatchDescriptor:return None
        if type(d.num_tokens) is not int or type(d.num_reqs) is not int or type(d.num_active_loras) is not int:return None
        if d.uniform is not True or d.has_lora is not False or d.num_active_loras!=0 or d.num_tokens not in (32,64) or d.num_reqs!=d.num_tokens or ctx.ubatch_slices is not None or ctx.skip_compiled is not False:return None
        entry=obj.concrete_graph_entries.get(d);g=None if entry is None else entry.graph
        return dict(wrapper=obj,descriptor=d,descriptor_values=descriptor(d),prior_entry=entry,prior_graph=g,native=[])
    def graph_call(self,obj,*args,**kwargs):
        self.graph_calls+=1
        frame=self.attempt(self.before,obj);self.supported_frames+=frame is not None;stack=getattr(self.local,'graphs',[]);self.local.graphs=stack;stack.append(frame)
        try:result=self.original_graph(obj,*args,**kwargs)
        except BaseException:raise
        else:self.attempt(self.after,frame);return result
        finally:stack.pop()
    def native_print(self,*args,**kwargs):
        result=self.original_print(*args,**kwargs)
        if len(args)>=2 and args[0]=='BIV150_KONLY_NATIVE_DECODE_CAPTURED':self.attempt(self.note_native,sys._getframe(1),args[1])
        return result
    def note_native(self,caller,payload):
        self.native_capture_prints+=1
        require(caller.f_code is self.native_code,'Native print caller code')
        candidates=[x for x in getattr(self.local,'graphs',[]) if x is not None and x['prior_graph'] is None]
        require(len(candidates)==1,'missing/ambiguous FULL capture frame')
        e=json.loads(payload);require(e['capture'] is True and e['native_launch_returned'] is True and e['library']['sha256']==SHA and e['operator']==OP and e['physical_page_bytes']==16384 and e['format']=='cjj-konly-rn32-rne-v2','Native captured identity')
        require(e['shape']==candidates[0]['descriptor_values']['num_tokens'] and e['segments']==(2 if e['shape']==64 else 4),'Native bucket')
        entry=caller.f_locals['entry'];runtime=caller.f_locals['runtime'];require(type(runtime.registry) is self.runtime.Registry,'actual registry class')
        require(type(entry.generation) is int and entry.generation==e['entry_generation'],'Native epoch binding')
        candidates[0]['native'].append(dict(event=e,entry=weakref.ref(entry),runtime=weakref.ref(runtime),observation=entry.observation))
    def fresh(self,cert):
        require(self.runtime.Registry.observe is self.observe_method and self.runtime.descriptor is self.descriptor_function,'registry metadata observer changed')
        for row in cert['native']:
            entry,runtime=row['entry'](),row['runtime']();require(entry is not None and runtime is not None,'Native entry/runtime expired')
            registry=runtime.registry;require('observe' not in vars(registry) and registry.broker is self.views,'registry metadata binding changed');require(registry.armed is True and type(entry.generation) is int and type(registry.generation) is int and entry.generation==row['event']['entry_generation'] and entry.generation<=registry.generation and any(x is entry for x in registry.entries.values()),'Native epoch no longer owned')
            require(registry.observe(entry.owner)==row['observation']==entry.observation,'Native storage reallocated/mutated')
        return True
    def after(self,frame):
        if frame is None:return
        obj,d=frame['wrapper'],frame['descriptor'];entry=obj.concrete_graph_entries.get(d)
        if entry is None or entry.graph is None:return
        graph=entry.graph;key=(os.getpid(),id(obj),id(entry),id(graph))
        if frame['prior_graph'] is None:
            # Never carry a certificate from another capture, even if its bucket matches.
            self.certificates.pop(key,None)
            if not frame['native']:return
            require(type(graph) is self.graph_type,'unknown graph type')
            cert=dict(wrapper=weakref.ref(obj),entry=weakref.ref(entry),graph=weakref.ref(graph),descriptor=frame['descriptor_values'],native=frame['native'])
            self.fresh(cert);self.certificates[key]=cert
            self.record(dict(kind='native_capture_certificate',bucket=cert['descriptor'],wrapper_id=id(obj),entry_id=id(entry),graph_id=id(graph),native_count=len(cert['native']),native_generations=sorted({x['event']['entry_generation'] for x in cert['native']}),native_payloads=[x['event'] for x in cert['native']],request_replay=False));return
        require(entry is frame['prior_entry'] and graph is frame['prior_graph'],'entry/graph replaced during replay')
        cert=self.certificates.get(key)
        if cert is None:return
        require(cert['wrapper']() is obj and cert['entry']() is entry and cert['graph']() is graph and cert['descriptor']==descriptor(d),'expired/changed graph certificate')
        scope=getattr(self.local,'request',None)
        if scope is None or scope.get('nested_origin_ambiguous'):return
        require(scope['valid_requests']<=d.num_reqs,'valid requests exceed padded bucket');self.fresh(cert)
        self.record(dict(kind='real_request_native_full_graph_replay_returned',bucket=cert['descriptor'],valid_requests=scope['valid_requests'],padding_rows=d.num_reqs-scope['valid_requests'],requests=scope['requests'],wrapper_id=id(obj),entry_id=id(entry),graph_id=id(graph),native_count=len(cert['native']),native_generations=sorted({x['event']['entry_generation'] for x in cert['native']}),replay_returned=True,native_library_sha256=SHA,operator=OP,GPU_completion_or_precision_proven=False))
    def install(self,environ=None):
        env=os.environ if environ is None else environ;require(env.get('CJJ_NATIVE_GRAPH_COVERAGE')=='1','explicit default-OFF diagnostic flag required')
        if self.restores:return self
        probe=self
        @functools.wraps(self.original_forward)
        def forward(*a,**kw):return probe.forward(*a,**kw)
        @functools.wraps(self.original_graph)
        def graph(*a,**kw):return probe.graph_call(*a,**kw)
        self.runner.ModelRunnerFL._model_forward=forward;self.graph.GraphWrapper.__call__=graph;self.native.print=self.native_print
        self.restores=[(self.runner.ModelRunnerFL,'_model_forward',self.original_forward),(self.graph.GraphWrapper,'__call__',self.original_graph)]
        self.attempt(self.record,dict(kind='observer_install',probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_pins=PINS,source_paths={k:str(v.__file__) for k,v in self.modules.items()}));return self
    def uninstall(self):
        for cls,name,value in self.restores:setattr(cls,name,value)
        self.restores.clear()
        if vars(self.native).get('print')==self.native_print:del self.native.print
    def summary(self):return dict(diagnostic_only=True,formal_score_eligible=False,quality_covered=False,source_pins=PINS,events=self.events,observer_errors=self.errors,observer_error_count=self.error_count,graph_calls_seen=self.graph_calls,supported_full_frames_seen=self.supported_frames,native_capture_prints_seen=self.native_capture_prints,overflow=self.overflow,recording_integrity=not self.errors and not self.overflow,all_requests_or_quality_covered=False,qualifying_replay_lower_bound=sum(x['kind']=='real_request_native_full_graph_replay_returned' for x in self.events))

def plan():return dict(default_OFF=True,stdlib_only=True,formal_score_eligible=False,quality_covered=False,source_pins=PINS,activation='explicit install(loaded modules); no bootstrap/framework import in this prototype',observes='specific FULL capture->same graph replay within actual _model_forward',padded_bucket_is_not_valid_request_count=True,device_operations_added=0,CLI_or_library_changes=False)

if __name__=='__main__':print(json.dumps(plan(),sort_keys=True,indent=2))
