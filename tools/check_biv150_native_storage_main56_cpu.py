#!/usr/bin/env python3
"""PR120 on main56: real source/5898 bytes and opaque CPU guard fixtures, never GPU."""
import argparse,ast,contextlib,hashlib,importlib.util,io,json,struct,subprocess,sys,tempfile,types
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace as NS
from dataclasses import dataclass
import check_biv150_native_k8_service_cpu as old_fixture
import check_biv150_native_storage_observation_cpu as old_sequences
from check_biv150_konly_encoded_cpu import Tensor,Storage,Kernel,load
from check_biv150_konly_native_boundary_cpu import tensor
ROOT=Path(__file__).resolve().parents[1]
MAIN='56f88737fcf3b940760b9ca3027a925cd5a7d77d';MIXED='69827253756a0ef9e618ae0322e9ff361e93504f'
MODULE='vllm_fl/patches/biv150_konly_encoded_native.py'
SO=Path('/private/tmp/cjj-native117-klookahead-compile-v1-raw-20261029/out/compile/build/cjj_native_k8_klookahead_probe.so')
HELPER=Path('/private/tmp/cjj-actual-owned-service-helper-94db-20261028.py')
SHA='5898a00a4a4e3314ee8ed1fc40dffde02a766ca6d0e5c116dbd161dccaa3286b'
sha=lambda x:hashlib.sha256(x).hexdigest()
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])
def blob(ref,name):return git('show',ref+':'+name)
def function(text,name):return next(ast.get_source_segment(text,n) for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name==name)
def restore(source):
 changes=[('    backing=storage(t);_,start,end=backing;pointer=int(t.data_ptr());offset=int(t.storage_offset())','    _,start,end=storage(t);pointer=int(t.data_ptr());offset=int(t.storage_offset())'),
  ('    return backing\n',''),
  ('        # Only this call: tensors keeps every object alive; retain all bounds\n        # checks in order and reuse their observed whole-storage intervals.\n        backing={id(t):bounds(t) for t in tensors}\n        def observed_overlap(a,b):\n            _,x,y=backing[id(a)];_,u,v=backing[id(b)];return x<v and u<y','        for t in tensors:bounds(t)'),
  ('observed_overlap(src,k)','overlaps(src,k)'),('observed_overlap(out,src)','overlaps(out,src)'),('observed_overlap(out,other)','overlaps(out,other)')]
 for new,old in changes:assert source.count(new)==1,new;source=source.replace(new,old,1)
 return source
class CurrentOps(old_fixture.Ops):
 def __init__(self,schema):
  super().__init__(schema)
  # Opaque dispatcher recorder only. Bind the current actual5898 namespace;
  # none of old_fixture.check/actual_elf's8b qualification is called.
  self.cjj_native_k8_klookahead_probe=old_fixture.Namespace(self)
def external(raw,name,folder):
 path=folder/(name+'.py');path.write_bytes(raw);spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def mixed_fixture(parent,path,folder):
 """Execute frozen698's own make()/production forward and producer AST."""
 serial=len(list(folder.glob('mixed-*.py')))
 runtime_raw=blob(MIXED,'vllm_fl/patches/biv150_konly_encoded_runtime.py')
 mixed_raw=blob(MIXED,'vllm_fl/patches/biv150_konly_encoded_mixed_native.py')
 fixture_raw=blob(MIXED,'tools/check_biv150_native_k8_mixed_cpu.py')
 gqa=blob(MIXED,'vllm_fl/patches/biv150_gqa_decode.py').decode();tree=ast.parse(gqa)
 pick=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_pick_segments')
 pickns=dict(TARGET_PROGRAMS=256);exec(compile(ast.Module(body=[pick],type_ignores=[]),'<frozen698-pick>','exec'),pickns)
 launches=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ((isinstance(n.func,ast.Name) and n.func.id=='_launch_producer') or (isinstance(n.func,ast.Subscript) and isinstance(n.func.value,ast.Name) and n.func.value.id=='_reduce_partials'))];assert len(launches)==2
 launch_code=compile(ast.fix_missing_locations(ast.Module(body=[ast.Expr(n) for n in launches],type_ignores=[])),'<frozen698-producer-reducer-AST>','exec')
 original_load=old_fixture.load
 def selected_load(name):
  if name=='native':return parent
  if name=='mixed_native':return external(mixed_raw,'mixed-'+str(serial),folder)
  if name=='runtime':return external(runtime_raw,'runtime-'+str(serial),folder)
  return original_load(name)
 env=dict(__name__='cpu698_make_'+str(serial),Tensor=Tensor,Storage=Storage,Kernel=Kernel,tensor=tensor,NS=NS,dataclass=dataclass,hashlib=hashlib,
  load=selected_load,fixture=old_fixture.fixture,SO=path,Op=old_fixture.Op,pick=pickns['_pick_segments'],launch_code=launch_code)
 module=types.ModuleType(env['__name__']);sys.modules[env['__name__']]=module;module.__dict__.update(env)
 selected={'new_view','sliced','narrow','Metadata','Plan','NewNamespace','make'}
 nodes=[n for n in ast.parse(fixture_raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in selected];assert len(nodes)==len(selected)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual698-CPU-make-only>','exec'),module.__dict__)
 with patch.object(Tensor,'__getitem__',module.sliced,create=True),patch.object(Tensor,'narrow',module.narrow,create=True),patch.object(old_fixture,'load',selected_load):
  x=module.make(61,2048,folder)
 # Keep exact extensions during actual forward invocations as well.
 return x,module,runtime_raw,mixed_raw,fixture_raw

def run(path,helper):
 reference_raw=blob(MAIN,MODULE);candidate=(ROOT/MODULE).read_bytes();assert restore(candidate.decode()).encode()==reference_raw
 for name in ('storage','overlaps','geometry','__init__','verify','try_produce','stats','attach'):
  assert function(candidate.decode(),name)==function(reference_raw.decode(),name),name
 assert len(ast.parse(function(candidate.decode(),'overlaps')).body[0].args.args)==2
 manifest=json.loads((ROOT/'vllm_fl/patches/biv150_konly_encoded_sources.json').read_text());original=json.loads(blob(MAIN,'vllm_fl/patches/biv150_konly_encoded_sources.json'))
 assert manifest['native_decode']['adapter']['sha256']==sha(candidate) and manifest['native_decode']['library_sha256']==SHA
 manifest['native_decode']['adapter']['sha256']=original['native_decode']['adapter']['sha256'];assert manifest==original
 assert not git('diff',MAIN,'--','csrc','tools/kernels')
 actual=path.read_bytes();assert sha(actual)==SHA and actual[:6]==b'\x7fELF\x02\x01' and struct.unpack_from('<H',actual,18)[0]==62
 assert b'cjj_native_k8_klookahead_probe' in actual and all(x.encode() in actual for x in ('launch_cjj_native_k8_klookahead_s2','launch_cjj_native_k8_klookahead_s4'))
 native=load('native');assert native.LIBRARY_SHA==SHA and native.OPERATOR=='cjj_native_k8_klookahead_probe::produce'
 rows=[];mixed=[]
 with tempfile.TemporaryDirectory(prefix='pr120-main56-cpu-') as tmp,patch.object(old_fixture,'Ops',CurrentOps):
  folder=Path(tmp);ref=external(reference_raw,'main56_reference_native',folder)
  requests=[(n,s,p,'unchanged') for n,s in ((64,2),(32,4)) for p in (2048,2560)]
  requests += [(32,4,2560,label) for label in ('storage_too_small','valid_new_storage','pointer','stride','valid_stride','offset','valid_offset','shape','epoch','owner_storage','owner_pointer','role','whole_storage_source_alias','scratch_alias','same_object_alias','two_errors')]
  requests += [(n,s,2560,label) for n,s in ((30,4),(31,4),(61,2),(63,2)) for label in ('unchanged','whole_storage_source_alias','role')]
  for n,s,p,label in requests:
   with contextlib.redirect_stdout(io.StringIO()):a=old_sequences.sequence(ref,path,n,s,p,label);b=old_sequences.sequence(native,path,n,s,p,label)
   assert [{k:v for k,v in x.items() if k!='storage_helper_reads'} for x in a]==[{k:v for k,v in x.items() if k!='storage_helper_reads'} for x in b]
   assert a[0]['storage_helper_reads']==57 and b[0]['storage_helper_reads']==9
   if a[1]['accepted']:assert a[1]['storage_helper_reads']==57 and b[1]['storage_helper_reads']==9
   rows.append(dict(N=n,S=s,pitch=p,mutation=label,reference=a,candidate=b))
  for label,parent in (('main56',ref),('candidate',native)):
   x,ext,rr,mr,fr=mixed_fixture(parent,path,folder);calls=[];original_storage=Tensor.untyped_storage
   def counted(t):
    if sys._getframe(1).f_code is parent.storage.__code__:calls.append(id(t))
    return original_storage(t)
   with patch.object(Tensor,'__getitem__',ext.sliced,create=True),patch.object(Tensor,'narrow',ext.narrow,create=True),patch.object(Tensor,'untyped_storage',counted),contextlib.redirect_stdout(io.StringIO()):
    per=[]
    for iteration in range(2):
     before=len(calls);x.call();per.append(len(calls)-before)
    assert len(x.op.calls)==len(x.reducers.calls)==2 and not x.kernels['decode'].calls and x.adapter.active is None
    # External public API observes a changed storage on the next call.
    assert parent.overlaps(x.out,x.q) is False
    keep=x.q.storage;x.q.storage=x.out.storage;assert parent.overlaps(x.out,x.q) is True;x.q.storage=keep
   mixed.append(dict(implementation=label,actual698_forward_two_calls=True,mocked_new_namespace_not_GPU=True,storage_helper_calls=per,new_native_calls=2,original_TL_calls=0,reducers=2,public_two_argument_overlap_fresh=True,
    source_hashes=dict(runtime=sha(rr),mixed_module=sha(mr),original_CPU_fixture=sha(fr))))
  assert mixed[0]['storage_helper_calls']==[94,91] and mixed[1]['storage_helper_calls']==[46,43]
 # Actual94db expected_fingerprint in standalone mode; no operator or GPU load.
 assert sha(helper.read_bytes())=='94dbbb0fbc8f3c8837ea239ec18fc0ec4872a2b0049a1843d96a3e6a9701e976'
 spec=importlib.util.spec_from_file_location('pr120_main56_actual94db',helper);owned=importlib.util.module_from_spec(spec);spec.loader.exec_module(owned)
 env=dict(VLLM_FL_BIV150_NATIVE_W8A8='0',VLLM_FL_BIV150_NATIVE_GQA='0',VLLM_FL_BIV150_KONLY_ENCODED_SERVICE='1',VLLM_FL_BIV150_KONLY_NATIVE_DECODE='1',VLLM_FL_BIV150_KONLY_NATIVE_LIBRARY=str(path.resolve()),VLLM_FL_BIV150_KONLY_NATIVE_LIBRARY_SHA256=SHA)
 fp=owned.expected_fingerprint(ROOT,env,{'torch':'2.10.0','cxx11_abi':False});assert fp['konly_encoded']['native_decode']==dict(enabled=True,library_sha256=SHA,operator=native.OPERATOR)
 for bad in ('8b86cb688ad72dcf9859994ed2092620312cdd5ae96d194904b2bde7926aacc5','0'*64):
  try:owned.expected_fingerprint(ROOT,{**env,'VLLM_FL_BIV150_KONLY_NATIVE_LIBRARY_SHA256':bad},{'torch':'2.10.0','cxx11_abi':False})
  except (RuntimeError,ValueError):pass
  else:raise AssertionError('old/unknown library identity accepted')
 assert not {'torch','triton','vllm','numpy'}.intersection(sys.modules)
 return dict(passed=True,CPU_only=True,baseline=MAIN,candidate_source_sha256=sha(candidate),source_inverse_exact=True,public_overlaps_AST_and_bytes_unchanged=True,
  single_validate_storage_helper_calls=dict(reference=57,candidate=9),sequence_count=len(rows),sequences=rows,actual698_mixed_forward=mixed,
  qualification=dict(actual5898_library_bytes_parsed_SHA=SHA,library_loaded=False,old8b_qualification_not_used=True,CPU_dispatcher_only=True,actual_SDK_or_GPU_tested=False),
  C3_actual94db=True,C3_fingerprint=fp['sha256'],cross_call_freshness=True,alias_and_error_order_preserved=True,Native_try_produce_unchanged=True,
  GPU_math_unchanged=True,performance_verified=False,service_verified=False,CPU_micro_ratio_not_model_gain=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path,default=SO);p.add_argument('--helper',type=Path,default=HELPER);a=p.parse_args();print(json.dumps(run(a.library,a.helper),sort_keys=True,indent=2))
