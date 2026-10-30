#!/usr/bin/env python3
"""Default stdlib CPU plan; explicit Root-only frozen gather warmup, never launch."""
import argparse,ast,hashlib,importlib.util,json,os,re,socket,subprocess,sys,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PARENT='2773e07f74999cd18e60c978ad7665c032bcece4'
KERNEL='vllm_fl/patches/biv150_dynamic_hidden_gather_kernel.py'
KERNEL_SHA='c5613dc8645e862fee10be013f2238d436aa5760d3f5501be560c257b8d3ece5'
JIT_SHA='66a168ae959e719186771e66815e807f1e449e66c5283ef596ec3eb0e35cb775'
ENV='VLLM_FL_BIV150_DYNAMIC_HIDDEN_GATHER'
INPUT_ROWS=(1,63,64,65,1782,2018,2048)
SELECTED_ROWS=(1,7,8,9,31,41,42,43,59,60,61,62,63,64)
CASES=tuple((m,n) for m in INPUT_ROWS for n in SELECTED_ROWS if n<=m)
OPTIONS=dict(BLOCK_R=8,BLOCK_D=128,num_warps=4,num_stages=1)
sha=lambda b:hashlib.sha256(b).hexdigest()


def need(ok,why):
 if not ok:raise ValueError(why)


def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()


def freeze(v):
 if v is None or type(v) in (str,int,float,bool):return v
 if isinstance(v,dict):return {str(k):freeze(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [freeze(x) for x in v]
 return repr(v)


def typed(v):
 if v is None or type(v) in (str,int,float,bool):return dict(type=type(v).__name__,value=v)
 if isinstance(v,dict):return dict(type='dict',items=[[typed(k),typed(x)] for k,x in v.items()])
 if isinstance(v,(tuple,list)):return dict(type=type(v).__name__,items=[typed(x) for x in v])
 return dict(type=type(v).__module__+'.'+type(v).__name__,repr=repr(v))


def plan():
 raw=(ROOT/KERNEL).read_bytes();need(sha(raw)==KERNEL_SHA,'Frozen2773 kernel bytes changed')
 need(raw==subprocess.check_output(['git','-C',str(ROOT),'show',PARENT+':'+KERNEL]),'Kernel differs from parent')
 return dict(parent=PARENT,kernel_path=KERNEL,kernel_sha256=KERNEL_SHA,input_rows=INPUT_ROWS,selected_rows=SELECTED_ROWS,
  cases=CASES,repeats_per_case=2,expected_warmups=2*len(CASES),options=OPTIONS,
  fixed_backings=dict(hidden=[2048,2048],indices=[64],output=[64,2048]),
  actual_views='narrow(0,0,M/N): fixed pointers, zero offsets, actual runtime shapes/strides',
  device_data_initialized=False,kernel_launches=0,default_CPU=True,production_feature_enabled=False,
  mask_audit='Actual TTIR SSA mask dependencies and oriented comparisons; unsupported grammar stays unknown',
  compile_only_does_create_CUDA_context_and_empty_allocations=True,
  real_cold_receipt_or_service_hit_verified=False,device_numeric_verified=False,performance_verified=False,service_verified=False)


def positions(values):
 names=('hidden','indices','output','input_rows','selected_rows','BLOCK_R','BLOCK_D');result={}
 if not isinstance(values,dict):return None
 for key,value in values.items():
  if isinstance(key,tuple) and len(key)==1:key=key[0]
  if isinstance(key,str) and key in names:key=names.index(key)
  if type(key) is not int or not 0<=key<len(names) or key in result:return None
  result[key]=value
 return result


def runtime_ABI(kernel):
 signature=positions(getattr(kernel.src,'signature',None));constants=positions(getattr(kernel.src,'constants',None))
 known=signature is not None and constants is not None
 return dict(supported=known,verified=known and all(str(signature.get(i))==t for i,t in enumerate(('*bf16','*i32','*bf16','i32','i32')))
  and 3 not in constants and 4 not in constants and constants.get(5)==8 and constants.get(6)==128,
  signature=freeze(signature),constants=freeze(constants),typed={k:typed(getattr(kernel.src,k,None)) for k in ('signature','constants','attrs')})


def mask_audit(text):
 """Narrow straight-line TTIR grammar; no success from merely finding source text."""
 result=dict(supported=False,verified=False,needs_manual_review=True)
 try:
  need(len(re.findall(r'tt\.func\s+(?:public\s+)?@',text))==1,'one TTIR function')
  header=text[:text.index(') attributes') if ') attributes' in text else text.index('\n    %')]
  need(re.search(r'%arg3\s*:\s*i32\b',header) and re.search(r'%arg4\s*:\s*i32\b',header),'actual runtime i32 args')
  defs={};ops=[]
  for number,line in enumerate(text.splitlines(),1):
   m=re.match(r'\s*(%[\w.$]+)\s*=\s*(.*)',line)
   if m:defs[m[1]]=(number,m[2])
   if re.search(r'\btt\.(?:load|store)\b',line):ops.append((number,line.strip()))
  loads=[(v,row) for v,row in defs.items() if re.match(r'tt\.load\b',row[1])]
  stores=[row for row in ops if re.match(r'tt\.store\b',row[1])]
  need(len(loads)==2 and len(stores)==1,'two loads and one store')
  def refs(expr):return re.findall(r'%[\w.$]+',expr)
  def deps(v,seen=None):
   seen=set() if seen is None else seen
   need(v not in seen,'straight-line SSA');seen=seen|{v}
   return {v}|set().union(*(deps(x,seen) for x in refs(defs[v][1]))) if v in defs else {v}
  def origin(v,arg):
   if v==arg:return True
   if v not in defs:return False
   expr=defs[v][1];rr=refs(expr)
   return len(rr)==1 and bool(re.match(r'(tt\.(?:splat|broadcast|expand_dims)|arith\.extsi)\b',expr)) and origin(rr[0],arg)
  def zero(v):
   return v in defs and bool(re.match(r'arith\.constant\s+(?:dense<0>(?:\s|:)|0\b)',defs[v][1]))
  def compare(v):
   if v not in defs:return None
   m=re.match(r'arith\.cmpi\s+(slt|sge),\s*(%[\w.$]+),\s*(%[\w.$]+)',defs[v][1]);return m.groups() if m else None
  def leaves(v):
   if compare(v):return {v}
   expr=defs[v][1];need(re.match(r'(arith\.andi|tt\.(?:splat|broadcast|expand_dims))\b',expr),'boolean mask chain')
   return set().union(*(leaves(x) for x in refs(expr)))
  (idx,idxrow),(payload,prow)=loads
  ip=refs(idxrow[1]);pp=refs(prow[1]);sp=refs(stores[0][1])
  need(len(ip)>=2 and len(pp)>=2 and len(sp)>=3,'masked actual memory ops')
  need('%arg1' in deps(ip[0]) and '%arg0' in deps(pp[0]) and '%arg2' in deps(sp[0]),'typed source roles')
  row_pred=compare(ip[1]);need(row_pred and row_pred[0]=='slt' and origin(row_pred[2],'%arg4'),'selected-row predicate')
  need(row_pred[1] in deps(ip[0]),'same row in index pointer and mask')
  needed=leaves(pp[1]);store_leaves=leaves(sp[2]);need(needed==store_leaves and ip[1] in needed,'same source/store mask and selected tail')
  pairs={v:compare(v) for v in needed}
  lower=[v for v,c in pairs.items() if c[0]=='sge' and origin(c[1],idx) and zero(c[2])]
  upper=[v for v,c in pairs.items() if c[0]=='slt' and origin(c[1],idx) and origin(c[2],'%arg3')]
  need(len(lower)==len(upper)==1,'loaded index lower/upper bound')
  columns=needed-{ip[1],*lower,*upper};need(len(columns)==1,'column limit predicate')
  column=compare(next(iter(columns)));need(column[0]=='slt' and column[1] in deps(pp[0]) and column[1] in deps(sp[0]),'column source/output addressing')
  def limit(v):
   if v not in defs:return False
   expr=defs[v][1];rr=refs(expr)
   return bool(re.match(r'arith\.constant\s+(?:dense<2048>(?:\s|:)|2048\b)',expr)) or (len(rr)==1 and bool(re.match(r'tt\.(?:splat|broadcast|expand_dims)\b',expr)) and limit(rr[0]))
  need(limit(column[2]),'actual column <2048')
  need(idx in deps(pp[0]) and payload in deps(sp[1]),'loaded index addresses payload; same loaded payload stored')
  need('i16' in prow[1] and 'bf16' not in prow[1],'actual integer16 payload load')
  result.update(supported=True,verified=True,needs_manual_review=False,index_load_line=idxrow[0],payload_load_line=prow[0],store_line=stores[0][0],
   row_predicate=dict(SSA=ip[1],expression=defs[ip[1]][1]),
   payload_and_store_predicates={v:dict(line=defs[v][0],expression=defs[v][1]) for v in sorted(needed)},
   checked='actual masked memory SSA: selected tail, signed index >=0/<M, same payload/store mask and integer16 copy',
   affine_address_and_machine_masks_require_manual_review=True)
 except (ValueError,KeyError,IndexError,TypeError,RecursionError) as e:result['reason']=str(e)
 return result


def metadata(kernel):
 md=getattr(kernel,'metadata',None)
 values=md._asdict() if hasattr(md,'_asdict') else (md if isinstance(md,dict) else vars(md))
 return freeze(values)


def resources(kernel):
 md=metadata(kernel);values={k:getattr(kernel,k,None) for k in ('n_regs','n_spills')}
 values.update({k:md.get(k) for k in ('shared','num_warps','num_stages','warp_size','use_sme')})
 values['known']=type(values['n_regs']) is int and values['n_regs']>0 and type(values['n_spills']) is int and values['n_spills']>=0 and all(type(values[k]) is int and values[k]>=0 for k in ('shared','use_sme')) and all(type(values[k]) is int and values[k]>0 for k in ('num_warps','num_stages','warp_size'))
 return values


def descriptor(t):
 storage=t.untyped_storage()
 return dict(dtype=str(t.dtype),device=str(t.device),shape=list(t.shape),stride=list(t.stride()),storage_offset=t.storage_offset(),
  pointer=t.data_ptr(),pointer_mod64=t.data_ptr()%64,storage_pointer=storage.data_ptr(),storage_bytes=storage.nbytes())


def save(path,value):path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')


def execute(a,p):
 need(re.fullmatch('[0-9a-f]{40}',a.expected_source_commit or '') and git('rev-parse','HEAD')==a.expected_source_commit,'exact frozen source HEAD')
 need(not git('status','--porcelain','--untracked-files=all'),'clean source required')
 subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',PARENT,a.expected_source_commit],check=True)
 need(not git('diff',PARENT,'--','vllm_fl','csrc'),'production must equal2773')
 need(a.root_owned_exclusive,'Root explicit serial admission required')
 need(os.environ.get(ENV,'0')=='0','production feature stays OFF during isolated compile')
 need(not any(os.environ.get(k) for k in ('TRITON_KERNEL_OVERRIDE','TRITON_OVERRIDE_DIR')),'external IR overrides forbidden')
 need(not {'torch','triton','numpy'}.intersection(sys.modules),'admit before frameworks')
 need(a.out.is_absolute() and not a.out.exists() and not a.out.resolve().is_relative_to(ROOT),'fresh absolute out outside source')
 with socket.socket() as s:need(s.connect_ex(('127.0.0.1',9031))!=0,'service9031 active')
 a.out.mkdir(parents=True);out=a.out;sys.dont_write_bytecode=True
 result=dict(plan=p,source_root=str(ROOT),source_commit=a.expected_source_commit,completed=False,errors=[],records=[],objects=[],
  kernel_launches=0,initialized_tensor_elements=0,JIT_warmups_attempted=0,JIT_warmups_completed=0,
  explicit_handle_initializations=0,CUDA_initialized_before=None,CUDA_initialized_after=None,
  admission='Root explicit serial/exclusive coordination + closed9031; no PID-table idle inference',
  compile_only_not_zero_CUDA_context_work=True,actual_runtime_scalars_verified=False,actual_TTIR_mask_dependencies_verified=False,actual_final_machine_masks_manually_reviewed=False,
  single_actual_cache_key_observed=False,device_numeric_verified=False,real_receipt_cold_or_service_hit_verified=False,
  precision_gate=False,performance_verified=False,service_verified=False)
 def checkpoint():save(out/'compile-summary.json',result)
 checkpoint()
 try:
  (out/'frozen-kernel.py').write_bytes((ROOT/KERNEL).read_bytes());os.environ['TRITON_CACHE_DIR']=str(out/'triton-cache')
  import torch
  result['CUDA_initialized_before']=torch.cuda.is_initialized();need(result['CUDA_initialized_before'] is False,'CUDA already initialized')
  import triton
  jit=importlib.import_module('triton.runtime.jit');jitpath=Path(jit.__file__).resolve()
  actual_jit=jitpath.read_bytes();(out/'installed-jit.py.txt').write_bytes(actual_jit)
  result['actual_installed_JIT_source']=dict(path=str(jitpath),sha256=sha(actual_jit))
  need(sha(actual_jit)==JIT_SHA,'unknown installed warmup implementation; actual source retained for review')
  result['installed_runtime']=dict(torch_version=torch.__version__,torch_path=torch.__file__,triton_version=triton.__version__,triton_path=triton.__file__,jit_path=str(jitpath),jit_sha256=JIT_SHA)
  need(torch.__version__=='2.10.0','unknown Torch version')
  target=triton.runtime.driver.active.get_current_target();result['target']=dict(backend=target.backend,arch=target.arch,warp_size=target.warp_size)
  need((target.backend,target.arch,target.warp_size)==('corex',71,64),'CoreX target mismatch')
  prop=torch.cuda.get_device_properties(0);need('BI-V150' in prop.name and prop.multi_processor_count==16,'BI-V150 target mismatch')
  result['deterministic_algorithms_enabled']=torch.are_deterministic_algorithms_enabled()
  need(result['deterministic_algorithms_enabled'] is False,'Deterministic empty auto-fill may issue data kernels; do not change the global policy')
  hidden=torch.empty((2048,2048),dtype=torch.bfloat16,device='cuda');indices=torch.empty((64,),dtype=torch.int32,device='cuda');output=torch.empty((64,2048),dtype=torch.bfloat16,device='cuda')
  result['backings']={k:descriptor(t) for k,t in dict(hidden=hidden,indices=indices,output=output).items()}
  spec=importlib.util.spec_from_file_location('cjj_frozen_dynamic_hidden_gather_compile',ROOT/KERNEL);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  fn=module.gather_hidden_bits;result['actual_JIT_function_cache_key']=str(fn.cache_key);result['actual_do_not_specialize']=freeze(getattr(fn,'do_not_specialize',None))
  keep=[];object_ids={}
  for m,n in CASES:
   tensors=dict(hidden=hidden.narrow(0,0,m),indices=indices.narrow(0,0,n),output=output.narrow(0,0,n))
   desc={k:descriptor(v) for k,v in tensors.items()}
   for k,d in desc.items():need(d['pointer']==result['backings'][k]['pointer'] and d['storage_offset']==0,'pointer/alignment changed')
   for repeat in range(2):
    result['JIT_warmups_attempted']+=1;checkpoint()
    compiled=fn.warmup(tensors['hidden'],tensors['indices'],tensors['output'],m,n,grid=((n+7)//8,16),**OPTIONS)
    result['JIT_warmups_completed']+=1
    key=id(compiled)
    if key not in object_ids:
     object_ids[key]='object'+str(len(keep));keep.append(compiled);label=object_ids[key];folder=out/label;folder.mkdir()
     files={}
     for ext,value in compiled.asm.items():
      if not isinstance(value,(str,bytes)):continue
      b=value.encode() if isinstance(value,str) else value;(folder/('actual.'+ext)).write_bytes(b);files[ext]=dict(path=str((folder/('actual.'+ext)).relative_to(out)),bytes=len(b),sha256=sha(b))
     source=compiled.src.fn.src.encode();(folder/'actual.source.py').write_bytes(source);files['source']=dict(path=str((folder/'actual.source.py').relative_to(out)),bytes=len(source),sha256=sha(source))
     record=dict(object=label,python_object_id=key,actual_compiled_hash=str(compiled.hash),files=files,actual_options=metadata(compiled),
      runtime_ABI=runtime_ABI(compiled),actual_mask=mask_audit(compiled.asm.get('ttir','')),handles_explicitly_initialized=False,
      complete_artifacts=all(k in files and files[k]['bytes']>0 for k in ('source','ttir','ttgir','llir','asm','cubin')))
     result['objects'].append(record);checkpoint()
     compiled._init_handles();result['explicit_handle_initializations']+=1;record.update(handles_explicitly_initialized=True,resources=resources(compiled));save(folder/'object.json',record)
    result['records'].append(dict(input_rows=m,selected_rows=n,repeat=repeat,grid=[(n+7)//8,16],descriptors=desc,
     actual_object=object_ids[key],python_object_id=key,actual_compiled_hash=str(compiled.hash)))
    checkpoint()
  hashes={x['actual_compiled_hash'] for x in result['records']}
  result.update(unique_actual_compiled_hashes=sorted(hashes),unique_python_objects=len(keep),single_actual_cache_key_observed=len(hashes)==1,
   all_cases_repeated_same_object=all(result['records'][i]['actual_object']==result['records'][i+1]['actual_object'] for i in range(0,len(result['records']),2)),
   actual_runtime_scalars_verified=all(x['runtime_ABI']['verified'] for x in result['objects']),actual_TTIR_mask_dependencies_verified=all(x['actual_mask']['verified'] for x in result['objects']),
   all_loaded_resources_known=all(x['resources']['known'] for x in result['objects']))
  need(result['JIT_warmups_completed']==2*len(CASES) and all(x['complete_artifacts'] for x in result['objects']),'incomplete warmups or actual source/IR/cubin exports')
  need(plan()==p and git('rev-parse','HEAD')==a.expected_source_commit and not git('status','--porcelain'),'source changed during compile')
  result['completed']=True
 except Exception as e:
  result['errors'].append(repr(e));(out/'failure.txt').write_text(traceback.format_exc())
 finally:
  if 'torch' in sys.modules:result['CUDA_initialized_after']=sys.modules['torch'].cuda.is_initialized()
  result['artifacts']=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='compile-summary.json']
  checkpoint()
 return result


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--compile-only',action='store_true');p.add_argument('--root-owned-exclusive',action='store_true');p.add_argument('--expected-source-commit');p.add_argument('--out',type=Path);a=p.parse_args();planned=plan()
 if not a.compile_only:
  if a.root_owned_exclusive or a.expected_source_commit or a.out:p.error('device arguments require --compile-only')
  need(not {'torch','triton','numpy'}.intersection(sys.modules),'stdlib default');result=planned
 else:
  if not a.root_owned_exclusive or not a.expected_source_commit or a.out is None:p.error('Root explicit admission, exact source and new out required')
  result=execute(a,planned)
 print(json.dumps(result,sort_keys=True,indent=2));return 1 if a.compile_only and not result['completed'] else 0

if __name__=='__main__':raise SystemExit(main())
