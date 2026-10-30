#!/usr/bin/env python3
"""Stdlib CPU plan by default; Root-only direct bit-copy of the reviewed cached cubin."""
import argparse,array,gzip,importlib.util,json,os,socket,sys,traceback,types
from pathlib import Path
import compile_biv150_dynamic_hidden_gather as c

ROOT=Path(__file__).resolve().parents[1]
PARENT='8d1959a443964b8d0789f9dac23bcb4945729dc1'
SUMMARY_SHA='08b4dcded55b22c871b64b9eb9c362b70893e3e6f47db1681f01108c91a4aa57'
REVIEW_SHA='bdb0fb8908506630c01e45323a921c4aa1d2adcb522405d28c2ce30e0dfb6a05'
COMPILER_SHA='a97f067023873acf2e348c28f61494e59170ca84c3ee2314f20e491d3e833dfd'
HASH='8a67dfc845796a7902db88222df5c2d589f211312a91537ff44523c59ca35042'
CUBIN_SHA='e40cfa77a95208fa593748561da6e02b8bd376a28f3bbb153cb127ca6cc1728c'
LAUNCHER_SHA='512414b6fbfae2135b62153619ef1f5919bd2fd07018058cd45035de1d83e361'
CASES=((65,9),(2018,61));D=2048;GUARD=128;IGUARD=64
SPECIAL=(0x0000,0x8000,0x7f80,0xff80,0x7fc1,0x7f81,0xffc1,0xff81)
need=c.need;sha=c.sha

def plan():
 need(sha((ROOT/c.KERNEL).read_bytes())==c.KERNEL_SHA,'frozen production kernel')
 return dict(parent=PARENT,kernel_sha256=c.KERNEL_SHA,cases=CASES,phases=['A','B'],candidate_launches_planned=4,
  width=2048,hidden_output_guard_uint16_words=GUARD,index_guard_int32_words=IGUARD,direct_cached_CompiledKernel=True,warmup_calls=0,IR_compile_API_calls=0,timing_executed=False,
  oracle='full independent CPU uint16 row indexing, exact bits including +/-0, Inf and distinct NaN payloads',
  guards='128 uint16 prefix/suffix on hidden/output;64 int32 prefix/suffix on indices; full input bytes readonly',
  change_B='same GPU storages; replace hidden and indices, poison every output payload word with expected XOR65535 before launch',
  compiled_hash=HASH,cubin_sha256=CUBIN_SHA,compile_summary_sha256=SUMMARY_SHA,actual_review_sha256=REVIEW_SHA,
  default_CPU=True,production_feature_enabled=False,device_numeric_verified=False,helper_hit_verified=False,
  Graph_verified=False,side_stream_verified=False,performance_verified=False,service_verified=False)

def typed(v):
 t=v['type']
 if t in ('str','int','float','bool','NoneType'):return v['value']
 if t=='dict':return {typed(k):typed(x) for k,x in v['items']}
 if t=='list':return [typed(x) for x in v['items']]
 if t=='tuple':return tuple(typed(x) for x in v['items'])
 raise ValueError('unknown saved typed ABI '+t)

def packed(v):
 need(type(v) is tuple and len(v)==3 and all(type(x) is int for x in v) and v==(4,1,0),'actual packed metadata must be exact three-int (4,1,0)')
 return dict(actual_packed_metadata=c.freeze(v),actual_packed_metadata_typed=c.typed(v),
  expected_launch_block=[v[0]*64,1,1],dimensions_basis='actual packed tuple + reviewed launcher multiplication by64',
  hardware_observed_launch_dimensions=None)

def words(raw,kind='H'):
 need(sys.byteorder=='little','fixture little endian');a=array.array(kind);a.frombytes(raw);return a

def indices(m,n,phase):
 need(type(m) is int and type(n) is int and 0<n<=m and phase in (0,1),'fixture scope')
 first=[m-1,0,m-1] if phase==0 else [0,m-1,0]
 result=(first+[((i*(137 if phase==0 else 193))+(17 if phase==0 else 37))%m for i in range(n)])[:n]
 need(all(type(x) is int and 0<=x<m for x in result),'independent CPU index bounds')
 return result

def fixture(m,n,phase):
 need((m,n) in CASES and type(phase) is int and phase in (0,1),'only two approved fixture geometries')
 need(array.array('H').itemsize==2 and array.array('i').itemsize==4,'fixture integer widths')
 data=array.array('H',(((r*D+j)*40503 ^ ((r>>5)*29771) ^ (phase*0x5a3c))&65535 for r in range(m) for j in range(D)))
 for r in range(m):data[r*D:r*D+len(SPECIAL)]=array.array('H',SPECIAL)
 idx=indices(m,n,phase);expected=b''.join(data[x*D:(x+1)*D].tobytes() for x in idx)
 hidden=(array.array('H',[0xe3a7])*GUARD).tobytes()+data.tobytes()+(array.array('H',[0xe3a7])*GUARD).tobytes()
 index=(array.array('i',[-987654321])*IGUARD).tobytes()+array.array('i',idx).tobytes()+(array.array('i',[-987654321])*IGUARD).tobytes()
 poison=array.array('H',(x^65535 for x in words(expected))).tobytes()
 output=(array.array('H',[0xd34b])*GUARD).tobytes()+poison+(array.array('H',[0xd34b])*GUARD).tobytes()
 return dict(hidden=hidden,indices=index,output_poison=output,expected=expected,logical_indices=idx)

def checks(f,hidden,index,output):
 expected=f['expected'];guard=2*GUARD;before=f['output_poison']
 return dict(output_all_uint16_bits=output[guard:-guard]==expected,hidden_full_bytes_readonly=hidden==f['hidden'],
  indices_full_bytes_readonly=index==f['indices'],output_size=len(output)==len(before),
  prefix_guard=output[:guard]==before[:guard],suffix_guard=output[-guard:]==before[-guard:],
  every_poison_word_differs=all(x!=y for x,y in zip(words(before[guard:-guard]),words(expected))))

def proof(folder,review):
 need(sha(review.read_bytes())==REVIEW_SHA,'Root-read independent actual review SHA')
 r=json.loads(review.read_text());need(r['compile_evidence_consistent'] is True and r['compile_summary_sha256']==SUMMARY_SHA and r['actual_hash']==HASH and r['cubin_sha256']==CUBIN_SHA,'review identities')
 raw=(folder/'compile-summary.json').read_bytes();need(sha(raw)==SUMMARY_SHA,'original compile summary SHA; original maskfalse retained')
 s=json.loads(raw);need(s['source_commit']==PARENT and s['completed'] is True and not s['errors'] and s['kernel_launches']==0,'original compile scope')
 for v in s['artifacts']:
  p=folder/v['path'];need(p.is_relative_to(folder) and p.is_file(),'artifact path');b=p.read_bytes();need(len(b)==v['bytes'] and sha(b)==v['sha256'],'actual artifact '+v['path'])
 need(len(s['objects'])==1 and s['objects'][0]['actual_compiled_hash']==HASH,'one actual compiled object')
 o=s['objects'][0];need(o['resources']==dict(known=True,n_regs=6,n_spills=0,num_stages=1,num_warps=4,shared=0,use_sme=0,warp_size=64),'reviewed loaded resources')
 need(sha((folder/'object0/actual.cubin').read_bytes())==CUBIN_SHA,'actual binary bytes')
 return s,o

def execute(a):
 need(a.root_owned_exclusive,'Root later explicit serial admission required')
 need(a.expected_source_commit and c.git('rev-parse','HEAD')==a.expected_source_commit and not c.git('status','--porcelain','--untracked-files=all'),'exact clean worker source')
 c.git('merge-base','--is-ancestor',PARENT,'HEAD')
 need(not c.git('diff',PARENT,'--','vllm_fl','csrc'),'production zero diff')
 need(not {'torch','triton','numpy'}.intersection(sys.modules),'admit before frameworks')
 need(os.environ.get(c.ENV,'0')=='0' and not any(os.environ.get(k) for k in ('TRITON_KERNEL_OVERRIDE','TRITON_OVERRIDE_DIR')),'production OFF, no external IR override')
 need(a.compile_out.is_absolute() and a.actual_review.is_absolute(),'absolute existing proof paths')
 s,old=proof(a.compile_out,a.actual_review)
 with socket.socket() as sock:need(sock.connect_ex(('127.0.0.1',9031))!=0,'service9031 active')
 need(a.out.is_absolute() and not a.out.exists() and not a.out.resolve().is_relative_to(ROOT) and not a.out.resolve().is_relative_to(a.compile_out),'fresh isolated output')
 a.out.mkdir(parents=True);out=a.out;sys.dont_write_bytecode=True
 result=dict(plan=plan(),source_commit=a.expected_source_commit,source_root=str(ROOT),binary_source_root=s['source_root'],
  compiled_out=str(a.compile_out),compile_summary_sha256=SUMMARY_SHA,actual_review_sha256=REVIEW_SHA,
  completed=False,errors=[],records=[],kernel_launches_attempted=0,kernel_launches_completed=0,warmup_calls=0,IR_compile_API_calls=0,
  host_launcher_may_load_or_build=True,device_numeric_verified=False,bit_copy_verified=False,helper_hit_verified=False,
  precision_gate=False,timing_executed=False,Graph_verified=False,side_stream_verified=False,performance_verified=False,service_verified=False,
  admission='Root explicit serial coordination and closed9031; PID namespace visibility is not exclusivity proof',CUDA_initialized_before=None,CUDA_initialized_after=None)
 def save():c.save(out/'bits-summary.json',result)
 def blob(raw):
  digest=sha(raw);p=out/'arrays'/('sha256-'+digest+'.bin.gz');p.parent.mkdir(exist_ok=True)
  if not p.exists():p.write_bytes(gzip.compress(raw,compresslevel=1,mtime=0))
  return dict(path=str(p.relative_to(out)),bytes=len(raw),sha256=digest)
 save()
 try:
  cache=out/'launcher-cache';cache.mkdir();os.environ['TRITON_CACHE_DIR']=str(cache)
  for v in s['artifacts']:
   if v['path'].endswith('.so'):
    rel=Path(v['path']).relative_to('triton-cache');dest=cache/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((a.compile_out/v['path']).read_bytes())
  import torch,numpy as np,triton
  from triton.compiler.compiler import ASTSource,CompiledKernel
  import triton.compiler.compiler as compiler
  result['runtime']=dict(torch_version=torch.__version__,numpy_version=np.__version__,triton_version=triton.__version__,compiler_path=compiler.__file__,compiler_sha256=sha(Path(compiler.__file__).read_bytes()))
  need(torch.__version__=='2.10.0' and result['runtime']['compiler_sha256']==COMPILER_SHA,'reviewed SDK loader API')
  result['CUDA_initialized_before']=torch.cuda.is_initialized();need(result['CUDA_initialized_before'] is False,'CUDA already initialized')
  need(not torch.are_deterministic_algorithms_enabled(),'deterministic empty auto-fill policy unsupported')
  target=triton.runtime.driver.active.get_current_target();need((target.backend,target.arch,target.warp_size)==('corex',71,64),'reviewed target')
  prop=torch.cuda.get_device_properties(0);need('BI-V150' in prop.name and prop.multi_processor_count==16,'reviewed device')
  spec=importlib.util.spec_from_file_location('cjj_frozen_dynamic_hidden_gather_compile',ROOT/c.KERNEL);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
  fn=mod.gather_hidden_bits;need(str(fn.cache_key)==s['actual_JIT_function_cache_key'],'frozen function cache key')
  abi=old['runtime_ABI']['typed'];src=ASTSource(fn,typed(abi['signature']),constexprs=typed(abi['constants']),attrs=typed(abi['attrs']))
  groups=list((a.compile_out/'triton-cache').glob('*/__grp__gather_hidden_bits.json'));need(len(groups)==1,'one original kernel cache group');bucket=groups[0].parent
  group={name:str(bucket/name) for name in json.loads((bucket/'__grp__gather_hidden_bits.json').read_text())['child_paths']}
  compiled=CompiledKernel(src,group,HASH);result['actual_options']=c.metadata(compiled);result.update(packed(compiled.packed_metadata));save()
  result['actual_runtime_ABI']=c.runtime_ABI(compiled);result['loaded_cubin_sha256']=sha(compiled.kernel);save()
  need(result['actual_options']==old['actual_options'] and result['actual_runtime_ABI']==old['runtime_ABI'] and result['loaded_cubin_sha256']==CUBIN_SHA,'original options, typed launcher ABI and loaded cubin')
  for ext in ('source','ttir','ttgir','llir','asm','cubin'):
   actual=compiled.asm[ext];actual=actual.encode() if isinstance(actual,str) else actual
   need(actual==(a.compile_out/'object0'/('actual.'+ext)).read_bytes(),'loaded original artifact '+ext)
  compiled._init_handles();result['loaded_resources']=c.resources(compiled);need(result['loaded_resources']==old['resources'],'actual loaded resources differ; stop before launch')
  launch=getattr(compiled._run,'launch',None);module=getattr(launch,'__self__',None)
  result['launcher_connection']=dict(type=type(compiled._run).__module__+'.'+type(compiled._run).__qualname__,attributes={k:type(v).__module__+'.'+type(v).__qualname__ for k,v in getattr(compiled._run,'__dict__',{}).items()},callable_type=type(launch).__name__,owner_type=type(module).__name__);save()
  need(isinstance(module,types.ModuleType) and getattr(module,'__file__',None),'unknown actual launch-extension ownership; retain metadata, stop before launch')
  launcher=Path(module.__file__).resolve();result['actual_launcher']=dict(path=str(launcher),sha256=sha(launcher.read_bytes()),module=module.__name__,callable_type=type(launch).__name__);save()
  need(result['actual_launcher']['sha256']==LAUNCHER_SHA,'actual loaded launcher changed; original hash not inherited')
  (out/'actual-launcher.so').write_bytes(launcher.read_bytes());(out/'actual.cubin').write_bytes(compiled.asm['cubin'])
  result['actual_compiled_hash']=str(compiled.hash);result['actual_python_object_id']=id(compiled);save()
  def read(t):return t.detach().cpu().numpy().tobytes()
  for m,n in CASES:
   previous=None;backings=None;descriptors=None
   for phase in (0,1):
    f=fixture(m,n,phase);record=dict(input_rows=m,selected_rows=n,phase='AB'[phase],grid=[(n+7)//8,16,1],inputs={k:blob(v) for k,v in f.items() if isinstance(v,bytes)},logical_indices=f['logical_indices'],launch_attempted=False,launch_completed=False)
    result['records'].append(record);save()
    hb=torch.frombuffer(bytearray(f['hidden']),dtype=torch.int16);ib=torch.frombuffer(bytearray(f['indices']),dtype=torch.int32);ob=torch.frombuffer(bytearray(f['output_poison']),dtype=torch.int16)
    if backings is None:backings=(hb.clone().to('cuda'),ib.clone().to('cuda'),ob.clone().to('cuda'))
    else:
     for dst,host in zip(backings,(hb,ib,ob)):dst.copy_(host)
    h,i,o=backings;hv=h.narrow(0,GUARD,m*D).view(torch.bfloat16).reshape(m,D);iv=i.narrow(0,IGUARD,n);ov=o.narrow(0,GUARD,n*D).view(torch.bfloat16).reshape(n,D)
    desc={k:c.descriptor(t) for k,t in dict(hidden=hv,indices=iv,output=ov).items()};record['descriptors']=desc
    need(all(d['pointer_mod64']==0 for d in desc.values()),'reviewed pointer alignment')
    if descriptors is None:descriptors=desc
    else:need(desc==descriptors,'phase B must reuse exact storages, pointers and descriptors')
    if previous is not None:
     need(previous['hidden']!=f['hidden'] and previous['indices']!=f['indices'] and previous['expected']!=f['expected'],'B changes all consumed data before launch')
     record['changed_from_A']=True
    record['launch_attempted']=True;result['kernel_launches_attempted']+=1;save()
    compiled[((n+7)//8,16,1)](hv,iv,ov,m,n,8,128)
    torch.cuda.synchronize();record['launch_completed']=True;result['kernel_launches_completed']+=1
    actual={k:read(t) for k,t in zip(('hidden','indices','output'),backings)};record['actual']={k:blob(v) for k,v in actual.items()};record['hard_checks']=checks(f,actual['hidden'],actual['indices'],actual['output']);save()
    need(all(record['hard_checks'].values()),'bit copy, poison, readonly or guard failure; full arrays saved')
    previous=f
  need(result['kernel_launches_completed']==4 and c.git('rev-parse','HEAD')==a.expected_source_commit and not c.git('status','--porcelain'),'complete four direct launches and frozen source')
  result.update(completed=True,device_numeric_verified=True,bit_copy_verified=True)
 except Exception as e:
  result['errors'].append(repr(e));(out/'failure.txt').write_text(traceback.format_exc())
 finally:
  if 'torch' in sys.modules:result['CUDA_initialized_after']=sys.modules['torch'].cuda.is_initialized()
  result['artifacts']=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='bits-summary.json'];save()
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-bits',action='store_true');p.add_argument('--root-owned-exclusive',action='store_true');p.add_argument('--expected-source-commit');p.add_argument('--compile-out',type=Path);p.add_argument('--actual-review',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
 if not a.run_bits:
  if any((a.root_owned_exclusive,a.expected_source_commit,a.compile_out,a.actual_review,a.out)):p.error('device arguments require --run-bits')
  result=plan();need(not {'torch','triton','numpy'}.intersection(sys.modules),'stdlib default')
 else:
  if not all((a.root_owned_exclusive,a.expected_source_commit,a.compile_out,a.actual_review,a.out)):p.error('Root explicit admission, exact source, old compile-out, actual review and fresh out required')
  result=execute(a)
 print(json.dumps(result,sort_keys=True,indent=2));return int(a.run_bits and not result['completed'])

if __name__=='__main__':raise SystemExit(main())
