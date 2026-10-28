#!/usr/bin/env python3
"""Finite CPU-only recheck of PR103's six full1024 numeric artifacts."""
import ast,hashlib,importlib.util,json,re,struct,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
R=Path('/private/tmp/cjj-compact-q32-hoist-numeric-v1-raw-20261028')
C=Path('/private/tmp/cjj-compact-q32-hoist-compile-v2-raw-20261028')
S=Path('/private/tmp/flagos-cjj-compact-q32-probe-frozen-6c0-20261028')
O=Path('/private/tmp/cjj-pr103-actual-numeric-review-20261028.json')
COMMIT='6c08ec2561b1c3cee96838bffd4e296b757ad877';sha=lambda b:hashlib.sha256(b).hexdigest();read=lambda p:json.loads(p.read_text())
def file_sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def mod(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
sys.path.insert(0,str(S/'tools'));import probe_biv150_compact_q32_hoist as frontend
c=frontend.c
m=read(R/'download-manifest.json');assert (m['raw_files'],m['raw_bytes'])==(71,374807676)
assert {str(p.relative_to(R)) for p in R.rglob('*') if p.is_file()}=={'download-manifest.json',*(x['relative'] for x in m['files'])}
for x in m['files']:assert (R/x['relative']).stat().st_size==x['bytes'] and file_sha(R/x['relative'])==x['sha256'],x['relative']
assert sum(x['bytes'] for x in m['files'])==m['raw_bytes']
archive=Path('/private/tmp/cjj-compact-q32-hoist-numeric-v1-raw-20261028.tar-stream-v1.tar.gz')
assert archive.stat().st_size==369081417 and file_sha(archive)=='149c2af1339ff47dfec8114c3ed68e5c073e511e8d8575cb80c2967aa4992f1a'
owner=read(next((R/'exports').glob('*owner.json')));s=read(R/'out/summary.json');cs=read(C/'out/summary.json');b=s['binding']
assert m['original_owner']==dict(pid=38862,start_ticks=354791020,same_start=True,state='Z')
assert (owner['pid'],owner['start_ticks'])==(38862,354791020) and owner['source_commit']==s['commit']==cs['commit']==COMMIT
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip()==COMMIT
assert subprocess.check_output(['git','diff',c.BASE,COMMIT,'--','vllm_fl','csrc'],cwd=S)==b''
assert frontend.source_identity()==s['source_identity']==cs['source_identity']
assert owner['bundle_sha256']==read(next((C/'exports').glob('*owner.json')))['bundle_sha256']
prereq=s['compile_prerequisite'];assert prereq==dict(fresh_numeric_reaudit_required=True,old_hash_not_an_admission_license=True,path='/workspace/experiments/minicpm-s2/cjj-compact-q32-hoist-compile-v2-20261028',summary_sha256=file_sha(C/'out/summary.json'))
assert cs['completed'] and cs['compile_passed'] and cs['errors']==[]
remote=Path(owner['out']);source_remote=Path(owner['command'][1]).parents[1]
assert owner['command']==['/usr/local/bin/python',str(source_remote/'tools/probe_biv150_compact_q32_hoist.py'),'--gpu','--root-owned-exclusive','--commit',COMMIT,'--out',str(remote),'--compile-receipt',prereq['path'],'--numeric-diagnostic']
assert s['mode']=='gpu-numeric' and s['errors']==[] and all(s[k] is True for k in ('completed','compile_passed','fresh_actual_compact_IR_native_resources_verified','requested_component_gates_passed','full_1024_final_output_verified','numerical_verified','Graph_verified','Graph_bypass_verified','Graph_changed_output_counterexample','side_stream_Q32_verified'))
assert all(s[k] is False for k in ('original_BF16_proxy_passed','precision_gate','service_verified','Graph_hoisted_qualification','full_validation_matrix_verified','mixed_or_42layer_capacity_verified','performance_measured'))
assert (s['attention_launches'],s['Graph_replays'],s['writer_launches'],s['encoder_launches'])==(10,4,0,0)
assert s['pointwise_atol']==s['pointwise_rtol']==s['original_BF16_RRMSE_proxy']==.005
assert s['partial']==s['maximum']==s['summation']=='not_exposed_by_2D_Q32'
assert s['side_stream']['stream']!=s['side_stream']['default'] and s['side_stream']['waited_for_input_stream']
for name,value in s['actual_imports'].items():
 if name.startswith('vllm_fl.'):
  rel=name.replace('.','/')+'.py';assert value['path']==str(source_remote/rel) and value['sha256']==file_sha(S/rel)
pre=s['pre_CUDA_admission'];assert pre['port_probe_code']==111 and frontend.pre_cuda(True,(R/'out/prelaunch-ixsmi.txt').read_text(),(R/'out/prelaunch-visible-pids.txt').read_text(),True)=={k:v for k,v in pre.items() if k!='port_probe_code'}
A=R/'out'/Path(b['artifact_directory']).relative_to(remote);assert read(A/'binding.json')==b
assert b['implementation_sha256']==sha(json.dumps(s['source_identity'],sort_keys=True).encode())
first=read(R/'out/first-unhoisted-before-admission/record.json');assert first==s['first_unhoisted']
for ext,record in first['files'].items():
 data=(R/'out/first-unhoisted-before-admission'/('actual.'+ext)).read_bytes();assert len(data)==record['bytes'] and sha(data)==record['sha256'] and data==(A/'unhoisted'/('actual.'+ext)).read_bytes()
for role in c.ROLES:
 assert read(A/role/'record.json')==b[role]
 for ext,digest in b[role]['artifacts'].items():assert file_sha(A/role/('actual.'+ext))==digest
 assert file_sha(A/role/'actual.jit_python_source.py')==b[role]['source_digest']
 assert str(source_remote/'vllm_fl/patches/biv150_konly_compact_q32.py') in (A/role/'actual.source').read_text().splitlines()[0]
for k in ('source_digest','function_name','ABI','launch_geometry'):assert b['unhoisted'][k]==b['hoisted'][k]
assert (A/'unhoisted/actual.ttir').read_bytes()==(A/'hoisted/actual.ttir').read_bytes()
assert (A/'unhoisted/actual.source').read_bytes()==(A/'hoisted/actual.source').read_bytes()
source=(S/'vllm_fl/patches/biv150_konly_compact_q32.py').read_text();fn=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name=='kernel_cjj_konly_q32')
assert ast.get_source_segment(source,fn).rstrip()==(A/'unhoisted/actual.jit_python_source.py').read_text().rstrip()
proposal=mod('compact_numeric_proposal',S/'vllm_fl/patches/biv150_konly_compact_hoist_proposal.py');native=mod('compact_numeric_native',S/'vllm_fl/patches/biv150_konly_encoded_hoist_native.py')
override,proof=proposal.propose((A/'unhoisted/actual.ttgir').read_text(),b['actual_metadata']);assert override==(A/'override.ttgir').read_text() and proof==b['proposal']
audit=native.audit(override,(A/'hoisted/actual.ttgir').read_text(),(A/'hoisted/actual.llir').read_text(),(A/'hoisted/actual.asm').read_text());assert audit==b['native']
assert len(native.llvm_observation((A/'unhoisted/actual.llir').read_text())['Q_shared_reads_inside_KV'])==16
assert b['actual_metadata']['pins']==proposal.PINS and b['actual_metadata']['target']==dict(backend='corex',arch=71,warp_size=64)
assert b['resources']==cs['binding']['resources']
key=b['actual_JIT_specialization_options_key'];sep=key.index(']{')+1;specs=ast.literal_eval(key[:sep]);options=ast.literal_eval(key[sep:]);assert len(specs)==len(fn.args.args)
by_name=dict(zip((x.arg for x in fn.args.args),specs));assert by_name['stride_k_cache_0']==('constexpr',12544) and by_name['stride_v_cache_0']==('constexpr',6272) and by_name['key_code_scale_ptr']==('*fp32','D')
assert options==dict(num_warps=16,num_stages=1,debug=False,instrumentation_mode='')
assert s['owner_epoch']==dict(generation=1,registry_component_only=True,shape=[139,2,16,2,128],storage_offset=128)
normal=read(R/'out/normalized2d128/actual-metadata.json');assert normal['hoist_verified'] is False and [normal['resources'][k] for k in ('num_warps','num_stages','warp_size')]==[8,1,64]
assert all(x==dict(actual_path='normalized2d128',hash=normal['hash'],hoist_verified=False,role=role,service_hoisted_capture_supported=False) for x,role in zip(s['capture_records'],c.ROLES))
sys.path.insert(0,'/private/tmp/flagos-cjj-native-k8-vfirst-20261028/tools');import audit_biv150_native_k8_compile_cpu as elfutil
cache={}
for p in (R/'out/triton-cache').glob('*/kernel_cjj_konly*.json'):
 v=read(p)
 if 'hash' in v:cache[v['hash']]=(p.parent,v)
assert set(cache)=={b['unhoisted']['hash'],b['hoisted']['hash'],normal['hash']}
binaries={}
for role,path,hash_id,resources in [('unhoisted',A/'unhoisted',b['unhoisted']['hash'],b['resources']['unhoisted']),('hoisted',A/'hoisted',b['hoisted']['hash'],b['resources']['hoisted']),('normalized2d128',R/'out/normalized2d128',normal['hash'],normal['resources'])]:
 folder,meta=cache[hash_id];kernel_name=meta['name']
 for ext in ('source','ttir','ttgir','llir','asm','cubin'):assert (folder/(kernel_name+'.'+ext)).read_bytes()==(path/('actual.'+ext)).read_bytes()
 assert meta['target']==b['actual_metadata']['target'] and meta['shared']==resources['shared'] and [meta[x] for x in ('num_warps','num_stages','warp_size')]==[resources[x] for x in ('num_warps','num_stages','warp_size')]
 assert meta['ir_override']==(b['override_path'] if role=='hoisted' else None)
 blob=(path/'actual.cubin').read_bytes();elf=elfutil.elf(blob);assert elf['machine']==248 and elf['extent']==len(blob)
 note=elf['sections']['.note']['bytes'];at=0;md=None
 while at<len(note):
  ns,ds,k=struct.unpack_from('<III',note,at);at+=12;n=note[at:at+ns].rstrip(b'\0');at=(at+ns+3)//4*4;d=note[at:at+ds];at=(at+ds+3)//4*4
  if n==b'Iluvatar' and k==10:assert md is None;md=elfutil.msgpack(d)
 assert len(md['iluvatar.kernels'])==1;km=md['iluvatar.kernels'][0]
 assert km['.name']==kernel_name and km['.vgpr_count']==resources['n_regs'] and resources['n_spills']==0
 assert km['.vgpr_spill_count']==km['.private_segment_fixed_size']==0
 if role in c.ROLES:assert km['.sgpr_spill_count']==(99 if role=='unhoisted' else 96)
 assembly=(path/'actual.asm').read_text()
 for field in ('.vgpr_count','.sgpr_count','.vgpr_spill_count','.sgpr_spill_count','.private_segment_fixed_size','.group_segment_fixed_size','.warp_size'):assert re.search(r'^\s*'+re.escape(field)+r':\s+'+str(km[field])+r'\s*$',assembly,re.M)
 binaries[role]=dict(compiler_hash=hash_id,cubin_sha256=sha(blob),ELF_resources={k:v for k,v in km.items() if k!='.args'},loaded_resources=resources,complete_binary_ABI=km['.args'],cache_metadata=meta)
assert binaries['unhoisted']['complete_binary_ABI']==binaries['hoisted']['complete_binary_ABI']
print('Fresh IR/ABI/binder/native and binary resources passed; regenerating full references.',flush=True)
# Reuse the pinned original canonical generator, exact RN32/I8 packer and full-D128 INT64/FP64 oracle.
phase_cache={};phase_reports=[]
for phase in range(3):
 started=time.monotonic();cpu=c.canonical(phase,np);encoded,kc,ks,repairs=c.packed(cpu,np);refs,refproof=c.references(cpu,kc,ks,np)
 expected=np.full(c.PAGES*12544+2*c.GUARD,0xa5,dtype=np.uint8);expected[c.GUARD:-c.GUARD]=encoded.reshape(-1)
 readonly={k:cpu[name].view(np.uint8).reshape(-1) for k,name in [('q','q_bits'),('table','table'),('lengths','lengths'),('cu_q','cu_q')]}
 readonly.update(k_descale=np.array([1.],dtype=np.float32).view(np.uint8),v_descale=np.array([1.],dtype=np.float32).view(np.uint8),writer_source_k=np.full((1,2,128),0x4040,np.uint16).view(np.uint8).reshape(-1),writer_source_v=np.full((1,2,128),0x40a0,np.uint16).view(np.uint8).reshape(-1))
 phase_cache[phase]=(cpu,refs,expected,readonly)
 phase_reports.append(dict(phase=phase,codec_boundary_repairs=repairs,reference=refproof,reference_SHA256={k:sha(v.tobytes()) for k,v in refs.items()},encoded_cache_sha256=sha(expected.tobytes())))
 print('Regenerated phase',phase,'all1024/full128 in',round(time.monotonic()-started,2),'CPU seconds',flush=True)
# The canonical arrays and independent RN32/FP64 implementation are not GPU outputs.
expected_stages=[('eager-A',0,'q32'),('capture-normalized-A-and-replay',0,'normalized2d128'),('Graph-B-first',1,'normalized2d128'),('fresh-normalized-eager-B',1,'normalized2d128'),('fresh-Q32-eager-B',1,'q32'),('side-Q32-C',2,'q32')]
assert [x['name'] for x in s['stages']]==[x[0] for x in expected_stages]
assert [x['name'] for x in s['incomplete_stage_records']]==[x[0] for x in expected_stages] and all(x['completed'] for x in s['incomplete_stage_records'])
stage_reports=[];outputs={};max_ref_difference=0.;max_metric_difference=0.
def same(a,b):return a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b)
def metric_match(actual,saved):
 global max_metric_difference
 assert actual.keys()==saved.keys()
 for k,v in actual.items():
  if type(v) in (bool,int) or v is None:assert v==saved[k]
  else:
   delta=abs(v-saved[k]);max_metric_difference=max(max_metric_difference,delta)
   assert np.isclose(v,saved[k],rtol=1e-12,atol=1e-15), (k,v,saved[k])
for (name,phase,path_kind),stage in zip(expected_stages,s['stages']):
 cpu,refs,expected,readonly=phase_cache[phase]
 assert stage['phase']=={k:phase_reports[phase][k] for k in ('phase','codec_boundary_repairs','reference')}
 file=R/'out'/(name+'.npz');data=np.load(file,allow_pickle=False)
 expected_keys=set(cpu)|set(refs)|{'cache_before_u8','expected_cache_u8'}|{'readonly_before_'+k for k in readonly}
 for role in c.ROLES:expected_keys|={'guarded_before_'+role,'guarded_after_'+role,'output_'+role,'cache_after_'+role}|{'readonly_after_'+role+'_'+k for k in readonly}
 assert set(data.files)==expected_keys and len(data.files)==42
 for k,v in cpu.items():assert same(data[k],v),(name,k)
 ref_errors={}
 for k,v in refs.items():
  saved=data[k];assert saved.shape==(1024,16,128) and saved.dtype==np.float64 and np.isfinite(saved).all()
  delta=float(np.max(np.abs(saved-v)));max_ref_difference=max(max_ref_difference,delta);ref_errors[k]=delta
  assert delta<=1e-12,'CPU FP64 host replay mismatch; no device gate changed'
 for k in ('cache_before_u8','expected_cache_u8'):assert same(data[k],expected)
 for k,v in readonly.items():assert same(data['readonly_before_'+k],v)
 metrics={};output_bits={}
 for role,record in zip(c.ROLES,stage['records']):
  assert record==dict(actual_path=path_kind,full_guard_ok=True,hash=b[role]['hash'] if path_kind=='q32' else normal['hash'],hoist_verified=path_kind=='q32' and role=='hoisted',readonly_unchanged=True,role=role,whole_cache_unchanged=True)
  before=data['guarded_before_'+role];after=data['guarded_after_'+role];out=data['output_'+role]
  assert before.shape==after.shape==(1088,16,128) and before.dtype==after.dtype==np.uint16 and out.shape==(1024,16,128) and out.dtype==np.uint16
  for v in (before,after):assert (v[:32]==0x428e).all() and (v[-32:]==0x428e).all()
  assert np.isnan(c.decode(before[32:-32],np)).all() and np.isfinite(c.decode(after[32:-32],np)).all() and same(out,after[32:-32])
  assert same(data['cache_after_'+role],expected)
  for k,v in readonly.items():assert same(data['readonly_after_'+role+'_'+k],v)
  decoded=c.decode(out,np);qm=c.metric(decoded,data['quantized_FP64'],np);om=c.metric(decoded,data['original_FP64'],np)
  metric_match(qm,stage['metrics'][role]['quantized_FP64']);metric_match(om,stage['metrics'][role]['original_BF16'])
  regenerated_quant=c.metric(decoded,refs['quantized_FP64'],np);regenerated_original=c.metric(decoded,refs['original_FP64'],np)
  assert qm['finite'] and qm['pointwise_passed'] and qm['pointwise_violations']==0 and regenerated_quant['pointwise_violations']==0
  assert om['relative_rmse']>.005 and regenerated_original['relative_rmse']>.005
  assert om['pointwise_passed'] is False and om['pointwise_violations']==regenerated_original['pointwise_violations']
  metrics[role]=dict(saved_reference_quantized=qm,saved_reference_original_BF16=om,regenerated_quantized=regenerated_quant,regenerated_original_BF16=regenerated_original,output_u16_sha256=sha(out.tobytes()))
  output_bits[role]=out
 assert same(data['guarded_after_unhoisted'],data['guarded_after_hoisted']) and same(output_bits['unhoisted'],output_bits['hoisted'])
 assert all(stage['hard_gates'].values()) and c.policy(stage['hard_gates'],False,True)=={k:stage[k] for k in c.policy(stage['hard_gates'],False,True)}
 assert stage['precision_gate'] is stage['model_verified'] is stage['service_verified'] is stage['original_BF16_proxy_passed'] is stage['thresholds_changed'] is False
 outputs[name]=output_bits
 stage_reports.append(dict(name=name,phase=phase,actual_path=path_kind,arrays=42,output_elements_per_arm=1024*16*128,full_guarded_pair_bitwise=True,all_readonly_expected_and_after_byte_equal=True,whole_cache_expected_and_after_byte_equal=True,FP64_replay_max_abs_differences=ref_errors,original_BF16_proxy_passed=False,precision_gate=False,metrics=metrics))
 data.close();print('Rechecked full stage',name,flush=True)
for role in c.ROLES:
 assert not same(outputs['capture-normalized-A-and-replay'][role],outputs['Graph-B-first'][role])
 assert same(outputs['Graph-B-first'][role],outputs['fresh-normalized-eager-B'][role])
 assert not same(outputs['fresh-Q32-eager-B'][role],outputs['side-Q32-C'][role])
# Count only explicit frontend fill/copy elements, with the same fixed tensors.
per_phase=c.PAGES*12544+2*c.GUARD+1024*16*128+8192+1+2+1+1+2*256
per_poison=(1088+1024)*16*128
assert 3*per_phase+6*2*per_poison==s['initialized_tensor_elements']==63454479
report=dict(CPU_only_review=True,reviewer_SSH_GPU_service_push=False,source_commit=COMMIT,source_identity=s['source_identity'],original_owner=m['original_owner'],raw_files=71,raw_bytes=374807676,all_transport_and_artifact_hashes_rechecked=True,archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=file_sha(archive)),transport_history='Root reports JSON collector600s timeout then binary gzip-tar collection; this is the same38862 same-start terminal owner, not a rerun. Collector says read_only/model_rerun false.',compile_prerequisite=prereq,compile_receipt_source_identity_matches=True,fresh_numeric_compiler_binding=b['native_binding_id'],fresh_native_audit=audit,complete_typed_ABI_and_source_TTIR_equal=True,binder=dict(actual_key=key,pins=b['actual_metadata']['pins'],target=b['actual_metadata']['target'],CPU_review_does_not_reexecute_live_binder=True),binaries=binaries,resource_interpretation='Loaded n_spills0 is distinct from ELF SGPR spill counters. Q32 SGPR99→96, VGPR spill0/private0. All dynamic/fixed shared and normalized resources are retained; no blanket all-spill-zero, occupancy or throughput claim.',all_3_full1024_INT64_FP64_references_regenerated=True,phase_reports=phase_reports,stages=stage_reports,FP64_replay_max_abs_difference=max_ref_difference,saved_metric_recompute_max_abs_difference=max_metric_difference,CPU_reference_replay_tolerance='1e-12 absolute only for cross-host FP64 recomputation; original device pointwise .005+.005*abs(reference) and original BF16 RRMSE.005 are unchanged and checked against saved plus regenerated arrays.',array_check_counts=dict(total_NPZ_arrays=6*42,pair_bitwise_full_guarded=6,full_output_elements_checked=6*2*1024*16*128,guard_regions_before_after=6*2*2*2,readonly_canonical_and_after_arrays=6*8*3,whole_cache_expected_before_and_after_arrays=6*4,same_path_GraphB_vs_fresh_eager_arms=2),Graph=dict(actual_path='normalized2d128 unhoisted for both role labels',changed_A_to_B_verified=True,GraphB_precedes_fresh_eagerB_in_source_and_stage_records=True,same_path_fresh_B_bitwise=True,hoisted_Q32_capture_qualified=False),side=dict(actual_path='Q32 unhoisted vs fresh hoisted',changed_B_to_C_verified=True,explicit_distinct_stream=s['side_stream']),counters={k:s[k] for k in ('attention_launches','Graph_replays','writer_launches','encoder_launches','initialized_tensor_elements')},all_12_original_BF16_proxy_failures_preserved=True,boundary=dict(component_quantized_reference_gates_passed=True,full_validation_matrix_verified=False,Graph_hoisted_qualification=False,model105_verified=False,precision_gate=False,service_verified=False,capacity42layer_verified=False,performance_verified=False,old16KiB_qualification_transferred=False),limits=['Q32 2D exposes final output only; no O/M/L tensors are claimed.','The FP64 oracle uses full-D128 independent INT64 quantized QK and FP64 dequantization/softmax/PV; probability BF16/online-rounding is explicitly not modeled.','Graph/stream chronology and live compiler-object/binder/resource identity are source-bound original frontend records plus full changed-input arrays, not a device-wide execution trace.','Compiler-produced actual LLVM/ASM audit is replayed. Cubin hashes, ELF metadata and complete ABI are rechecked, but originals do not contain independent cubin instruction disassembly.','BF16 proxy RRMSE about0.0083–0.0086 is above unchanged0.005 in every arm/stage; numeric-diagnostic preserves this failure and never establishes model quality.'],replay=dict(script=str(Path(__file__)),script_sha256=file_sha(Path(__file__)),python=sys.version,numpy=np.__version__,dependencies_sha256={str(p):file_sha(p) for p in (S/'tools/biv150_compact_q32_probe_contract.py',S/'tools/probe_biv150_compact_q32_hoist.py',S/'vllm_fl/patches/biv150_konly_compact_hoist_proposal.py',S/'vllm_fl/patches/biv150_konly_encoded_hoist_native.py',Path(elfutil.__file__))}))
assert not {'torch','triton','vllm_fl'}.intersection(sys.modules)
O.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(report=str(O),sha256=file_sha(O),all_component_checks_passed=True,original_BF16_proxy_passed=False,precision_gate=False,max_host_reference_difference=max_ref_difference),indent=2),flush=True)
