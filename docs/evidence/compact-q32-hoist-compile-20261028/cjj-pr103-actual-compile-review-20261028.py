#!/usr/bin/env python3
"""Finite CPU-only rereview of PR103 compile-v2 raw. Never imports a GPU framework."""
import ast,hashlib,importlib.util,json,re,struct,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
RAW=Path('/private/tmp/cjj-compact-q32-hoist-compile-v2-raw-20261028')
SRC=Path('/private/tmp/flagos-cjj-compact-q32-probe-frozen-6c0-20261028')
OUT=Path('/private/tmp/cjj-pr103-actual-compile-review-20261028.json')
COMMIT='6c08ec2561b1c3cee96838bffd4e296b757ad877';BASE='deab3ab474925c08dba48c4c8cdb00bf108b3b82'
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(p.read_text())
git=lambda *a:subprocess.check_output(['git',*a],cwd=SRC,text=True).strip()
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
m=read(RAW/'download-manifest.json')
assert m['raw_files']==50 and m['raw_bytes']==4937816
assert {str(p.relative_to(RAW)) for p in RAW.rglob('*') if p.is_file()}=={'download-manifest.json',*(x['relative'] for x in m['files'])}
for x in m['files']:
 b=(RAW/x['relative']).read_bytes();assert len(b)==x['bytes'] and sha(b)==x['sha256'],x['relative']
assert sum(x['bytes'] for x in m['files'])==m['raw_bytes']
owner=read(next((RAW/'exports').glob('*owner.json')));s=read(RAW/'out/summary.json');b=s['binding']
assert (owner['pid'],owner['start_ticks'])==(38713,354708474)
assert m['original_owner']==dict(pid=38713,start_ticks=354708474,same_start=True,state='Z')
assert owner['source_commit']==s['commit']==git('rev-parse','HEAD')==COMMIT
assert not git('status','--porcelain') and not git('diff',BASE,COMMIT,'--','vllm_fl','csrc')
sys.path.insert(0,str(SRC/'tools'));import probe_biv150_compact_q32_hoist as frontend
assert frontend.source_identity()==s['source_identity']
for path,digest in s['source_identity']['source_hashes'].items():
 assert sha((SRC/path).read_bytes())==digest
 assert (SRC/path).read_bytes()==subprocess.check_output(['git','show',COMMIT+':'+path],cwd=SRC)
remote=Path(owner['out']);source_remote=Path(owner['command'][1]).parents[1]
for name,value in s['actual_imports'].items():
 if name.startswith('vllm_fl.'):
  rel=name.replace('.','/')+'.py';assert value['path']==str(source_remote/rel) and value['sha256']==sha((SRC/rel).read_bytes())
assert owner['command']==['/usr/local/bin/python',str(source_remote/'tools/probe_biv150_compact_q32_hoist.py'),'--compile-only','--root-owned-exclusive','--commit',COMMIT,'--out',str(remote)]
assert s['mode']=='compile-only' and all(s[k] is True for k in ('completed','compile_passed','fresh_actual_compact_IR_native_resources_verified','GPU_executed','CUDA_API_entered','context_or_allocations_attempted'))
assert s['errors']==s['stages']==[] and s['compile_prerequisite'] is None
zero=('attention_launches','writer_launches','encoder_launches','Graph_replays','initialized_tensor_elements')
assert all(s[k]==0 for k in zero)
assert all(s[k] is False for k in ('numerical_verified','precision_gate','service_verified','Graph_hoisted_qualification','mixed_or_42layer_capacity_verified','performance_measured'))
pre=s['pre_CUDA_admission'];assert pre['port_probe_code']==111 and pre['port9031_closed'] and pre['root_owned_exclusive_acknowledged']
assert pre['visible_native_pids']==[] and pre['memory_used_mib']<=1024 and not pre['no_visible_pid_is_idle_proof']
# Reuse the actual frontend admission parser on original observations.
assert frontend.pre_cuda(True,(RAW/'out/prelaunch-ixsmi.txt').read_text(),(RAW/'out/prelaunch-visible-pids.txt').read_text(),True)=={k:v for k,v in pre.items() if k!='port_probe_code'}
A=RAW/'out'/Path(b['artifact_directory']).relative_to(remote)
assert read(A/'binding.json')==b
assert b['implementation_sha256']==sha(json.dumps(s['source_identity'],sort_keys=True).encode())
first=read(RAW/'out/first-unhoisted-before-admission/record.json');assert first==s['first_unhoisted']
for ext,row in first['files'].items():
 data=(RAW/'out/first-unhoisted-before-admission'/('actual.'+ext)).read_bytes()
 assert len(data)==row['bytes'] and sha(data)==row['sha256'] and data==(A/'unhoisted'/('actual.'+ext)).read_bytes()
for role in ('unhoisted','hoisted'):
 assert read(A/role/'record.json')==b[role]
 for ext,digest in b[role]['artifacts'].items():assert sha((A/role/('actual.'+ext)).read_bytes())==digest
 assert sha((A/role/'actual.jit_python_source.py').read_bytes())==b[role]['source_digest']
for key in ('source_digest','function_name','ABI','launch_geometry'):assert b['unhoisted'][key]==b['hoisted'][key]
assert (A/'unhoisted/actual.ttir').read_bytes()==(A/'hoisted/actual.ttir').read_bytes()
assert (A/'unhoisted/actual.source').read_bytes()==(A/'hoisted/actual.source').read_bytes()
source=(SRC/'vllm_fl/patches/biv150_konly_compact_q32.py').read_text()
fn=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name=='kernel_cjj_konly_q32')
assert ast.get_source_segment(source,fn).rstrip()==(A/'unhoisted/actual.jit_python_source.py').read_text().rstrip()
patch=SRC/'vllm_fl/patches'
proposal=module('compact_proposal_review',patch/'biv150_konly_compact_hoist_proposal.py')
native=module('encoded_native_review',patch/'biv150_konly_encoded_hoist_native.py')
proposed,proof=proposal.propose((A/'unhoisted/actual.ttgir').read_text(),b['actual_metadata'])
assert proposed==(A/'override.ttgir').read_text() and sha(proposed.encode())==b['override_sha256'] and proof==b['proposal']
actual=native.audit(proposed,(A/'hoisted/actual.ttgir').read_text(),(A/'hoisted/actual.llir').read_text(),(A/'hoisted/actual.asm').read_text())
assert actual==b['native']
before=native.llvm_observation((A/'unhoisted/actual.llir').read_text())
assert len(before['Q_shared_reads_inside_KV'])==16 and actual['LLVM']['Q_shared_reads_inside_KV']==[]
assert b['actual_metadata']['pins']==proposal.PINS and b['actual_metadata']['target']==dict(backend='corex',arch=71,warp_size=64)
assert b['actual_metadata']['grid']==[33,2] and b['actual_metadata']['query_rows']==1024
assert s['owner_epoch']==dict(generation=1,registry_component_only=True,shape=[139,2,16,2,128],storage_offset=128)
# Actual binder key recorded by pinned runtime, not reconstructed with a mock binder.
key=b['actual_JIT_specialization_options_key'];sep=key.index(']{')+1
specializations=ast.literal_eval(key[:sep]);options=ast.literal_eval(key[sep:])
assert len(specializations)==len(fn.args.args)
abi_by_name=dict(zip((x.arg for x in fn.args.args),specializations))
assert abi_by_name['stride_k_cache_0']==('constexpr',12544) and abi_by_name['stride_v_cache_0']==('constexpr',6272)
assert abi_by_name['key_cache_ptr']==('*i8','D') and abi_by_name['value_cache_ptr']==('*bf16','D') and abi_by_name['key_code_scale_ptr']==('*fp32','D')
assert options==dict(num_warps=16,num_stages=1,debug=False,instrumentation_mode='')
# Reuse prior stdlib ELF/msgpack reader; never load binary code.
sys.path.insert(0,'/private/tmp/flagos-cjj-native-k8-vfirst-20261028/tools')
import audit_biv150_native_k8_compile_cpu as elfutil
binary={};cache_rows={}
for path in (RAW/'out/triton-cache').glob('*/kernel_cjj_konly_q32.json'):
 value=read(path);cache_rows[value['hash']]=(path.parent,value)
assert set(cache_rows)=={b['unhoisted']['hash'],b['hoisted']['hash']}
for role in ('unhoisted','hoisted'):
 folder,value=cache_rows[b[role]['hash']]
 for ext,digest in b[role]['artifacts'].items():assert sha((folder/('kernel_cjj_konly_q32.'+ext)).read_bytes())==digest
 assert value['target']==b['actual_metadata']['target'] and [value[x] for x in ('num_warps','num_stages','warp_size')]==[16,1,64]
 assert value['shared']==b['resources'][role]['shared']
 assert value['ir_override']==(b['override_path'] if role=='hoisted' else None)
 raw=(A/role/'actual.cubin').read_bytes();elf=elfutil.elf(raw);assert elf['machine']==248 and elf['extent']==len(raw)
 note=elf['sections']['.note']['bytes'];at=0;meta=None
 while at<len(note):
  ns,ds,kind=struct.unpack_from('<III',note,at);at+=12;name=note[at:at+ns].rstrip(b'\0');at=(at+ns+3)//4*4;desc=note[at:at+ds];at=(at+ds+3)//4*4
  if name==b'Iluvatar' and kind==10:assert meta is None;meta=elfutil.msgpack(desc)
 assert len(meta['iluvatar.kernels'])==1;kernel=meta['iluvatar.kernels'][0]
 assert kernel['.name']=='kernel_cjj_konly_q32' and kernel['.warp_size']==64
 assert kernel['.vgpr_count']==b['resources'][role]['n_regs']
 assert kernel['.vgpr_spill_count']==kernel['.private_segment_fixed_size']==kernel['.group_segment_fixed_size']==0
 assert kernel['.sgpr_count']==104 and kernel['.sgpr_spill_count']==(99 if role=='unhoisted' else 96)
 asm=(A/role/'actual.asm').read_text()
 for field in ('.vgpr_count','.sgpr_count','.vgpr_spill_count','.sgpr_spill_count','.private_segment_fixed_size','.group_segment_fixed_size','.warp_size'):
  assert re.search(r'^\s*'+re.escape(field)+r':\s+'+str(kernel[field])+r'\s*$',asm,re.M),field
 assert '; ScratchSize: 0' in asm
 resource=b['resources'][role]
 assert resource==dict(known=True,n_regs=135 if role=='unhoisted' else 122,n_spills=0,num_stages=1,num_warps=16,phase='handles_initialized_pre_reader',shared=57344 if role=='unhoisted' else 33792,warp_size=64)
 binary[role]=dict(cubin_sha256=sha(raw),cubin_bytes=len(raw),text_sha256=sha(elf['sections']['.text']['bytes']),text_bytes=len(elf['sections']['.text']['bytes']),ELF_metadata=kernel,compiler_cache_metadata=value,loaded_resources=resource)
assert binary['unhoisted']['ELF_metadata']['.args']==binary['hoisted']['ELF_metadata']['.args']
# Source-bound launch corrections; original report remains byte-identical.
prior=Path('/private/tmp/cjj-pr103-next-compile-review-20261028.json');assert sha(prior.read_bytes())=='2b965bec8663366d3bc49a5a5c15db41be35541f805b3efd18da71be29077d14'
old=Path('/private/tmp/cjj-compact-q32-hoist-compile-v1-20261028-launch.json')
v1=Path('/private/tmp/cjj-compact-q32-hoist-compile-v1-20261028-root-reviewed-launch.json')
v2=Path('/private/tmp/cjj-compact-q32-hoist-compile-v2-20261028-root-reviewed-launch.json')
a,c,d=map(read,(old,v1,v2));assert {k for k in a if a[k]!=c[k]}=={'base'} and c['base']==d['base']==BASE
assert c['commit']==d['commit']==COMMIT and {k for k in c if c[k]!=d[k]}=={'tag','args'}
assert c['args'][:-1]==d['args'][:-1] and d['args'][-1]==str(remote)
launcher=Path('/private/tmp/flagos_launch_personal_component_v3_bundle_base_20261027.py');assert sha(launcher.read_bytes())==owner['launcher_sha256']
assert 'spec["base"], "--", "vllm_fl", "csrc"' in launcher.read_text()
old_diffs=git('diff','--name-only',a['base'],COMMIT,'--','vllm_fl','csrc').splitlines();assert len(old_diffs)==32
erratum=dict(prior_report_preserved=str(prior),prior_report_SHA256=sha(prior.read_bytes()),correction='Prior statement that existing launch arguments/specs needed no changes omitted the outer launcher base gate and output-parent requirement. The probe-internal6c/deab/source identity and thresholds were unchanged.',outer_launcher=dict(path=str(launcher),sha256=sha(launcher.read_bytes()),original_base=a['base'],changed_production_paths=old_diffs,corrected_comparison_base=BASE,diff_deab_production_csrc_empty=True),attempts=[dict(tag=c['tag'],source_commit=COMMIT,status='Root reports failed in args.out.mkdir due to missing parent, before Torch/CUDA import; this review does not possess its original exception log.',device_compilation_pass=False,actual_probe_semantics='mkdir(exist_ok=False), not parents=True; occurs before framework imports',out=c['args'][-1]),dict(tag=d['tag'],source_commit=COMMIT,status='Original v2 completed compile-only, same source and compiler arguments; output changed to fresh top-level directory with existing parent.',out=d['args'][-1],original_owner=m['original_owner'])],source_or_threshold_changes=False,current_main_qualification=False,numeric_launch_needs='Root-reviewed numeric spec: outer base deab, receipt at actual v2 top-level output, fresh top-level numeric output. Internal source6c and gates unchanged.',Root_local_review_KeyError='Root reports one local summary-check attempt used nonexistent passed field and raised KeyError before corrected checks; it is not a device failure and is not independently represented in these50 raw files.')
err=Path('/private/tmp/cjj-pr103-next-compile-review-erratum-20261028.json');err.write_text(json.dumps(erratum,ensure_ascii=False,indent=2)+'\n')
report=dict(CPU_only_review=True,reviewer_GPU_SSH_PR_actions=False,original_files=m['raw_files'],original_bytes=m['raw_bytes'],all_raw_SHA256_verified=True,source_commit=COMMIT,production_binding_commit=BASE,source_identity=s['source_identity'],actual_imports=s['actual_imports'],local_source_identity_and_Git_blobs_verified=True,owner=m['original_owner'],pre_CUDA=pre,mode=s['mode'],compile_completed=True,compile_passed=True,zero_frontend_counters={k:s[k] for k in zero},actual_GPU_context_and_allocations=True,CUDA_initialized_after_torch_import=s['CUDA_initialized_after_torch_import'],source_ABI_TTIR_equal_across_arms=True,source_kernel_function_text_matches_frozen_module=True,first_unhoisted_all_artifacts_match_adapter_original=True,proposal_replay_exact=proof,actual_native_replay_exact=actual,unhoisted_LLVM_observation=before,binder=dict(recorded_actual_key=key,ordered_source_arguments_checked=len(specializations),target=b['actual_metadata']['target'],pins=b['actual_metadata']['pins'],runtime_binder_not_reexecuted_by_CPU_review=True),artifacts={role:b[role]['artifacts'] for role in ('unhoisted','hoisted')},binary=binary,complete_binary_ABI28_args_equal=True,resource_interpretation='Runtime n_spills=0 is preserved as reported. ELF VGPR spill0/private segment0 but SGPR spill99→96; no blanket all-spills-zero or occupancy/speedup claim. ELF fixed shared0 is distinct from Triton dynamic shared57344→33792.',boundary=dict(numerical_verified=False,Graph_hoisted_qualification=False,Graph_or_side_numerics_verified=False,model105_verified=False,service_verified=False,precision_gate=False,mixed_42layer_capacity_verified=False,performance_verified=False,current_main_qualification=False,old_16KiB_qualification_transferred=False),limits=['Fresh actual compiler TTGIR/LLVM/ASM are independently replayed through exact pinned CPU auditors. Cubin SHA, ELF machine, complete ABI and resource notes are parsed and tied to matching preserved/cache artifacts; no independent cubin instruction disassembly was supplied or run locally.','Installed SDK/helper/JIT pins and live loaded-resource/object/binder values are original runtime attestations under the verified frontend/source. CPU review rehashes local frozen sources and archived artifacts; it does not requery the driver or reload cubins.','Zero attention/writer/encoder/Graph/input-init counts are verified frontend provenance, not a device-wide profiler trace. Real CUDA initialization/allocations/init_handles occurred.','V1 missing-parent failure and local check KeyError are explicitly attributed to Root, not misclassified as v2 device failure.'],erratum=dict(path=str(err),sha256=sha(err.read_bytes())),review_script_sha256=sha(Path(__file__).read_bytes()),review_dependencies_SHA256={str(p):sha(p.read_bytes()) for p in (patch/'biv150_konly_compact_hoist_proposal.py',patch/'biv150_konly_encoded_hoist_native.py',Path(elfutil.__file__))})
assert not {'torch','triton','numpy','vllm_fl'}.intersection(sys.modules)
OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(report=str(OUT),sha256=sha(OUT.read_bytes()),erratum=str(err),erratum_sha256=sha(err.read_bytes()),compile_only=True,numerical_verified=False),indent=2))
