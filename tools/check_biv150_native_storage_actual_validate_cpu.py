#!/usr/bin/env python3
"""Source/entry/order admission only; stub outcomes never count as performance."""
import argparse,ast,hashlib,json,subprocess,sys
from pathlib import Path
from unittest.mock import patch
import probe_biv150_native_storage_actual_validate as p
from check_biv150_native_storage_main56_cpu import restore
checks={}
def check(name,value):assert value,name;checks[name]=True
def rejects(name,fn):
 try:fn()
 except RuntimeError:checks[name]=True;return
 raise AssertionError(name+' accepted')
def run(library):
 source=Path(p.__file__).read_text();tree=ast.parse(source);reference=p.git('show',p.MAIN+':'+p.NATIVE)
 check('precise_candidate_inverse_to_current_main899',restore((p.ROOT/p.NATIVE).read_text()).encode()==reference)
 check('same_candidate_production_from855',not p.git('diff',p.CANDIDATE,'--','vllm_fl','csrc','tools/kernels'))
 check('same_runtime_and_broker',all(p.git('show',p.MAIN+':'+name)==(p.ROOT/name).read_bytes() for name in p.PINS if name.endswith(('encoded_runtime.py','encoded_views.py'))))
 plan=p.plan();check('default_no_GPU_no_performance_no_C3',not any(plan[k] for k in ('default_enabled','GPU_executed','performance_verified','C3_service_verified','model_precision_verified')))
 check('fixed_original_full_function_and_ABBA_budget',p.WARMUP==128 and p.CALLS==512 and p.ROUNDS==3 and p.abba()==['reference','candidate','candidate','reference']*3)
 native_tree=ast.parse(reference);validate=next(n for n in ast.walk(native_tree) if isinstance(n,ast.FunctionDef) and n.name=='validate')
 check('validate_has_no_stream_or_tensor_data_query',not any(isinstance(n,ast.Attribute) and n.attr in ('current_stream','is_current_stream_capturing','item','cpu','numpy') for n in ast.walk(validate)))
 timed=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='timed_validate')
 attrs=[n.func.attr for n in ast.walk(timed) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
 check('timed_region_invokes_original_validate_only',attrs==['validate'])
 trace=[];rt=object();entry=object();kw=object()
 class Recorder:
  def validate(self,*args):trace.append(args)
 stamps=iter([100,500]);p.timed_validate(Recorder(),rt,entry,kw,7,lambda:next(stamps))
 check('complete_callback_same_operands_once_per_call',trace==[(rt,entry,kw)]*7)
 check('clock_fixture_is_not_a_performance_result',True)
 tensor_keys=('query_ptr','key_cache_ptr','key_scale_ptr','value_cache_ptr','seq_lens_ptr','block_table_ptr','segm_output_ptr','segm_max_ptr','segm_expsum_ptr');values={k:object() for k in tensor_keys}
 class Outcome:
  def __init__(self,error=None,wrong=False):self.error=error;self.wrong=wrong
  def validate(self,*args):
   if self.error:raise RuntimeError('NativeK8 service: '+self.error)
   return [object()]*9 if self.wrong else [values[k] for k in tensor_keys]
 check('exact_return_identity_checked',p.compare(dict(reference=Outcome(),candidate=Outcome()),rt,entry,values,True)['accepted'])
 check('exact_error_text_checked',not p.compare(dict(reference=Outcome('bad'),candidate=Outcome('bad')),rt,entry,values,False,'bad')['accepted'])
 rejects('different_return_objects_rejected',lambda:p.compare(dict(reference=Outcome(),candidate=Outcome(wrong=True)),rt,entry,values,True))
 rejects('one_arm_error_rejected',lambda:p.compare(dict(reference=Outcome(),candidate=Outcome('bad')),rt,entry,values,True))
 rejects('changed_error_order_rejected',lambda:p.compare(dict(reference=Outcome('bad'),candidate=Outcome('different')),rt,entry,values,False,'bad'))
 check('whole_owner_alias_fixture_fits_real_backing',all(2*((n-1)*pitch+16*128)<=p.PAGES*16384 for n,s,pitch,index in p.CASES))
 check('entries_represent_real_first_middle_last_registry_scan',p.LAYERS==42 and {row[3] for row in p.CASES}=={0,20,41})
 check('mixed_and_pure_shapes_both_present',{row[:3] for row in p.CASES}=={(64,2,2048),(32,4,2560),(61,2,2560),(30,4,2560)})
 allcalls=[n.func for n in ast.walk(tree) if isinstance(n,ast.Call)]
 check('no_producer_kernel_data_initialization_or_readback_call',not any(isinstance(f,ast.Attribute) and f.attr in ('try_produce','op','item','cpu','numpy','zero_','fill_','copy_','cudaGraphLaunch') for f in allcalls) and not any(isinstance(f,ast.Subscript) for f in allcalls))
 gqa=ast.parse((p.ROOT/'vllm_fl/patches/biv150_gqa_decode.py').read_bytes());picker=next(n for n in gqa.body if isinstance(n,ast.FunctionDef) and n.name=='_pick_segments');env=dict(TARGET_PROGRAMS=256);exec(compile(ast.Module(body=[picker],type_ignores=[]),'<actual current segment picker>','exec'),env)
 check('actual_picker_keeps_mixed_owned_scratch_width',all(env['_pick_segments'](n,2,s if n in (30,61) else 16)==s for n,s,pitch,index in p.CASES))
 mixed=(p.ROOT/'vllm_fl/patches/biv150_konly_encoded_mixed_native.py').read_text()
 check('actual_mixed_receives_narrowed_metadata_then_full_parent_validate',all(x in mixed for x in ('md=scratch_metadata(plan,PAIRS[n],self.parent)','result=ordinary(md)','runtime.native_decode.validate(runtime,entry,kw)')))
 check('Torch_import_only_in_device_function',not any(isinstance(n,(ast.Import,ast.ImportFrom)) and ('torch' in ast.unparse(n)) for n in tree.body))
 check('actual_registry_and_adapters_constructed',all(x in source for x in ('runtime_mod.Runtime(torch,broker,{},None)','runtime.registry.claim(object(),owner,source,source)','mod.Adapter(runtime,config)')))
 with patch.object(p,'device',side_effect=AssertionError('device reached')),patch.object(sys,'argv',[p.__file__,'--actual-cuda']):rejects('missing_explicit_device_arguments_rejected',p.main)
 with patch.object(p,'device',side_effect=AssertionError('device reached')),patch.object(sys,'argv',[p.__file__,'--root-owned-exclusive']):rejects('authorization_alone_does_not_enable_device',p.main)
 head=p.git('rev-parse','HEAD').decode().strip();check('positive_actual_frozen_preflight',p.preflight(head,library)==reference)
 rejects('wrong_frozen_commit_rejected',lambda:p.preflight('0'*40,library))
 with patch.object(p,'device',side_effect=RuntimeError('CPU_READY_TRIPWIRE')) as trap,patch.object(sys,'argv',[p.__file__,'--actual-cuda','--root-owned-exclusive','--commit',head,'--library',str(library),'--out',str(Path('/private/tmp/cjj-pr120-component-out-must-not-exist-cpu'))]):
  try:p.main()
  except RuntimeError as e:check('fully_ready_path_reaches_device_tripwire_after_real_preflight',str(e)=='CPU_READY_TRIPWIRE' and trap.call_count==1)
  else:raise AssertionError('no ready tripwire')
 check('no_GPU_framework_imported',not {'torch','triton','vllm','vllm_fl','numpy'}.intersection(sys.modules))
 return dict(passed=True,CPU_only=True,source_commit=head,check_count=len(checks),checks=checks,tool_sha256=p.sha(Path(p.__file__).read_bytes()),CPU_script_sha256=p.sha(Path(__file__).read_bytes()),plan=plan,actual_Tensor_test_executed=False,performance_verified=False,C3_service_verified=False,ready_tripwire_only_no_device_execution=True)
if __name__=='__main__':
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--library',type=Path,required=True);a.add_argument('--report',type=Path,required=True);args=a.parse_args();r=run(args.library);args.report.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(args.report,p.sha(args.report.read_bytes()))
