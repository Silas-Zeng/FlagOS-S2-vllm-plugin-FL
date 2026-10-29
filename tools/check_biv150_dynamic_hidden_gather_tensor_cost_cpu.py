#!/usr/bin/env python3
"""Bounded stdlib checks for private Tensor baseline/full-frame adaptation only."""
import ast,inspect,json,subprocess,sys,tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import run_biv150_dynamic_hidden_gather_tensor_cost as x
c=x.c;checks=[]
def ok(name,value=True):c.need(value,name);checks.append(name)
def bad(name,call):
 try:call()
 except (ValueError,AssertionError):checks.append(name);return
 raise AssertionError('negative accepted '+name)
def run():
 p=x.plan();ok('default does not import Torch/Triton/NumPy',not {'torch','triton','numpy'}.intersection(sys.modules))
 ok('same existing ordinary and Plan fixture geometries',p['cases']==[[65,9,'ordinary'],[65,9,'Plan'],[2018,61,'ordinary'],[2018,61,'Plan']])
 hidden=object();controls=NS(hidden=hidden,arm='candidate');head=NS(compute_logits=lambda value:('head',value));model=x.TensorModel(controls,head)
 ok('ON returns exact same real-hidden object',model() is hidden);controls.arm='control';ok('OFF returns exact same object, no index_select proxy',model() is hidden)
 hidden2=object();controls.hidden=hidden2;ok('changed input not cached across calls',model() is hidden2);ok('original controlled no-op head retained',model.compute_logits(hidden2)==('head',hidden2))
 generated,edits=x.transformed_execute();original=inspect.getsource(x.base.execute)
 for before,after in reversed(edits):ok('unique bounded private timing edit '+before[:28],generated.count(after)==1);generated=generated.replace(after,before)
 ok('exact inverse restores all original numeric/codec/ABI/receipt/refusal gates',generated==original)
 ok('warmup8 and before/after outside timing markers',len(edits)==3 and 'range(8)' in x.transformed_execute()[0] and "_tensor_cost_block('after'" in x.transformed_execute()[0])
 worker=ast.parse((x.ROOT/'vllm_fl/worker/worker.py').read_text());found=[]
 for node in ast.walk(worker):
  if isinstance(node,ast.If) and ast.unparse(node.test)=='fl_envs.USE_FLAGGEMS':found.append(node)
 ok('one actual Worker FlagGems boundary',len(found)==1)
 call=next(v for v in ast.walk(found[0]) if isinstance(v,ast.Call) and ast.unparse(v.func)=='flag_gems.enable' and not any(k.arg=='unused' for k in v.keywords))
 ok('default branch exact record/once/path API',[(k.arg,ast.unparse(k.value)) for k in call.keywords]==[('record','should_record'),('once','True'),('path','fl_envs.FLAGGEMS_ENABLE_OPLIST_PATH')])
 source=Path(x.__file__).read_text();tree=ast.parse(source)
 install=next(v for v in ast.walk(tree) if isinstance(v,ast.FunctionDef) and v.name=='install')
 call=next(v for v in ast.walk(install) if isinstance(v,ast.Call) and ast.unparse(v.func)=='flag_gems.enable')
 ok('adapter uses same rank0 defaults without unused/include overrides',[(k.arg,ast.unparse(k.value)) for k in call.keywords]==[('record','True'),('once','True'),('path','envs.FLAGGEMS_ENABLE_OPLIST_PATH')])
 ok('policy rejects contradictory inherited filters',"name not in os.environ" in source and 'get_flag_gems_whitelist_blacklist()==(None,None)' in source)
 ok('FlagGems cache/log paths set before original worker imports SDK',"pre-framework private FlagGems paths" in source and source.index("os.environ['FLAGGEMS_CACHE_DIR']=str(a.out")<source.index('result=run(a)'))
 ok('dispatch observed on actual same hidden tensor',"frame.f_code in self.index_codes" in source and "frame.f_locals.get('inp') is self.current.controls.hidden" in source)
 ok('actual OFF must reach index implementation before timing',"len(self.dispatch)>start" in source and "len(self.numeric)==8" in source)
 ok('only numeric profiling, restores original callbacks',source.count('sys.setprofile(previous)')==2 and 'def profile' in source and 'def block' in source)
 ok('actual complete baseline arrays and GPU hidden-after saved',"actual_control=self.blob(self.read(control))" in source and "actual_hidden_after=self.blob(self.read(entire))" in source)
 ok('actual source/options/typed ABI/resources/kernel/launcher exported',all(t in source for t in ('kernel.asm.items()','typed_ABI','resources=c.resources(kernel)','actual-launcher.so','actual_options=c.metadata(kernel)')))
 ok('backend compile barrier and full runtime cache comparisons during timing',"prohibit_backend_compile(type(backend))" in source and 'before==after' in source and 'dict(object=id(v),hash=str(v.hash))' in source)
 with tempfile.TemporaryDirectory(prefix='cjj-tensor-cost-cpu-') as tmp:
  root=Path(tmp);a=x.Adapter(NS(out=root),'derived');(root/'flaggems-cache').mkdir();p1=root/'flaggems-cache/code.py';p1.write_text('x=1\n');p2=root/'flaggems-cache/bench.db-wal';p2.write_bytes(b'one')
  bad('unknown/unobserved OFF dispatch cannot enter timing',lambda:a.block('before',None,'control'))
  before=a.files();p1.write_text('x=2\n');ok('shader source mutation changes tracked timing inventory',a.files()!=before);before=a.files();p2.write_bytes(b'two');ok('SQLite WAL mutation changes tracked timing inventory',a.files()!=before)
 ok('public production files exact parent',not c.git('diff',x.PARENT,'--','vllm_fl','csrc'))
 ok('original full helper worker byte exact parent',(x.ROOT/'tools/run_biv150_dynamic_hidden_gather_helper.py').read_bytes()==subprocess.check_output(['git','-C',str(x.ROOT),'show',x.PARENT+':tools/run_biv150_dynamic_hidden_gather_helper.py']))
 ok('no source/SDK/numeric/perf/service qualification fabricated',p['actual_device_work']==0 and p['performance_verified'] is p['service_verified'] is False)
 return dict(passed=True,CPU_only=True,checks=checks,check_count=len(checks),parent=x.PARENT,adapter_sha256=c.sha(Path(x.__file__).read_bytes()),production_diff=False,actual_dispatch_pending=True,actual_Tensor_baseline_numeric_pending=True,timing_pending=True,service_verified=False)
if __name__=='__main__':print(json.dumps(run(),sort_keys=True,indent=2))
