"""Finite N1 S1/S2 numerical adapter of the immutable 522c screen, not production."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
BASE='899ce77470b0b539aebc108fd122f668c2d5894d'
PARENT='2e0929793d9e744a964c719e27d253927bcb52f3'
TEMPLATE='522c0b0617a8d2d862f60ebb03f05f6e08ccfcfe'
PINS={
 'biv150_konly_q64_compile_source.py':'1ed4e3e752b1c43a31fafa8e6ace5802ee932409aa3e8bc9f1423ffed3f1621d',
 'biv150_konly_q64_compile_arguments.py':'b82f517b94b196822f00037773b1c80806716ac77d9ef44881829ef1fb8fa52f',
 'biv150_konly_q64_screen_contract.py':'912e485d189f611c571a4439e6356c72fd699c80a88eb8f8db92bea23a67ac31',
 'biv150_konly_q64_reviewed_cache.py':'9d38d2a69946e7ba9c8670714e732c9ba1cb5d04834e40b595bda04195812a1f',
 'probe_biv150_konly_q64_screen.py':'c186982bc964a1eef66c9d5edd228b6d13978b4a129a850bb38bdc03c293efea',
 'probe_biv150_konly_q64_compile.py':'749a662b7e203640c43b4282ae70abed318792dff302c7ef88ecab534fc02c7a',
}


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(data):return hashlib.sha256(data.encode() if isinstance(data,str) else data).hexdigest()


def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])


def original(name):
    data=git('show',TEMPLATE+':tools/'+name)
    need(sha(data)==PINS[name],'immutable 522c dependency '+name)
    return data.decode()


def replace(text,edits):
    old=text
    for before,after in edits:
        need(text.count(before)==1 and after not in text,'unique adapter seam '+before[:90])
        text=text.replace(before,after,1)
    restored=text
    for before,after in reversed(edits):
        need(restored.count(after)==1,'inverse adapter seam')
        restored=restored.replace(after,before,1)
    need(restored==old,'complete screen inverse')
    return text


_DEPENDENCIES=None
def dependencies():
    """Stdlib helpers only. No Torch/Triton import, compiler or file export."""
    global _DEPENDENCIES
    if _DEPENDENCIES is not None:return _DEPENDENCIES
    modules={}
    for filename in PINS:
        if filename=='probe_biv150_konly_q64_screen.py':continue
        name=filename[:-3]
        need(name not in sys.modules,'private helper module already occupied '+name)
        m=types.ModuleType(name);m.__file__=str(ROOT/'tools'/filename)
        sys.modules[name]=m
        exec(compile(original(filename),'<immutable-522c/'+filename+'>','exec'),vars(m))
        modules[name]=m
    c=modules['biv150_konly_q64_compile_source'];c.ROOT=ROOT;c.BASE=BASE
    _DEPENDENCIES=types.SimpleNamespace(c=c,
        contract=modules['biv150_konly_q64_screen_contract'],
        cache=modules['biv150_konly_q64_reviewed_cache'],
        compiler=modules['probe_biv150_konly_q64_compile'],
        kwargs=modules['biv150_konly_q64_compile_arguments'].kwargs)
    return _DEPENDENCIES


def run_edits():
    # Numeric/codec, archive, Graph-before-fresh and nondefault-stream bodies
    # remain from 522c. Only provenance/role/stage/reference-length seams move.
    return [
      ("        result['reviewed_cache']=reviewed_cache.prepare(a.raw,a.out/'triton-cache',pre['records'],f.MANIFEST);save()",
       "        f.admit(a,pre)\n        result.update(Root_numeric_admission=True,Root_numeric_admission_receipt_sha256=a.admission_sha256,reader_cache_warmups_observed=0,writer_warmups_observed=0)\n        result['reviewed_cache']=f.prepare_cache(a.out/'triton-cache',pre);save()\n        a.references=a.out/'reference-sources'\n        f.export_references(a.references)"),
      ("        c.need(interval(guarded['q32'])!=interval(guarded['q64']),'role output storage alias')",
       "        c.need(interval(guarded['s1'])!=interval(guarded['s2']),'role output storage alias')"),
      ("dict(readonly_tensors,output_q32=guarded['q32'],output_q64=guarded['q64'])",
       "dict(readonly_tensors,output_s1=guarded['s1'],output_s2=guarded['s2'])"),
      ("            bq=int(role[1:]);paths=pre['runtime_paths'][role];source=Path(paths['source']);override=Path(paths['override'])",
       "            bq=32;paths=pre['runtime_paths'][role];source=Path(paths['source']);override=Path(paths['override'])"),
      ("            c.need(override.read_bytes()==(a.raw/'out'/role/'override.ttgir').read_bytes(),'original override path/content '+role)",
       "            c.need(c.sha(override.read_bytes())==paths['override_sha256'],'original override path/content '+role)"),
      ("            module=c.load_file('cjj_q64_screen_control' if bq==32 else 'cjj_q64_screen_candidate',source)",
       "            module=c.load_file(paths['module_name'],source)"),
      ("            launch_options[role]=dict(num_warps=16,num_stages=1,ir_override=str(override))",
       "            launch_options[role]=dict(num_warps=16,num_stages=f.STAGES[role],ir_override=str(override))"),
      ("            snapshot=adapter.kernel_snapshot(kernel,fn);adapter.Adapter._preserve(a.out/role,kernel,snapshot)",
       "            result['reader_cache_warmups_observed']+=1\n            recorder=f.snapshot_module(role)\n            snapshot=recorder.kernel_snapshot(kernel,fn);adapter.Adapter._preserve(a.out/role,kernel,snapshot)"),
      ("        wk=writer.warmup(*writer_args,grid=(2048,2),num_warps=1,num_stages=1);wk._init_handles()",
       "        wk=writer.warmup(*writer_args,grid=(2048,2),num_warps=1,num_stages=1);result['writer_warmups_observed']+=1;wk._init_handles()"),
      ("            cpu=fixture.cpu_inputs([2048],[6144],20261025,pattern)",
       "            length=f.LENGTHS[pattern]\n            prefix=length-2048\n            cpu=fixture.cpu_inputs([2048],[length],20261025,pattern)"),
      ("                visible=4096+torch.arange(2048)+1", "                visible=prefix+torch.arange(2048)+1"),
      ("            positions=torch.arange(4096,6144,dtype=torch.int64)",
       "            positions=torch.arange(prefix,length,dtype=torch.int64)"),
      ("actual_roles='hoisted Q32 and hoisted Q64'", "actual_roles='main899 N1 hoisted S1 and preheader N1 hoisted S2'"),
      ("byte_cpu(outputs['q32'])==byte_cpu(outputs['q64'])", "byte_cpu(outputs['s1'])==byte_cpu(outputs['s2'])"),
      ("legacy_policy_bitwise_key_means='actual hoisted Q32 versus actual hoisted Q64'",
       "legacy_policy_bitwise_key_means='actual N1 hoisted S1 versus actual N1 hoisted S2'"),
    ]


def run_source():
    text=original('probe_biv150_konly_q64_screen.py')
    tree=ast.parse(text);fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='run')
    before='\n'.join(text.splitlines()[fn.lineno-1:fn.end_lineno])+'\n'
    # Remove the entire old timing region, including its warmups. Keep a byte
    # inverse of this removal as well as every other explicit seam above.
    start=before.index('        for mode in f.MODES:\n            with torch.cuda.stream(stream):\n                for role in f.ROLES:\n                    for _ in range(8):replay(role,mode)')
    end=before.index('        result.update(completed=True,performance_measured=True,',start)
    timing=before[start:end]
    value=replace(before,run_edits()+[(timing,'        # No timing or timing warmups in this numerical-only frontend.\n'),
        ('completed=True,performance_measured=True,','completed=True,performance_measured=False,'),
        ("timing_scope='reader: complete Q quantization/staging/QK/softmax/PV/epilogue; append-reader: unchanged stride2560 K/V writer plus same complete reader; no service claim'",
         "timing_scope=None,reader_ABBA_pending=True,full_service_verified=False")])
    need(not any(isinstance(n,ast.Attribute) and n.attr in ('elapsed_time','Event','perf_counter') for n in ast.walk(ast.parse(value))),
         'timing remains in numeric frontend')
    return value


def run(args,pre):
    import errno,gzip,inspect,os,socket,time
    import biv150_konly_q32_s2_numeric_contract as contract
    d=dependencies();ns=dict(errno=errno,gzip=gzip,inspect=inspect,os=os,socket=socket,time=time,
        json=json,Path=Path,sys=sys,subprocess=subprocess,c=d.c,f=contract,compiler=d.compiler,
        reviewed_cache=d.cache,kwargs=d.kwargs)
    exec(compile(run_source(),'<522c-N1-S1-S2-numeric-only>','exec'),ns)
    return ns['run'](args,pre)
