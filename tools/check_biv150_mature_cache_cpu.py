"""CPU cache-byte/relocation and diagnostic-controller preservation contracts."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import time

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cache',ROOT/'biv150_mature_triton_cache.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);seed=root/'seed';source=root/'source';source.mkdir();cache=seed/'triton-cache/hash';cache.mkdir(parents=True)
    (seed/'manifest.json').write_text(json.dumps({'commit':'cpu-fixture','source':str(source),'small_binding':'mm','stable_small_o':'1','native_mode':'on','gate_library_sha256':'fixture-lib'}))
    (seed/'state.json').write_text(json.dumps({'complete':True,'cleanup_complete':True}))
    binary=cache/'mm_kernel.cubin';binary.write_bytes(b'CPU fixture, not executable GPU code')
    group=cache/'__grp__mm_kernel.json';group.write_text(json.dumps({'child_paths':{'mm_kernel.cubin':str(binary)}}))
    target=root/'copy';result=m.restore(seed,target,source=source,commit='cpu-fixture',binding='mm',library_sha256='fixture-lib')
    assert result['copied_files']==2 and len(result['rebased_groups'])==1 and result['kernel_artifact_bytes_unchanged']
    assert (target/'hash/mm_kernel.cubin').read_bytes()==binary.read_bytes()
    assert json.loads((target/'hash/__grp__mm_kernel.json').read_text())['child_paths']['mm_kernel.cubin']==str((target/'hash/mm_kernel.cubin').resolve())
    for field,value in [('binding','linear'),('commit','other'),('library_sha256','other')]:
        args=dict(source=source,commit='cpu-fixture',binding='mm',library_sha256='fixture-lib');args[field]=value
        try:m.restore(seed,root/('bad-'+field),**args)
        except AssertionError:pass
        else:raise AssertionError(('bad provenance accepted',field))
    (cache/'escape').symlink_to(binary)
    try:m.restore(seed,root/'symlink',source=source,commit='cpu-fixture',binding='mm',library_sha256='fixture-lib')
    except AssertionError:pass
    else:raise AssertionError('symlink accepted')
    (cache/'escape').unlink()
    external=root/'foreign';external.write_bytes(b'other');group.write_text(json.dumps({'child_paths':{'foreign':str(external)}}))
    try:m.restore(seed,root/'foreign-copy',source=source,commit='cpu-fixture',binding='mm',library_sha256='fixture-lib')
    except ValueError:pass
    else:raise AssertionError('foreign group reference accepted')

old=ast.parse((ROOT/'run_biv150_small_bf16_acceptance.py').read_text());new=ast.parse((ROOT/'run_biv150_mm_compile_diagnostic.py').read_text())
def finals(tree):return [ast.dump(item,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.Try) and n.finalbody for item in n.finalbody]
# New benchmark-window finally is intentional; compare the server-owned finally only.
def owned(tree):return [n for n in ast.walk(tree) if isinstance(n,ast.Try) and n.finalbody and any(isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and isinstance(x.value.func,ast.Name) and x.value.func.id=='remember' for x in n.finalbody)]
assert len(owned(old))==len(owned(new))==1
assert [ast.dump(x,include_attributes=False) for x in owned(old)[0].finalbody]==[ast.dump(x,include_attributes=False) for x in owned(new)[0].finalbody]
def serve(tree):return [ast.dump(n.value,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='command' for t in n.targets)]
assert serve(old)==serve(new),'Service CLI changed'
with tempfile.TemporaryDirectory() as directory:
    calls=[];sentinel=object();out=Path(directory);scope={'time':time,'json':json,'windows':[],'args':SimpleNamespace(out=out)}
    def raw(*args):calls.append(args);return sentinel
    scope['raw_benchmark']=raw
    nested=next(n for n in ast.walk(new) if isinstance(n,ast.FunctionDef) and n.name=='observed_benchmark')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[nested],type_ignores=[])),'cpu-benchmark-wrapper','exec'),scope)
    assert scope['observed_benchmark'](out,'score-4096',4096,1024,256,0) is sentinel
    assert calls==[(out,'score-4096',4096,1024,256,0)] and len(scope['windows'])==1
    def fail(*args):raise RuntimeError('original benchmark error')
    scope['raw_benchmark']=fail
    try:scope['observed_benchmark'](out,'fail',4096,1024,256,0)
    except RuntimeError as e:assert str(e)=='original benchmark error'
    else:raise AssertionError('benchmark error swallowed')
    assert scope['windows'][-1]['error']=='RuntimeError'
# Syntax-check exactly the generated bootstrap literal, without importing its dependencies.
for n in ast.walk(new):
    if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='write_text' and n.args:
        try:value=ast.literal_eval(n.args[0])
        except (ValueError,TypeError):continue
        if isinstance(value,str) and 'biv150_compile_observer' in value:ast.parse(value)
print(json.dumps({'passed':True,'gpu_executed':False,'cache_bytes_preserved':True,'group_links_relocated':True,'mismatched_provenance_rejected':3,'symlinks_and_foreign_refs_rejected':True,'owned_cleanup_ast_unchanged':True,'serve_cli_unchanged':True,'benchmark_args_returns_errors_preserved':True,'bootstrap_syntax_checked':True}))
