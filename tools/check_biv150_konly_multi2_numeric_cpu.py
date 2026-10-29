"""CPU source/coordinate/loader checks; no Torch import or device qualification."""
import argparse
import ast
import copy
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch
import biv150_konly_multi2_hoist_source as c
import biv150_konly_multi2_numeric_source as s
import biv150_konly_multi2_numeric_contract as f


def main(a):
    checks={}
    def ok(name,value):c.need(value,name);checks[name]=True
    def bad(name,call):
        try:call()
        except (ValueError,KeyError,FileNotFoundError):checks[name]=True;return
        raise AssertionError(name+' accepted')
    real=subprocess.Popen
    def local_git(argv,*args,**kwargs):
        c.need(isinstance(argv,list) and argv[:2]==['git','-C'] and argv[3] in ('show','diff','status','rev-parse'),'CPU-only subprocess tripwire')
        return real(argv,*args,**kwargs)
    with patch.object(subprocess,'Popen',local_git),tempfile.TemporaryDirectory(prefix='cjj-multi2-numeric-cpu-') as temp:
        folder=Path(temp);refdir=folder/'references';refdir.mkdir()
        for name,digest in f.REFERENCES.items():
            raw=subprocess.check_output(['git','-C',str(c.ROOT),'show',f.REFERENCE_COMMIT+':tools/'+name]);ok('original_reference_'+name,c.sha(raw)==digest);(refdir/name).write_bytes(raw)
        code,edits=s.adapted_driver();ast.parse(code)
        restored=code
        for e in reversed(edits):
            n=e['offset'];ok('inverse_'+str(len(checks)),restored[n:n+len(e['after'])]==e['after']);restored=restored[:n]+e['before']+restored[n+len(e['after']):]
        ok('exact_executed_116_driver_inverse',restored==s.original('probe_biv150_konly_q64_screen.py'))
        src,expanded=f.reference_sources(refdir);original=src['biv150_konly_q32_numeric_contract.py']
        restored=expanded
        for before,after in reversed(f.REFERENCE_EDITS):restored=restored.replace(after,before)
        ok('reference_only_coordinate_inverse',restored==original)
        originals={n.name:ast.dump(n) for n in ast.parse(original).body if isinstance(n,ast.FunctionDef)}
        adapted={n.name:ast.dump(n) for n in ast.parse(expanded).body if isinstance(n,ast.FunctionDef)}
        for name in ('metrics','gates','policy','quantize_cpu','encoded_bytes'):ok('unchanged_original_math_'+name,originals[name]==adapted[name])
        inherited=s.original('biv150_konly_q64_screen_contract.py')
        for name in ('check_loaded','execution_snapshot','execution_sources'):
            fn=next(n for n in ast.parse(inherited).body if isinstance(n,ast.FunctionDef) and n.name==name)
            selected=next(n for n in s.inherited_contract().body if isinstance(n,ast.FunctionDef) and n.name==name)
            ok('original_final_loader_gate_'+name,ast.dump(fn)==ast.dump(selected))
        for i,(qlens,klens) in enumerate(f.PHASES):
            geometry=f.phase_geometry(qlens,klens);positions=[];start=0
            for q,k in zip(qlens,klens):
                visible=[k-q+r+1 for r in range(q)];ok('causal_bounds_'+str(i)+'_'+str(start),min(visible)>0 and max(visible)==k)
                positions.extend(start+r for r in range(q));start+=q
            ok('whole_rows_covered_once_'+str(i),positions==list(range(2048)))
            ok('fixed_cache_and_grid_'+str(i),2*sum((k+15)//16 for k in klens)+5==773 and geometry['query_tiles']<=66)
        for q,k in [([True,2047],[3072,3072]),([0,2048],[3072,3072]),([1024],[6144]),([1024,1023],[3072,3072]),([1024,1024],[5120,5120]),([1024,1024],[10,6134])]:bad('bad_geometry_'+repr(q)+repr(k),lambda q=q,k=k:f.phase_geometry(q,k))
        ok('no_writer_or_timing_execution',all(x not in code for x in ('writer.warmup','writer[','elapsed_time(','perf_counter(','timed_graph_replays')))
        ok('actual_B_Graph_precedes_fresh_B',code.index("stage('Graph-B-first'")<code.index("stage('eager-B-after-Graph'"))
        ok('distinct_changed_C',"execution_stream=side if label=='side-C' else stream" in code and "changed-input side C counterexample absent" in code)
        ok('prelaunch_full_arrays_persisted',code.index("record['prelaunch_artifacts'][role]=persist")<code.index('if use_graph:replay'))
        ok('no_production_diff_main899',not subprocess.check_output(['git','-C',str(c.ROOT),'diff',f.MAIN,'--','vllm_fl','csrc']))
        reference,policy,_=f.load_reference(refdir)
        ok('original_pointwise_threshold',reference.metrics([.006],[0.])['pointwise_passed'] is False and reference.metrics([.005],[0.])['pointwise_passed'] is True)
        hard={n:True for n in ('finite','codec','output_canaries','read_only_inputs','actual_IR','hoisted_unhoisted_bitwise','quantized_reference')}
        policy(hard,True,False)
        for name in hard:bad('reject_hard_'+name,lambda name=name:policy(dict(hard,**{name:False}),True,True))
        bad('reject_BF16_without_explicit_diagnostic',lambda:policy(hard,False,False))
        ok('preserve_BF16_diagnostic_boundary',policy(hard,False,True)['original_BF16_quality_passed'] is False)
        pre=None
        if a.raw is not None:
            c.need(a.review is not None,'actual review required');pre=f.preflight(a.raw,a.review,refdir)
            ok('actual_44_originals_and_compile_review',pre['raw_manifest_sha256']==f.MANIFEST)
            groups=f.reviewed_cache.prepare(a.raw,folder/'reviewed-cache',pre['records'],f.MANIFEST)
            ok('both_original_cache_groups_copied',set(groups)==set(f.ROLES))
            for role,record in pre['records'].items():
                binary=(a.raw/'out/multi2'/role/'actual.cubin').read_bytes();r=f.RESOURCES[role]
                kernel=NS(kernel=binary,n_regs=r['n_regs'],n_spills=r['n_spills'],metadata=NS(shared=r['shared'],num_warps=16,num_stages=1,warp_size=64),_init_handles=lambda:None)
                f.check_loaded(role,kernel,record,record);ok('fake_handle_exact_gate_'+role,True)
                kernel.n_spills=1;bad('reject_runtime_spill_'+role,lambda:f.check_loaded(role,kernel,record,record));kernel.n_spills=0
                kernel.kernel=b'corrupt';bad('reject_final_cubin_'+role,lambda:f.check_loaded(role,kernel,record,record))
                snapshot=copy.deepcopy(record);snapshot['ABI']={};bad('reject_ABI_'+role,lambda:f.check_loaded(role,kernel,snapshot,record))
                group=groups[role];manager=NS(get_group=lambda name:dict(group['paths']))
                f.reviewed_cache.validate_group(manager,group)
                path=Path(next(iter(group['paths'].values())));path.write_bytes(path.read_bytes()+b'x')
                bad('reject_modified_cache_child_'+role,lambda:f.reviewed_cache.validate_group(manager,group))
        ok('no_framework_or_device_import',not {'torch','triton','numpy','vllm','vllm_fl'}.intersection(sys.modules))
        return dict(completed=True,errors=[],checks=checks,check_count=len(checks),plan=f.plan(),
            adapted_driver_sha256=c.sha(code),source_edits=len(edits),expanded_reference_sha256=c.sha(expanded),
            actual_preflight=pre,CPU_only=True,GPU_executed=False,numeric_verified=False,
            limitations='Source/geometry/fake-handle rejection checks only; real original cache, loaded ABI, Graph, full-array numerical and side C qualification await Root-owned device execution.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--raw',type=Path);parser.add_argument('--review',type=Path);args=parser.parse_args()
    print(json.dumps(main(args),indent=2,sort_keys=True,allow_nan=False))
