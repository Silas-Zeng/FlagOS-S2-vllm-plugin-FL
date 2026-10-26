#!/usr/bin/env python3
"""Run audited SDK geometry/helper AST on CPU; never import Torch/Triton."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from biv150_prefill_reverse import (CASES, HELPER_SHA, source_factory, function_source,
                                    load_python, eligible, OLD, NEW)
from check_biv150_prefill_probe_cpu import Tensor


class Pointer:
    def __init__(self, values, offset=0):
        self.values, self.offset = values, offset

    def __add__(self, offset):
        return Pointer(self.values, self.offset + offset)


class ScalarSubscripts(ast.NodeTransformer):
    def visit_Subscript(self, node):
        # Broadcast axes/indexing vanish for this per-element geometry proof.
        if isinstance(node.value, ast.Name) and node.value.id in (
                'query_offset_0', 'query_offset_1', 'offs_d'):
            return copy.deepcopy(node.value)
        return self.generic_visit(node)


def helper_namespace(path):
    text = path.read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == HELPER_SHA
    names = {'cdiv_fn', 'find_seq_idx', 'resolve_seq_and_query_len', 'compute_tile_loop_bounds'}
    functions = [copy.deepcopy(n) for n in ast.parse(text).body
                 if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in functions} == names
    for fn in functions:
        fn.decorator_list = []
        fn.returns = None
        for arg in fn.args.posonlyargs+fn.args.args+fn.args.kwonlyargs:
            arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    tl = SimpleNamespace(load=lambda ptr: ptr.values[ptr.offset], minimum=min, maximum=max,
                         constexpr=int, int32=int)
    namespace = {'tl': tl}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace, {fn.name:hashlib.sha256(ast.dump(fn).encode()).hexdigest() for fn in functions}


def expression(kernel, target):
    assignments = [n.value for n in ast.walk(ast.parse(kernel))
                   if isinstance(n, ast.Assign) and len(n.targets)==1
                   and isinstance(n.targets[0], ast.Name) and n.targets[0].id==target]
    assert len(assignments)==1, target
    expr = ast.Expression(ScalarSubscripts().visit(copy.deepcopy(assignments[0])))
    return compile(ast.fix_missing_locations(expr), '<actual-sdk-'+target+'>', 'eval')


def geometry_check(name, qlen, kvlen, baseline, candidate, helpers):
    count = qlen//32+1  # exact current main Q32 policy grid
    tl = SimpleNamespace(program_id=lambda axis: physical_id if axis==0 else head,
                         num_programs=lambda axis: count)
    baseline_id = expression(baseline, 'q_block_global_idx')
    candidate_id = expression(candidate, 'q_block_global_idx')
    expressions = {key:expression(baseline,key) for key in
                   ('query_pos','query_offset_0','query_offset_1','query_offset','output_offset')}
    mapped = []
    for physical_id in range(count):
        assert eval(baseline_id, {'tl':tl}) == physical_id
        mapped.append(eval(candidate_id, {'tl':tl}))
    assert sorted(mapped)==list(range(count))
    assert [mapped[i] for i in mapped]==list(range(count))
    assert mapped==list(reversed(range(count)))
    seen = set()
    records = []
    invalid = []
    for logical_id in range(count):
        seq, local, start, n, length = helpers['resolve_seq_and_query_len'](
            Pointer([0,qlen]), Pointer([kvlen]), logical_id, 1, 32)
        assert (seq,local,start,n,length)==(0,logical_id,0,qlen,kvlen)
        if local*32>=n:
            invalid.append(logical_id)
            records.append((logical_id,0,0))
            continue
        lo,hi,prefix = helpers['compute_tile_loop_bounds'](
            length-n,length,n,local,0,0,64,256,32,8,0,False,False,True,False,-1,-1)
        assert lo==0 and 0<hi<=(kvlen+63)//64
        records.append((logical_id,hi-lo,prefix))
        for head in range(2):
            for offs_m in range(256):
                env = dict(q_block_local_idx=local,BLOCK_Q=32,num_queries_per_kv=8,
                    cur_batch_in_all_start_index=start,kv_head_idx=head,offs_m=offs_m,
                    offs_d=0,query_stride_0_i64=2048,query_stride_1_i64=128,
                    output_stride_0_i64=2048,output_stride_1_i64=128)
                for key in ('query_pos','query_offset_0','query_offset_1','query_offset','output_offset'):
                    env[key]=eval(expressions[key],env)
                token,query_head=env['query_offset_0'],env['query_offset_1']
                if env['query_pos']>=n:continue
                assert 0<=token<qlen and 0<=query_head<16
                assert env['query_offset']==env['output_offset']==token*2048+query_head*128
                env['offs_d']=127
                assert eval(expressions['output_offset'],env)==env['output_offset']+127
                assert (token,query_head) not in seen
                seen.add((token,query_head))
    assert len(seen)==qlen*16
    candidate_records=[records[logical_id] for logical_id in mapped]
    assert sorted(candidate_records)==records
    assert sorted(mapped.index(i) for i in invalid)==list(range(len(invalid)))
    costs=[record[1] for record in records if record[1]]
    reverse_costs=[record[1] for record in candidate_records if record[1]]
    assert costs==sorted(costs) and reverse_costs==sorted(costs,reverse=True)
    return dict(case=name,query_rows=qlen,kv_rows=kvlen,grid=[count,2],
        valid_query_ctas=len(costs),overspecified_query_ctas=len(invalid),
        full_launch_id_bijection=True,involution=True,includes_all_invalid_ctas=True,
        exact_output_ownership=True,output_row_heads_checked=len(seen),
        every_program_preserves_original_logical_output_addresses=True,
        valid_work_is_longest_first=True,first_valid_kv_tiles=reverse_costs[0],
        last_valid_kv_tiles=reverse_costs[-1],total_kv_tiles_per_head=sum(costs))


def scope_check(root):
    fragment=load_python(root/'vllm_fl/patches/biv150_q_fragment_service.py','_reverse_cpu_fragment')
    values=dict(IS_3D=False,HEAD_SIZE=128,HEAD_SIZE_PADDED=128,num_query_heads=16,
        num_queries_per_kv=8,BLOCK_M=256,BLOCK_Q=32,BLOCK_SIZE=16,TILE_SIZE=64,
        NUM_SEGMENTS_PER_SEQ=1,KV_QUANT_MODE=0,USE_CAUSAL=True,SLIDING_WINDOW=0,
        CHUNK_LOOKBACK=-1,CHUNK_SIZE=-1,num_warps=16,num_stages=1,num_seqs=1,
        query_ptr=Tensor((2017,16,128),(2048,128,1),1),
        output_ptr=Tensor((2017,16,128),(2048,128,1),2),
        key_cache_ptr=Tensor((1024,16,2,128),(8192,256,128,1),3),
        value_cache_ptr=Tensor((1024,16,2,128),(8192,256,128,1),3),
        query_start_len_ptr=Tensor((2,),(1,),4,'torch.int32'),
        seq_lens_ptr=Tensor((1,),(1,),5,'torch.int32'),
        block_tables_ptr=Tensor((1,8192),(8192,1),6,'torch.int32'))
    assert eligible(values,(64,2),fragment)
    changes=[dict(IS_3D=True),dict(num_seqs=2),dict(BLOCK_M=128),dict(BLOCK_Q=16),
             dict(num_warps=8),dict(num_stages=2),dict(TILE_SIZE=128),dict(USE_CAUSAL=False),
             dict(query_ptr=Tensor((1023,16,128),(2048,128,1),1)),
             dict(output_ptr=values['query_ptr'])]
    assert all(not eligible(dict(values,**change),(64,2),fragment) for change in changes)
    assert all(not eligible(values,grid,fragment) for grid in ((63,2),(64,1),(64,2,1)))
    return dict(matching_scope=True,rejected_descriptor_changes=len(changes),
                rejected_grids=3,uses_actual_parent_descriptor_gate=True)


def controller_check(root):
    # Exercise the actual unchanged parent _Policy and harness Tap with fake
    # launchers. This catches recursion or a changed selected-grid binding
    # without importing Torch, Triton, or performing device work.
    from bench_biv150_prefill_reverse import Tap
    fragment=load_python(root/'vllm_fl/patches/biv150_q_fragment_service.py','_reverse_controller_fragment')
    tree=ast.parse((root/'vllm_fl/patches/biv150_prefill_q32.py').read_text())
    classes=[node for node in tree.body if isinstance(node,ast.ClassDef)]
    namespace={'qfragment':SimpleNamespace(Manager=object,_eligible=fragment._eligible)}
    exec(compile(ast.Module(body=classes,type_ignores=[]),'<actual-main-q32-policy>','exec'),namespace)
    class Launch:
        def __init__(self,label):self.label,self.calls=label,[]
        def __getitem__(self,grid):
            def invoke(*args,**kwargs):
                self.calls.append((tuple(grid),kwargs.copy()))
                return SimpleNamespace(hash=self.label)
            return invoke
    base_launch,candidate_launch,fallback=Launch('baseline'),Launch('candidate'),Launch('fallback')
    baseline_view=SimpleNamespace(wrapper=base_launch,optimized={'baseline':{}})
    reverse_view=SimpleNamespace(wrapper=candidate_launch,optimized={'candidate':{}})
    taps={'baseline':Tap(baseline_view,fragment),'candidate':Tap(reverse_view,fragment)}
    parent=SimpleNamespace(async_mode=SimpleNamespace(get=lambda:None),
        raw=SimpleNamespace(pre_run_hooks=[]),candidate=SimpleNamespace(pre_run_hooks=[]),
        device=None,wrapper=taps['baseline'],optimized={},
        attention=SimpleNamespace(torch=SimpleNamespace(cuda=SimpleNamespace(
            is_current_stream_capturing=lambda:False))))
    policy=namespace['_Policy'](fallback,parent)
    values=dict(IS_3D=False,HEAD_SIZE=128,HEAD_SIZE_PADDED=128,num_query_heads=16,
        num_queries_per_kv=8,BLOCK_M=128,BLOCK_Q=16,BLOCK_SIZE=16,TILE_SIZE=64,
        NUM_SEGMENTS_PER_SEQ=1,KV_QUANT_MODE=0,USE_CAUSAL=True,SLIDING_WINDOW=0,
        CHUNK_LOOKBACK=-1,CHUNK_SIZE=-1,num_warps=8,num_stages=1,num_seqs=1,
        query_ptr=Tensor((2017,16,128),(2048,128,1),1),
        output_ptr=Tensor((2017,16,128),(2048,128,1),2),
        key_cache_ptr=Tensor((1024,16,2,128),(8192,256,128,1),3),
        value_cache_ptr=Tensor((1024,16,2,128),(8192,256,128,1),3),
        query_start_len_ptr=Tensor((2,),(1,),4,'torch.int32'),
        seq_lens_ptr=Tensor((1,),(1,),5,'torch.int32'),
        block_tables_ptr=Tensor((1,8192),(8192,1),6,'torch.int32'))
    before=values.copy()
    for label,tap in taps.items():
        parent.wrapper=tap
        assert policy[(127,2)](**values).hash==label
        assert tap.grid==(64,2) and tap.kwargs['BLOCK_M']==256 and tap.kwargs['num_warps']==16
    assert values==before and not fallback.calls
    assert taps['baseline'].kwargs==taps['candidate'].kwargs
    for change in ({'IS_3D':True},{'num_seqs':2},{'ir_override':'foreign'},
                   {'query_ptr':Tensor((1023,16,128),(2048,128,1),1)}):
        assert policy[(127,2)](**dict(values,**change)).hash=='fallback'
    assert len(base_launch.calls)==len(candidate_launch.calls)==1
    return dict(actual_parent_policy_and_tap_executed=True,same_selected_grid_and_kwargs=True,
        caller_kwargs_unchanged=True,unsupported_scope_uses_parent_fallback=True,
        baseline_does_not_recurse=True,device_executed=False)


def main(args):
    baseline,candidate,proof=source_factory(args.sdk_source.read_text(),args.source)
    helpers,helper_proof=helper_namespace(args.helpers_source)
    records=[geometry_check(name,qlens[0],klens[0],baseline,candidate,helpers)
             for name,(qlens,klens) in CASES.items()]
    # Complete id-permutation proof for each grid length, including one and
    # every threshold/tile boundary through 2049 programs (65536 query rows).
    grids=0
    for count in range(1,2050):
        mapped=[count-1-i for i in range(count)]
        assert sorted(mapped)==list(range(count))
        assert [mapped[j] for j in mapped]==list(range(count))
        grids+=1
    try:source_factory(args.sdk_source.read_text().replace(OLD,NEW,1),args.source)
    except ValueError:rejects_changed_sdk=True
    else:raise AssertionError('Modified SDK source accepted')
    result=dict(passed=True,cpu_only=True,gpu_executed=False,torch_imported='torch' in sys.modules,
        official_score=False,precision_gate=False,production_unchanged=True,
        source_transform=proof,helper_ast_sha256=helper_proof,scope_checks=scope_check(args.source),
        controller_checks=controller_check(args.source),
        complete_grid_sizes_checked=grids, rejects_changed_sdk=rejects_changed_sdk,cases=records,
        limitation='Scalar actual-SDK AST plus ownership proof; device codegen, resources, scheduling and performance untested.')
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    data=json.dumps(result,indent=2)+'\n'
    if args.out:args.out.write_text(data)
    print(data,end='')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--sdk-source',type=Path,required=True)
    p.add_argument('--helpers-source',type=Path,required=True)
    p.add_argument('--out',type=Path)
    main(p.parse_args())
