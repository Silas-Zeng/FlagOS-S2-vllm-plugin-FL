#!/usr/bin/env python3
"""Mixed reader source, compact-address and scalar-reference contracts; stdlib only."""
import ast
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

from biv150_konly_i8_qk_reference import bf16
import biv150_konly_mixed_factory as factory
import biv150_konly_mixed_reference as reference

ROOT = Path(__file__).resolve().parents[1]


def reject(call):
    try:
        call()
    except (ValueError, AssertionError):
        return
    raise AssertionError('Invalid mixed contract was accepted')


def run():
    dependencies = json.loads((ROOT / 'docs/evidence/konly-mixed-decode-20261027/dependency-provenance.json').read_text())
    for item in dependencies['copied_files']:
        data = (ROOT / item['path']).read_bytes()
        assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256']
    main = (ROOT / 'vllm_fl/patches/biv150_gqa_decode.py').read_text()
    sources, proof = factory.build(main)
    mutants = 0
    for mode, source in sources.items():
        producer = factory.span(source, factory.NAMES[mode]); reducer = factory.span(source, '_reduce_mixed_partials')
        factory.restore_producer(producer, mode); factory.restore_reducer(reducer)
        for fn in (producer, reducer):
            tree = ast.parse(fn)
            loads = sorted([n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'tl.load'], key=lambda n: n.lineno)
            assert all('query_start_len_ptr' in ast.unparse(n) for n in loads[:2])
            body = tree.body[0].body
            guard = next(n for n in body if isinstance(n, ast.If) and 'q_end - q_start' in ast.unparse(n.test))
            assert len(guard.body) == 1 and isinstance(guard.body[0], ast.Return)
            assert loads[1].lineno < guard.lineno < loads[2].lineno
            assert all('tl.int64' in ast.unparse(n) for n in body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('q_start', 'q_end') for t in n.targets))
        assert 'token_idx = seq_idx.to(tl.int64)' in producer
        assert '        q_start * query_stride_token' in producer
        assert 'output_ptr + q_start * output_stride_token' in reducer
        for old, new in [('        q_start * query_stride_token', '        token_idx * query_stride_token'),
                         ('token_idx * segment_stride_token', 'q_start * segment_stride_token'),
                         ('accumulator = accumulator * rescale[:, None]', 'accumulator = accumulator + rescale[:, None]')]:
            assert old in producer; changed = producer.replace(old, new, 1)
            reject(lambda changed=changed, mode=mode: factory.restore_producer(changed, mode)); mutants += 1
        reject(lambda: factory.restore_reducer(reducer.replace('output_ptr + q_start * output_stride_token', 'output_ptr + token_idx * output_stride_token'))); mutants += 1
    geometry = []
    for lengths, kv in [([1]*63+[1985], 4096), ([1985]+[1]*63, 4096),
                        ([1]*31+[1985]+[1]*32, 4096), ([1]*31+[2017], 16384),
                        ([2017]+[1]*31, 16384), ([1]*15+[2017]+[1]*16, 16384),
                        ([0,1,0,17,1], 129), ([2,1,3,1], 17), ([1,1], 0), ([4,8], 15)]:
        cu = reference.cumulative(lengths); capacity = cu[-1] + 3
        rows = reference.metadata(cu, [kv]*len(lengths), capacity)
        partition = reference.writer_partition(cu, [kv]*len(lengths), capacity, True)
        assert len(partition['decode_tokens']) == sum(n == 1 for n in lengths)
        for row in rows:
            if not row['decode']:
                continue
            for mode in factory.MODES:
                a = reference.addresses(mode=mode, q_start=row['q_start'], sequence=row['sequence'],
                    physical_page=2**30+3, slot=15, kv_head=1, query_head=15, dim=127)
                assert a['partial_sequence_row'] == row['sequence']
                assert a['query_bytes'] == a['output_bytes'] == 2*(row['q_start']*2048+2047)
                assert a['v_bytes'] > 2**43 and a['k_bytes'] > 2**43
        geometry.append(dict(query_lengths=lengths, total_queries=cu[-1], kv=kv, decode_sequences=len(partition['decode_tokens']),
            nonidentity_q_start=any(r['decode'] and r['q_start'] != r['sequence'] for r in rows)))
    # Full per-page coordinate proof, in bytes, independent of tensor dtype metadata.
    offsets = [reference.addresses(mode='k8',q_start=0,sequence=0,physical_page=0,slot=s,kv_head=h,query_head=0,dim=d)
               for s in range(16) for h in range(2) for d in range(128)]
    assert {v['k_bytes'] for v in offsets} == set(range(4096))
    assert {v['k_scale_bytes'] for v in offsets} == set(range(4096,4224,4))
    assert {v['v_bytes'] for v in offsets} == set(range(8192,16384,2))
    binding = dict(mode='k8',format_id=reference.FORMAT,num_seqs=64,total_queries=2048,
        query_strides=(2560,128,1),output_strides=(2048,128,1),key_strides=(16384,256,128,1),
        value_strides=(8192,256,128,1),scale_strides=(4096,2,1))
    assert reference.validate_binding(**binding)['descriptor_contract_passed']
    for change in (dict(format_id='unknown'),dict(format_id=reference.STOCK),dict(key_strides=(8192,256,128,1)),
                   dict(value_strides=(16384,256,128,1)),dict(scale_strides=(1024,2,1)),dict(query_strides=(2048,64,1)),
                   dict(initial_source_storage='packed-kv'),dict(output_storage='q'),dict(num_seqs=65)):
        reject(lambda change=change: reference.validate_binding(**(binding | change))); mutants += 1
    reject(lambda: reference.metadata([0,2,1],[1,1],2)); reject(lambda: reference.metadata([0,2**31],[1],2**31-1))
    reject(lambda: reference.writer_partition([0,1,3],[1,3],3,False))
    rng = random.Random(20261027)
    for length in (0,1,15,16,17):
        q = [bf16(rng.uniform(-1,1)) for _ in range(128)]
        keys = [[bf16(rng.uniform(-1,1)) for _ in range(128)] for _ in range(length)]
        values = [[bf16(rng.uniform(-1,1)) for _ in range(128)] for _ in range(length)]
        goal = reference.attention(q,keys,values)
        assert all(abs(x) <= 128*127**2 for x in goal['integer_scores'])
        zero = reference.attention([0.]*128,keys,values)
        assert zero['original'] == zero['quantized']
    subprocess.run(['git','-C',str(ROOT),'diff','--exit-code',factory.MAIN,'--','vllm_fl','csrc'],check=True)
    assert not {'torch','triton','numpy','vllm'}.intersection(sys.modules)
    return dict(passed=True,cpu_only=True,factory=proof,geometry=geometry,full_page_coordinates=len(offsets),
        negative_source_and_binding_mutants=mutants,scalar_fp64_int64_goal_lengths=[0,1,15,16,17],
        gpu_behavior_simulated=False,gpu_executed=False,service_installed=False,production_changed=False,
        source_guards_are_not_actual_cfg_proof=True)


if __name__ == '__main__':
    print(json.dumps(run(),indent=2))
