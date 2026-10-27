#!/usr/bin/env python3
"""Historical and synthetic changed-source regression only, NOT service proof."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
import biv150_konly_service_q32_hoist as h
ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'docs/evidence/konly-service-q32-hoist-20261027/historical-unhoisted-q32.ttgir'
HISTORICAL_SHA='5e87a8fddac1cf254628a409212f5a6fed2be616179bdefb894e101c40b78264'


def meta(text):
    return dict(schema=h.SCHEMA,enabled=True,profile='q32',format=h.FORMAT,
        target=dict(backend='corex',arch=71,warp_size=64),launch_options=dict(num_warps=16,num_stages=1),
        constants=dict(BLOCK_M=256,BLOCK_Q=32,TILE_SIZE=64,num_seqs=1,num_query_heads=16,num_queries_per_kv=8,KV_QUANT_MODE=0,IS_3D=False),
        query_rows=2048,grid=[65,2],kernel_hash='1'*64,function_name='kernel_cjj_konly_q32',
        source_digest='2'*64,ttgir_sha256=h.sha(text),pins=h.PINS)


def run():
    raw=INPUT.read_text();assert h.sha(raw)==HISTORICAL_SHA
    assert h.propose(None,{})==(None,dict(enabled=False,override_generated=False,actual_I8_hoist_verified=False))
    changed=raw.replace('/workspace/experiments/minicpm-s2/cjj-konly-prefill-q32-compile-v1-20261027/sources/candidate.py',
                        '/synthetic/new-service-root/private_q32.py')
    # Rename SSA uses/defs consistently, preserving lexical scopes and source
    # locations. Renamed numerical results demonstrate no old SSA/hash predicate.
    changed=''.join(re.sub(h.SSA,lambda m:'%service_'+m[0][1:],line) for line in changed.splitlines(True))
    changed=changed.replace('#shared','#service_shared').replace('#mma','#service_mma')
    accepted=[]
    for label,text in [('historical_regression',raw),('synthetic_new_debug_SSA_layout_names',changed)]:
        out,proof=h.propose(text,meta(text));assert out!=text and proof['exact_byte_inverse'] and proof['moved_operations']==1
        assert not proof['actual_override_compiled'] and not proof['actual_I8_hoist_verified']
        accepted.append(dict(label=label,input_sha256=h.sha(text),override_sha256=h.sha(out)))
    for q in (1024,2048,4096,16384):
        m=meta(changed);m.update(query_rows=q,grid=[q//32+1,2]);h.propose(changed,m)
    rejected={}
    facts=h.grammar(raw,'kernel_cjj_konly_q32');lines=raw.splitlines(True)
    faults={
        'alias_Q_memory':lambda s:s.replace(lines[facts['Q_operand_load']-1],lines[facts['Q_operand_load']-1]+'      ttg.local_dealloc '+facts['memory']+' : !ttg.memdesc<256x128xi8, #shared, #smem>\n'),
        'extra_operand_use':lambda s:s.replace(lines[facts['QK']-1],lines[facts['QK']-1]+'      "test.escape"('+facts['operand']+') : () -> ()\n'),
        'extra_KV_loop':lambda s:s.replace('    %acc:3 = scf.for','    %second:3 = scf.for\n    %acc:3 = scf.for',1),
        'BM128_shape':lambda s:s.replace('256x128xi8','128x128xi8'),
        'wrong_operand_index':lambda s:s.replace('opIdx = 0, parent = #mma, kWidth = 4','opIdx = 1, parent = #mma, kWidth = 4'),
        'wrong_QK_C':lambda s:s.replace('dense<0> : tensor<256x64xi32','dense<1> : tensor<256x64xi32'),
        'wrong_PV_dtype':lambda s:s.replace('tensor<256x64xbf16','tensor<256x64xi8'),
        'wrong_target':lambda s:s.replace('cuda:71','cuda:80'),
        'V_instead_of_Q_pointer':lambda s:s.replace('tt.splat %query_ptr :', 'tt.splat %value_cache_ptr :'),
        'wrong_warps':lambda s:s.replace('"ttg.num-warps" = 16','"ttg.num-warps" = 8'),
        'memory_store':lambda s:s.replace(lines[facts['Q_operand_load']-1],lines[facts['Q_operand_load']-1]+'      "test.write"('+facts['memory']+') : () -> ()\n'),
    }
    for name,fn in faults.items():
        text=fn(raw)
        try:h.propose(text,meta(text))
        except h.Unsupported as e:rejected[name]=str(e)
        else:raise AssertionError('accepted IR fault '+name)
    # Moving an allocation into a sibling branch is not textual dominance.
    moved=lines.copy();allocation=moved.pop(facts['q_allocation']-1)
    idx=next(i for i,s in enumerate(moved) if s.startswith('  ^bb1:'))+1;moved.insert(idx,allocation)
    text=''.join(moved)
    try:h.propose(text,meta(text))
    except h.Unsupported as e:rejected['allocation_in_early_return_sibling']=str(e)
    else:raise AssertionError('false allocation dominance')
    for name,mutate in [('nseq2',lambda m:m['constants'].update(num_seqs=2)),
                        ('BQ16',lambda m:m['constants'].update(BLOCK_Q=16)),
                        ('query_short',lambda m:m.update(query_rows=1023)),
                        ('incomplete_grid',lambda m:m.update(grid=[64,2])),
                        ('stale_input_SHA',lambda m:m.update(ttgir_sha256='0'*64)),
                        ('SDK_changed',lambda m:m.update(pins=dict(h.PINS,sdk='0'*64))),
                        ('bool_numseq',lambda m:m['constants'].update(num_seqs=True))]:
        m=deepcopy(meta(raw));mutate(m)
        try:h.propose(raw,m)
        except h.Unsupported as e:rejected[name]=str(e)
        else:raise AssertionError('accepted metadata fault '+name)
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    return dict(CPU_only=True,accepted_regressions=accepted,rejected=rejected,default_off=True,
        metadata_fixture_is_not_runtime_attestation=True,query_scope_CPU_metadata=[1024,2048,4096,16384],
        old_fixture_reused_as_runtime_proof=False,actual_service_TTGIR_seen=False,
        actual_override_compiled=False,actual_I8_hoist_verified=False,native_synchronization_verified=False,
        runtime_binding_qualified=False,precision_gate=False,performance_verified=False,service_verified=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path);a=p.parse_args();result=run()
    if a.out:a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(CPU_contract_passed=True,rejections=len(result['rejected']),actual_I8_hoist_verified=False)))


if __name__=='__main__':main()
