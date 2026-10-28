#!/usr/bin/env python3
"""Narrow stdlib source/geometry checks, not device math or compiler proof."""
import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
import biv150_konly_q64_compile_source as c


def run(historical=None):
    result = c.plan()
    tree = ast.parse(c.candidate_source())
    fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == c.NAME)
    assignments = {n.targets[0].id:n.value for n in ast.walk(fn) if isinstance(n,ast.Assign)
        and len(n.targets) == 1 and isinstance(n.targets[0],ast.Name)}
    pos = compile(ast.Expression(assignments['query_pos']),'<actual-query-position>','eval')
    head = compile(ast.Expression(assignments['query_offset_1']),'<actual-query-head>','eval')
    scope = sorted(set([1024+i for i in range(64)] + [1985,2017,2048]))
    cases=[]; checked=0
    for q in scope:
        for bq in (32,64):
            config, grid = c.geometry(q,bq)
            seen=set(); masked=0; early=0
            for block in range(grid[0]):
                if block*bq >= q: early+=1; continue
                for kv in range(2):
                    for m in range(config['BLOCK_M']):
                        env=dict(q_block_local_idx=block,BLOCK_Q=bq,offs_m=m,num_queries_per_kv=8,kv_head_idx=kv)
                        row=eval(pos,{'__builtins__':{}},env); h=eval(head,{'__builtins__':{}},env)
                        if row >= q: masked+=1; continue
                        # Complete 128-wide contiguous read and store interval
                        # at stride(2048,128,1); masks/expressions AST unchanged.
                        lo=row*2048+h*128; hi=lo+127
                        c.need(0 <= lo <= hi < q*2048, 'query/output bounds')
                        c.need((row,h) not in seen,'duplicate query/head output')
                        seen.add((row,h)); checked+=1
            c.need(len(seen) == q*16 and seen == {(r,h) for r in range(q) for h in range(16)},'missing query/head')
            cases.append(dict(q=q,bq=bq,grid=list(grid),early_return_blocks=early,
                masked_query_head_rows=masked,query_head_rows=len(seen)))
    # Every byte of the exact unchanged physical page roles is in range and
    # nonoverlapping, including both end pages of the actual compile allocation.
    addresses=0
    for p in range(773):
        base=p*16384
        for slot in range(16):
            for head_index in range(2):
                k=base+slot*256+head_index*128
                scale=(p*4096+1024+slot*2+head_index)*4
                value=base+8192+(slot*256+head_index*128)*2
                c.need(base <= k <= k+127 < base+4096, 'K page range')
                c.need(base+4096 <= scale <= scale+3 < base+4224, 'scale page range')
                c.need(base+8192 <= value <= value+255 < base+16384 <= 773*16384, 'V page range')
                addresses+=128
    rejects=[]
    def reject(name,call):
        try: call()
        except ValueError: rejects.append(name); return
        raise AssertionError('accepted invalid '+name)
    for q,bq in [(1023,64),(2049,64),(True,64),(2048,128),(2048,True),(2048,32.0)]:
        reject('geometry '+repr((q,bq)),lambda q=q,bq=bq:c.geometry(q,bq))
    h=c.proposal(64)
    text='\n'  # Binding-only test; intentionally not an actual compiled kernel.
    snapshot=dict(hash='1'*64,function_name=c.NAME,source_digest='2'*64,artifacts={'ttgir':c.sha(text)})
    meta=c.metadata(h,snapshot,2048,64,dict(backend='corex',arch=71,warp_size=64))
    h.validate_meta(text,meta)
    c.need(h.propose(None,{}) == (None,dict(enabled=False,override_generated=False,actual_I8_hoist_verified=False)), 'default off')
    changes=[('grid',lambda m:m.update(grid=[65,2])),('BM256',lambda m:m['constants'].update(BLOCK_M=256)),
        ('BQ32',lambda m:m['constants'].update(BLOCK_Q=32)),('K128',lambda m:m['constants'].update(TILE_SIZE=128)),
        ('two seq',lambda m:m['constants'].update(num_seqs=2)),('w8',lambda m:m['launch_options'].update(num_warps=8)),
        ('s2',lambda m:m['launch_options'].update(num_stages=2)),('old profile',lambda m:m.update(profile='q32')),
        ('old schema',lambda m:m.update(schema=c.proposal(32).SCHEMA)),('stale TTGIR',lambda m:m.update(ttgir_sha256='0'*64)),
        ('SDK',lambda m:m['pins'].update(sdk='0'*64)),('bool BM',lambda m:m['constants'].update(BLOCK_M=True))]
    for name,change in changes:
        bad=deepcopy(meta);change(bad)
        reject(name,lambda bad=bad:h.validate_meta(text,bad))
    reject('missing actual IR',lambda:h.propose(text,meta))
    fixture=None
    if historical:
        raw=historical.read_text()
        c.need(c.sha(raw)=='5e87a8fddac1cf254628a409212f5a6fed2be616179bdefb894e101c40b78264','historical parser fixture pin')
        # Synthetic dimension-expanded OLD Q32 IR tests parser mechanics only.
        synthetic=raw.replace('kernel_cjj_konly_q32(',c.NAME+'(').replace('256x','512x')
        mock=dict(snapshot,artifacts={'ttgir':c.sha(synthetic)})
        m=c.metadata(h,mock,2048,64,meta['target'])
        out,proof=h.propose(synthetic,m)
        c.need(proof['exact_byte_inverse'] and proof['moved_operations']==1,'single hoist inverse')
        facts=h.grammar(synthetic,c.NAME);lines=synthetic.splitlines(True)
        for name,broken in [
            ('memory escape',synthetic.replace(lines[facts['Q_operand_load']-1],lines[facts['Q_operand_load']-1]+'      "test.escape"('+facts['memory']+') : () -> ()\n')),
            ('wrong QK type',synthetic.replace('tensor<512x64xi32','tensor<512x64xf32')),
            ('wrong PV type',synthetic.replace('tensor<512x64xbf16','tensor<512x64xi8')),
            ('old tile shape',synthetic.replace('512x','256x'))]:
            mm=deepcopy(m);mm['ttgir_sha256']=c.sha(broken)
            reject(name,lambda broken=broken,mm=mm:h.propose(broken,mm))
        fixture=dict(historical_sha256=c.sha(raw),synthetic_shape_only=True,actual_Q64_IR=False,
            one_move_inverse=True,synthetic_input_sha256=c.sha(synthetic),synthetic_override_sha256=c.sha(out))
    # Source ABI mapper is the old Q32 mapper, with only its module docstring
    # changed. No new numerical reference is introduced by this screen.
    args_source=(c.ROOT/'tools/biv150_konly_q64_compile_arguments.py').read_text()
    body=next(n for n in ast.parse(args_source).body if isinstance(n,ast.FunctionDef) and n.name=='kwargs')
    args_digest=c.sha(ast.get_source_segment(args_source,body))
    c.need(args_digest=='80353ca42956db04c2c1c5aae186b7286968f34e334bb34dfe2304da4a056b2e','original Q32 ABI mapper changed')
    c.need(not {'torch','triton','numpy','vllm'}.intersection(sys.modules),'CPU checker imported device framework')
    c.need(not subprocess.check_output(['git','-C',str(c.ROOT),'diff',c.BASE,'--','vllm_fl','csrc']),'production changed')
    return dict(result,CPU_checks_passed=True,geometry_cases=cases,query_head_rows_checked=checked,
        K_and_V_dimension_positions_checked=addresses,metadata_and_IR_rejections=rejects,
        historical_parser_regression=fixture,ABI_mapper_body_sha256=args_digest,
        address_checks_are_CPU_geometry_not_device_memory_safety=True,
        actual_Q64_compiled=False,actual_native_resources_or_hoist_verified=False)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--historical-ttgir',type=Path);p.add_argument('--out',type=Path)
    a=p.parse_args();result=run(a.historical_ttgir)
    text=json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if a.out:a.out.write_text(text)
    print(text)
