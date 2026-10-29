#!/usr/bin/env python3
"""CPU-only source/contract checks; optional exact archived multi2 TTGIR regression."""
import argparse
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
sys.dont_write_bytecode = True
import biv150_konly_multi2_hoist_source as c

ACTUAL_TTGIR_SHA = 'bd9c7c5178d0db8a41f6f45c39f3d17a71a375b36645173ee7796e6343a51fe5'
ACTUAL_KERNEL_HASH = '11a3be344b55d63f3a96655db85cd46227f2952c3397bbf3206bcdbb2d836364'


def run(actual_ttgir=None):
    checks = []
    def check(ok, label):
        c.need(ok, label); checks.append(label)
    def reject(call, label):
        try:
            call()
        except ValueError:
            checks.append(label)
        else:
            raise AssertionError('not rejected: ' + label)
    plan = c.plan()
    check(not subprocess.check_output(['git','-C',str(c.ROOT),'diff',c.BASE,'--','vllm_fl','csrc']),
          'production bytes unchanged from main56')
    check(c.candidate_source() == c.original('biv150_konly_encoded_multi2.py'), 'entire GPU source byte identical')
    check(c.sha(c.candidate_source()) == c.FILES['biv150_konly_encoded_multi2.py'], 'complete GPU source hash')
    old = c.module_from_source('cpu_original_q32_proposal', c.original('biv150_konly_encoded_hoist_proposal.py'))
    h = c.proposal(32)
    old_tree = ast.parse(c.original('biv150_konly_encoded_hoist_proposal.py'))
    new_tree = ast.parse(c.proposal_source())
    for name in ('grammar','propose','dominates','regions','blocks','uses'):
        a = next(n for n in old_tree.body if isinstance(n,ast.FunctionDef) and n.name == name)
        b = next(n for n in new_tree.body if isinstance(n,ast.FunctionDef) and n.name == name)
        check(ast.dump(a) == ast.dump(b), 'original proof/move function unchanged: ' + name)
    check(c.driver_source() and c.proposal_source(), 'both exact-count transformations and byte inverses')
    ast.parse(c.driver_source())
    reject(lambda:c.driver().run(SimpleNamespace(root_owned_exclusive=False)),
           'compile entry rejects absent Root ownership before SDK/device work')
    reject(lambda:c.driver().run(SimpleNamespace(root_owned_exclusive=True,commit='0'*40,
           out=Path('/private/tmp/cjj-multi2-cpu-must-not-create-20261029'))),
           'compile entry rejects wrong Git freeze before output/device work')
    check(h.propose(object(), {})[0] is None, 'default disabled does not inspect IR')
    check(h.propose(object(), {'enabled':False})[0] is None, 'explicit disabled does not inspect IR')
    for value in (0,1,None,'1'):
        reject(lambda v=value:h.propose(object(),{'enabled':v}), 'nonbool enable ' + repr(value))
    for q in (1024,1025,1987,1988,1989,2019,2048):
        cfg, grid = c.geometry(q,32)
        check(grid == (q//32+2,2) and cfg == dict(BLOCK_M=256,BLOCK_Q=32,TILE_SIZE=64,IS_3D=False),
              'two-sequence geometry total Q=' + str(q))
    for q,bq in ((True,32),(1023,32),(2049,32),(2048,64),(2048,True),(2048,16)):
        reject(lambda q=q,bq=bq:c.geometry(q,bq), 'out-of-scope geometry ' + repr((q,bq)))
    # This synthetic text is used only for metadata validation, never as a numerical/IR proof.
    text = 'metadata-only CPU fixture\n'
    snapshot = dict(hash=ACTUAL_KERNEL_HASH, function_name=c.NAME, source_digest=c.FUNCTION_SHA,
                    artifacts={'ttgir':c.sha(text)})
    target = dict(backend='corex',arch=71,warp_size=64)
    meta = c.metadata(h,snapshot,2048,32,target)
    h.validate_meta(text,meta); checks.append('private metadata exact contract')
    reject(lambda:old.validate_meta(text,meta), 'production q32 still rejects new schema/profile')
    mutations = [({'schema':old.SCHEMA},'old schema'), ({'profile':'q32'},'old profile'),
                 ({'grid':[65,2]},'single-sequence grid'), ({'grid':[66,True]},'bool grid'),
                 ({'launch_options':{'num_warps':16,'num_stages':2}},'stage2'),
                 ({'launch_options':{'num_warps':8,'num_stages':1}},'eight warps')]
    for patch,label in mutations:
        bad=copy.deepcopy(meta); bad.update(patch)
        reject(lambda bad=bad:h.validate_meta(text,bad),label)
    for key,value in (('num_seqs',1),('num_seqs',True),('BLOCK_Q',64),('BLOCK_M',512),
                      ('TILE_SIZE',128),('IS_3D',True),('KV_QUANT_MODE',1)):
        bad=copy.deepcopy(meta); bad['constants'][key]=value
        reject(lambda bad=bad:h.validate_meta(text,bad),'changed constant '+key+'='+repr(value))
    for key,value in (('function_name','kernel_cjj_konly_q32'),('source_digest','0'*64)):
        bad=copy.deepcopy(snapshot); bad[key]=value
        reject(lambda bad=bad:c.metadata(h,bad,2048,32,target),'actual snapshot '+key+' mismatch')
    class Tensor:
        def __init__(self,*stride): self.strides=stride
        def stride(self,i): return self.strides[i]
    t=dict(q=Tensor(2048,128,1),out=Tensor(2048,128,1),k=Tensor(16384,256,128,1),
           v=Tensor(8192,256,128,1),table=Tensor(8192,1),kscale=object(),lengths=object(),
           cu_q=object(),k_descale=object(),v_descale=object())
    cfg,_=c.geometry(2048,32)
    kw=c.kwargs(dict(profile='multi2',sequences=2,segments=1,config=cfg),t,True)
    check(kw['num_seqs']==2 and kw['query_start_len_ptr'] is t['cu_q'] and kw['seq_lens_ptr'] is t['lengths'],
          'original ABI mapper two-sequence metadata identities')
    check(kw['key_code_scale_ptr'] is t['kscale'] and kw['BLOCK_Q']==32 and kw['BLOCK_M']==256
          and kw['TILE_SIZE']==64 and kw['stride_k_cache_0']==16384 and kw['stride_v_cache_0']==8192
          and kw['query_stride_0']==2048 and kw['block_table_stride']==8192,
          'original ABI mapper encoded page/stride/geometry')
    actual = None
    if actual_ttgir:
        text=actual_ttgir.read_text()
        check(c.sha(text)==ACTUAL_TTGIR_SHA,'exact archived B multi2 TTGIR')
        snapshot['artifacts']['ttgir']=c.sha(text)
        meta=c.metadata(h,snapshot,2048,32,target)
        moved,proof=h.propose(text,meta)
        check(proof['exact_byte_inverse'] and proof['moved_operations']==1,'actual multi2 one-op exact IR inverse')
        check(proof['all_other_source_math_masks_stores_barriers_unchanged'],'actual remaining math/masks/stores/barriers exact')
        check(proof['after']['q_allocation'] < proof['after']['Q_operand_load'] < proof['after']['KV_loop'],
              'actual initialized Q local-load preheader ordering')
        # Re-hoisting, alias use and wrong pointer origin fail the original grammar, not a looser replacement.
        def try_ir(ir):
            updated=copy.deepcopy(meta); updated['ttgir_sha256']=c.sha(ir)
            return h.propose(ir,updated)
        reject(lambda:try_ir(moved),'already-hoisted actual IR rejected')
        before=proof['before']; lines=text.splitlines(keepends=True)
        lines.insert(before['Q_operand_load'], '        ttg.local_dealloc '+before['memory']+' : !ttg.memdesc<256x128xi8, #shared, #smem>\n')
        reject(lambda:try_ir(''.join(lines)),'extra actual Q-memory use rejected')
        bad=text.replace('= tt.dot '+before['operand']+',','= tt.dot %wrong_q_operand,')
        reject(lambda:try_ir(bad),'actual QK operand origin change rejected')
        actual=dict(input_sha256=ACTUAL_TTGIR_SHA,override_sha256=c.sha(moved),before=proof['before'],after=proof['after'],
                    archived_source='5ca2afaa91bc9133d4170a22410daef4e22f5e00',
                    host_metadata='CPU-selected 2048-row contract; not a claim about the archived request lengths',
                    exact_byte_inverse=True,backend_compilation_performed=False)
    check(not {'torch','triton','vllm','vllm_fl'}.intersection(sys.modules),'no SDK/Torch imports')
    return dict(schema='cjj-private-multi2-operand-hoist-cpu-v1',completed=True,errors=[],
        checks=checks,check_count=len(checks),plan=plan,actual_multi2_structure=actual,
        GPU_executed=False,production_modified=False,numerical_verified=False,Graph_verified=False,
        actual_backend_verified=False,performance_verified=False,service_verified=False,
        source_files={p.name:c.sha(p.read_bytes()) for p in sorted(Path(__file__).parent.glob('*multi2_hoist*.py'))})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actual-ttgir',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    result=run(args.actual_ttgir)
    text=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if args.report:args.report.write_text(text)
    print(text,end='')
