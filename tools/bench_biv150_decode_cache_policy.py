"""Probe streaming KV load policies; compare bit-exact output and compiled IR."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import torch
from biv150_benchmark_utils import load_module,inputs,full_call,main_call,graph_time,check_reference


def variant(text,cache,evict):
    for name,pointer,offsets,mask in [('K','key_cache_ptr','k_offsets','column_valid[None, :]'),
                                      ('V','value_cache_ptr','v_offsets','column_valid[:, None]')]:
        full=f'        {name} = tl.load({pointer} + {offsets})'
        tail=f'        {name} = tl.load({pointer} + {offsets}, mask={mask}, other=0.0)'
        assert text.count(full)==text.count(tail)==1
        option=f'cache_modifier={cache!r}, eviction_policy={evict!r}'
        text=text.replace(full,full[:-1]+', '+option+')')
        text=text.replace(tail,tail[:-1]+', '+option+')')
    return text


def main(args):
    with socket.socket() as guard:assert guard.connect_ex(('127.0.0.1',9031))!=0,'Service running'
    args.out.parent.mkdir(parents=True,exist_ok=True)
    original=args.source.read_text()
    configs=[('baseline','','')]+[(cache+'_'+(evict or 'default'),cache,evict)
        for cache in ['.ca','.cg','.cv'] for evict in ['', 'evict_first','evict_last']]
    mods={}
    for name,cache,evict in configs:
        text=original if name=='baseline' else variant(original,cache,evict)
        path=args.out.parent/(name.replace('.','')+'.py');path.write_text(text)
        mods[name]=load_module(path,'decode_cache_policy_'+name.replace('.',''))
    records=[]
    for b,kv,mixed in [(64,4096,False),(32,16384,False),(31,16896,False),(8,1024,True)]:
        c=inputs(b,kv,mixed=mixed);base=mods['baseline'];segments=base._pick_segments(b,2,16)
        full_call(base,c,segments);reference=c['out'].clone()
        before=graph_time(lambda:full_call(base,c,segments),repeats=15,unroll=4)[0]
        for name,cache,evict in configs:
            row=dict(batch=b,kv=kv,mixed=mixed,segments=segments,name=name,
                     cache_modifier=cache,eviction_policy=evict,baseline_ms=before)
            module=mods[name]
            try:
                kernel=main_call(module,c,segments)
                full_call(module,c,segments)
                torch.testing.assert_close(c['out'],reference,atol=0,rtol=0)
                check_reference(c)
                ms=graph_time(lambda:full_call(module,c,segments),repeats=15,unroll=4)[0]
                row.update(candidate_ms=ms,speedup=before/ms,bit_exact=True,
                    n_regs=kernel.n_regs,n_spills=kernel.n_spills,shared=kernel.metadata.shared,
                    llir_sha256=hashlib.sha256(kernel.asm['llir'].encode()).hexdigest(),
                    asm_sha256=hashlib.sha256(kernel.asm['asm'].encode()).hexdigest())
            except Exception as error:row['error']=str(error)[:800]
            records.append(row);print(json.dumps(row),flush=True)
            args.out.write_text(json.dumps(dict(source_sha256=hashlib.sha256(original.encode()).hexdigest(),
                records=records,service_installed=False),indent=2))
        after=graph_time(lambda:full_call(base,c,segments),repeats=15,unroll=4)[0]
        print(json.dumps(dict(batch=b,kv=kv,baseline_before_ms=before,baseline_after_ms=after)),flush=True)
        del c,reference
        torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
