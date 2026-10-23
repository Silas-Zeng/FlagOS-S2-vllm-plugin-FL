"""Alternate unchanged BF16 decode and explicit .cv streaming loads."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import socket
import statistics
import torch
from biv150_benchmark_utils import load_module,inputs,full_call,main_call,graph_time,check_reference
from bench_biv150_decode_cache_policy import variant


def main(args):
    with socket.socket() as guard:assert guard.connect_ex(('127.0.0.1',9031))!=0,'Service running'
    args.out.parent.mkdir(parents=True,exist_ok=True)
    original=args.source.read_text();text=variant(original,'.cv','')
    file=args.out.parent/'stream-kv-candidate.py';file.write_text(text)
    base=load_module(args.source,'stream_kv_alternate_baseline')
    candidate=load_module(file,'stream_kv_alternate_candidate')
    records=[]
    for b,kv,mixed in [(64,4096,False),(64,5120,False),(32,16384,False),
                       (31,16896,False),(16,17408,False),(40,16896,True),(8,1024,True),(1,100,False)]:
        c=inputs(b,kv,mixed=mixed);segments=base._pick_segments(b,2,16)
        full_call(base,c,segments);reference=c['out'].clone()
        kernel=main_call(candidate,c,segments);full_call(candidate,c,segments)
        torch.testing.assert_close(c['out'],reference,atol=0,rtol=0);check_reference(c)
        values=[]
        for index in range(5):
            order=[('baseline',base),('candidate',candidate)]
            if index%2:order.reverse()
            times={}
            for name,module in order:
                times[name]=graph_time(lambda:full_call(module,c,segments),repeats=15,unroll=4)[0]
            values.append(dict(round=index,**times,speedup=times['baseline']/times['candidate']))
        row=dict(batch=b,kv=kv,mixed=mixed,segments=segments,bit_exact=True,rounds=values,
                 median_speedup=statistics.median(v['speedup'] for v in values),
                 min_speedup=min(v['speedup'] for v in values),max_speedup=max(v['speedup'] for v in values),
                 n_regs=kernel.n_regs,n_spills=kernel.n_spills,shared=kernel.metadata.shared)
        if b==64 and kv==4096:
            for name,module in [('baseline',base),('candidate',candidate)]:
                built=main_call(module,c,segments)
                asm=built.asm['asm'];llir=built.asm['llir']
                (args.out.parent/(name+'.asm')).write_text(asm)
                (args.out.parent/(name+'.llir')).write_text(llir)
                flags=Counter(re.findall(r'@llvm\.bi\.load\.kop\.v2bf16\(.*i32 (\d+), i1 (\w+)\)',llir))
                row[name+'_bf16_load_flags']={str(k):v for k,v in flags.items()}
        records.append(row);print(json.dumps(row),flush=True)
        args.out.write_text(json.dumps(dict(source_sha256=hashlib.sha256(original.encode()).hexdigest(),
            candidate_sha256=hashlib.sha256(text.encode()).hexdigest(),records=records),indent=2))
        del c,reference
        torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
