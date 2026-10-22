"""Compare the actual selected low-shared policy to the accepted source snapshot."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from biv150_benchmark_utils import load_module,inputs,full_call,graph_time,check_reference


def main(args):
    base=load_module(args.baseline,"gqa_accepted_snapshot")
    candidate=load_module(args.source,"gqa_final_candidate")
    records=[]
    for b,kv in [(64,4096),(56,5120),(48,5120),(40,5120),(32,16384),(31,17408),
                 (24,17408),(16,17408),(8,32768),(1,100)]:
        c=inputs(b,kv,mixed=True)
        s0=base._pick_segments(b,2,16)
        s1=candidate._pick_segments(b,2,16)
        chosen=candidate._gqa_decode_attention_low_shared if b>=16 else candidate._gqa_decode_attention
        mod=SimpleNamespace(_gqa_decode_attention=chosen,_reduce_partials=candidate._reduce_partials,LOG2E=candidate.LOG2E)
        full_call(base,c,s0);ref=c["out"].clone()
        full_call(mod,c,s1)
        torch.testing.assert_close(c["out"],ref,atol=.005,rtol=.005)
        check_reference(c)
        old=graph_time(lambda:full_call(base,c,s0),repeats=9,unroll=4)[0]
        new=graph_time(lambda:full_call(mod,c,s1),repeats=9,unroll=4)[0]
        row=dict(batch=b,kv=kv,baseline_segments=s0,candidate_segments=s1,baseline_ms=old,candidate_ms=new,
                 speedup=old/new,max_abs_diff=(c["out"].float()-ref.float()).abs().max().item())
        print(json.dumps(row),flush=True);records.append(row)
        del c,ref
        torch.cuda.empty_cache()
    args.out.write_text(json.dumps(records,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    main(p.parse_args())
