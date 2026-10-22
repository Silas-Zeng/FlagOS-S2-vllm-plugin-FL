"""Check whether delayed V addressing/loading reduces GQA register/shared pressure."""
import argparse
import json
from pathlib import Path
import re
import socket

import torch
from bench_biv150_fastpaths import load_module,inputs,full_call,main_call,graph_time,check_reference


def variants(text):
    a=text.index("@triton.jit\ndef _gqa_decode_attention")
    b=text.index("@triton.jit\ndef _reduce_partials",a)
    body=text[a:b]
    vblocks=re.findall(r"        v_offsets = \(\n.*?        \)\n",body,re.S)
    assert len(vblocks)==2
    vlines=[line+"\n" for line in body.splitlines() if "V = tl.load(value_cache_ptr + v_offsets" in line]
    assert len(vlines)==2
    versions={"baseline":body}
    for name,addresses in [("late_load",False),("late_address_load",True)]:
        result=body
        anchor="        accumulator = accumulator * rescale[:, None] + tl.dot(\n"
        pos=0
        for block,line in zip(vblocks,vlines):
            old_at=result.index(line,pos)
            result=result[:old_at]+result[old_at+len(line):]
            moved=line
            if addresses:
                old_at=result.index(block,pos)
                result=result[:old_at]+result[old_at+len(block):]
                moved=block+moved
            target=result.index(anchor,pos)
            result=result[:target]+moved+result[target:]
            pos=target+len(moved)+len(anchor)
        versions[name]=result
    return {k:text[:a]+s+text[b:] for k,s in versions.items()}


def main(args):
    with socket.socket() as s:
        assert s.connect_ex(("127.0.0.1",9031))!=0,"A service is running"
    args.out.mkdir(parents=True,exist_ok=True)
    mods={}
    for name,source in variants(args.source.read_text()).items():
        f=args.out/(name+".py");f.write_text(source)
        mods[name]=load_module(f,"gqa_liveness_"+name)
    records=[]
    for b,kv in [(64,4096),(64,5120),(32,16384),(16,17408),(8,32768)]:
        seg=mods["baseline"]._pick_segments(b,2,16)
        for mixed in [False,True]:
            c=inputs(b,kv,mixed=mixed)
            full_call(mods["baseline"],c,seg)
            ref=c["out"].clone()
            base=graph_time(lambda:full_call(mods["baseline"],c,seg))[0]
            for name,mod in mods.items():
                kernel=main_call(mod,c,seg)
                full_call(mod,c,seg)
                torch.testing.assert_close(c["out"],ref,atol=0,rtol=0)
                if mixed:check_reference(c)
                ms=graph_time(lambda:full_call(mod,c,seg))[0]
                row=dict(batch=b,kv=kv,mixed=mixed,name=name,baseline_ms=base,median_ms=ms,
                         speedup=base/ms,bit_exact=True,n_regs=kernel.n_regs,
                         spills=kernel.n_spills,shared=kernel.metadata.shared)
                records.append(row);print(json.dumps(row),flush=True)
        del c,ref
        torch.cuda.empty_cache()
    (args.out/"results.json").write_text(json.dumps(records,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    main(p.parse_args())
