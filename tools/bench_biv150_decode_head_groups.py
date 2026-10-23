"""Isolated BF16 GQA query-head subgroup probe; no serving installation."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
from types import SimpleNamespace
import torch
from biv150_benchmark_utils import load_module,inputs,full_call,graph_time,check_reference


class SplitLaunch:
    def __init__(self,kernel,groups,rows,warps):
        self.kernel,self.groups,self.rows,self.warps=kernel,groups,rows,warps
        self.last=None
    def __getitem__(self,grid):
        def invoke(*args,**kwargs):
            kwargs.update(ROWS=self.rows,num_warps=self.warps)
            self.last=self.kernel[(grid[0],grid[1]*self.groups,grid[2])](*args,**kwargs)
            return self.last
        return invoke


def main(args):
    with socket.socket() as guard:assert guard.connect_ex(('127.0.0.1',9031))!=0,'Service running'
    args.out.parent.mkdir(parents=True,exist_ok=True)
    original=args.source.read_text()
    baseline=load_module(args.source,'gqa_head_group_baseline')
    variants=[]
    for groups in [2,4]:
        text=original.replace('def _gqa_decode_attention(','def _gqa_decode_head_groups(',1)
        text=text.replace('    kv_head_idx = tl.program_id(1)',
            f'    kv_head_idx = tl.program_id(1) // {groups}\n    head_group = tl.program_id(1) % {groups}',1)
        text=text.replace('    row_valid = offs_row < queries_per_kv',f'    row_valid = offs_row < {8//groups}',1)
        text=text.replace('    head_idx = kv_head_idx * queries_per_kv + offs_row',
            f'    head_idx = kv_head_idx * queries_per_kv + head_group * {8//groups} + offs_row',1)
        path=args.out.parent/f'head-groups-{groups}.py';path.write_text(text)
        raw=load_module(path,'gqa_head_groups_'+str(groups))
        for rows in sorted({8//groups,8}):
            for warps in [1,2]:
                launch=SplitLaunch(raw._gqa_decode_head_groups,groups,rows,warps)
                variants.append((groups,rows,warps,SimpleNamespace(
                    _gqa_decode_attention=launch,_reduce_partials=raw._reduce_partials,LOG2E=raw.LOG2E),launch))
    records=[]
    for batch,kv,mixed in [(64,4096,False),(32,16384,False),(8,100,True)]:
        c=inputs(batch,kv,mixed=mixed)
        base_segments=baseline._pick_segments(batch,2,16)
        full_call(baseline,c,base_segments);reference=c['out'].clone()
        base_ms=graph_time(lambda:full_call(baseline,c,base_segments),repeats=11,unroll=4)[0]
        for groups,rows,warps,module,launch in variants:
            for segments in sorted({max(1,base_segments//groups),base_segments,min(16,base_segments*2)}):
                row=dict(batch=batch,kv=kv,mixed=mixed,groups=groups,rows=rows,warps=warps,
                         segments=segments,baseline_segments=base_segments,baseline_ms=base_ms)
                try:
                    full_call(module,c,segments)
                    torch.testing.assert_close(c['out'],reference,atol=.005,rtol=.005)
                    check_reference(c)
                    ms=graph_time(lambda:full_call(module,c,segments),repeats=11,unroll=4)[0]
                    row.update(candidate_ms=ms,speedup=base_ms/ms,
                        max_abs_diff=(c['out'].float()-reference.float()).abs().max().item(),
                        n_regs=getattr(launch.last,'n_regs',None),n_spills=getattr(launch.last,'n_spills',None),
                        shared=getattr(getattr(launch.last,'metadata',None),'shared',None))
                except Exception as error:row['error']=str(error)[:800]
                records.append(row);print(json.dumps(row),flush=True)
                args.out.write_text(json.dumps(dict(source_sha256=hashlib.sha256(original.encode()).hexdigest(),
                    records=records,service_installed=False),indent=2))
        del c,reference
        torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
