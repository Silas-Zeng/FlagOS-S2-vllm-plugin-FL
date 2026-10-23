"""Test a pipeline-capable for loop against the exact shipping while loop."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import torch
from biv150_benchmark_utils import load_module,inputs,full_call,graph_time,check_reference


class LaunchOptions:
    def __init__(self,kernel,stages):self.kernel=kernel;self.stages=stages;self.last=None
    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            kwargs['num_stages']=self.stages
            self.last=self.kernel[grid](*args,**kwargs)
            return self.last
        return launch
    def metadata(self):
        return dict(n_regs=getattr(self.last,'n_regs',None),n_spills=getattr(self.last,'n_spills',None),shared=getattr(getattr(self.last,'metadata',None),'shared',None))


def main(args):
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',9031))!=0,'Service is running'
    text=args.source.read_text()
    old='    position = lower\n    while position < full_upper:'
    advance='        position += PAGE_SIZE\n'
    assert text.count(old)==text.count(advance)==1
    args.out.parent.mkdir(parents=True,exist_ok=True)
    generated=args.out.parent/'decode-pipeline-generated';generated.mkdir(exist_ok=True)
    variants={}
    for kind in ['while','for']:
        changed=text if kind=='while' else text.replace(old,'    for position in range(lower, full_upper, PAGE_SIZE):').replace(advance,'')
        for stages in [1,2,3]:
            name=f'{kind}_s{stages}';path=generated/(name+'.py');path.write_text(changed)
            mod=load_module(path,'biv150_decode_pipeline_'+name)
            launch=LaunchOptions(mod._gqa_decode_attention,stages);mod._gqa_decode_attention=launch
            variants[name]=(mod,launch)
    records=[]
    for batch,kv,mixed in [(64,4096,False),(32,16384,False),(31,16896,False),(40,16896,True),(8,1024,True),(1,100,False)]:
        c=inputs(batch,kv,mixed=mixed)
        baseline=variants['while_s1'][0];segments=baseline._pick_segments(batch,2,16)
        full_call(baseline,c,segments);reference=c['out'].clone()
        base_ms=graph_time(lambda:full_call(baseline,c,segments),repeats=9,unroll=4)[0]
        for name,(module,launch) in variants.items():
            row=dict(batch=batch,kv=kv,mixed=mixed,segments=segments,variant=name,baseline_ms=base_ms)
            try:
                full_call(module,c,segments)
                torch.testing.assert_close(c['out'],reference,atol=.005,rtol=.005)
                check_reference(c)
                row.update(bit_exact=torch.equal(c['out'],reference),max_abs_diff=(c['out'].float()-reference.float()).abs().max().item(),candidate_ms=graph_time(lambda:full_call(module,c,segments),repeats=9,unroll=4)[0],**launch.metadata())
                row['speedup']=base_ms/row['candidate_ms']
            except Exception as error:row['error']=str(error)[:500]
            records.append(row);print(json.dumps(row),flush=True)
            args.out.write_text(json.dumps(dict(source_sha256=hashlib.sha256(text.encode()).hexdigest(),records=records),indent=2))
        del c,reference
        torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
