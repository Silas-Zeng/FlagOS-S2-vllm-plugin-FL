"""CPU-only Dynamo regression: guarded row decision leaves no FX SymBool op."""
import argparse
import json
from pathlib import Path
import torch
from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

def original(x):
    if tuple(x.shape)!=(64,8):return x+1
    return x+2

def guarded(x):
    if not guard_size_oblivious(x.shape[0]==64):return x+1
    return x+2

def main(args):
    torch.set_num_threads(2);records=[]
    for fn in (original,guarded):
        torch._dynamo.reset();captures=[]
        def backend(gm,inputs):
            captures.append(dict(inputs=[type(v).__name__ for v in inputs],code=gm.code,
                operations=[str(n.target) for n in gm.graph.nodes if n.op=='call_function']))
            return gm.forward
        compiled=torch.compile(fn,backend=backend,fullgraph=True,dynamic=True)
        for rows in (2048,64,32,64):
            x=torch.zeros((rows,8));torch._dynamo.mark_static(x,1)
            assert torch.equal(compiled(x),torch.full_like(x,2 if rows==64 else 1))
        has_eq=any('eq = ' in g['code'] for g in captures)
        assert has_eq==(fn is original)
        records.append(dict(function=fn.__name__,cpu_only=True,has_fx_eq=has_eq,captures=captures))
    result=dict(completed=True,passed=True,cpu_only=True,actual_model=False,records=records)
    args.out.write_text(json.dumps(result,indent=2));print(json.dumps(dict(completed=True,passed=True,cpu_only=True)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
