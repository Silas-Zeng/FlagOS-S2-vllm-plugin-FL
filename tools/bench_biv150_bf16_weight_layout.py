"""Compare the same FlagGems BF16 mm with two value-identical weight layouts.

No operator switching, quantization, model hooks or installed-package mutation.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import socket
import torch
import flag_gems
from biv150_benchmark_utils import graph_time


def main(args):
    with socket.socket() as guard:assert guard.connect_ex(('127.0.0.1',9031))!=0,'Service running'
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(20261023)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    records=[]
    for n,k in [(2560,2048),(2048,2048),(12288,2048),(2048,6144)]:
        weight=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.02
        packed=weight.t().contiguous()
        assert torch.equal(packed.t(),weight)
        for m in [1,8,32,64,2048]:
            x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
            old=lambda:flag_gems.mm(x,weight.t())
            new=lambda:flag_gems.mm(x,packed)
            reference=old();candidate=new()
            row=dict(m=m,n=n,k=k,original_weight_stride=list(weight.stride()),
                     packed_public_weight_stride=list(packed.t().stride()),logical_weight_bit_exact=True,
                     baseline_operator='flag_gems.mm',candidate_operator='flag_gems.mm')
            try:
                torch.testing.assert_close(candidate,reference,atol=.01,rtol=.01)
                gold=(x.float()@weight.float().t()).to(torch.bfloat16)
                torch.testing.assert_close(candidate,gold,atol=.01,rtol=.01)
                rounds=[]
                for index in range(3):
                    order=[('baseline',old),('candidate',new)]
                    if index%2:order.reverse()
                    times={name:graph_time(fn,repeats=11,unroll=4)[0] for name,fn in order}
                    rounds.append(dict(**times,speedup=times['baseline']/times['candidate']))
                row.update(rounds=rounds,median_speedup=sorted(r['speedup'] for r in rounds)[1],
                           max_abs_diff=(candidate.float()-reference.float()).abs().max().item(),
                           max_abs_diff_fp32=(candidate.float()-gold.float()).abs().max().item(),
                           rmse=(candidate.float()-gold.float()).square().mean().sqrt().item())
            except Exception as error:row['error']=str(error)[:1200]
            records.append(row);print(json.dumps(row),flush=True)
            args.out.write_text(json.dumps(dict(flag_gems_version=importlib.metadata.version('flag_gems'),
                flag_gems_path=flag_gems.__file__,probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                records=records,service_installed=False),indent=2))
            del x,reference,candidate
            torch.cuda.empty_cache()
        del weight,packed
        torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
