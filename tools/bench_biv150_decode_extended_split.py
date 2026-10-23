"""Isolated split-K probe beyond builder16, using operator-owned scratch only."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import torch
from biv150_benchmark_utils import load_module, inputs, full_call, graph_time
from bench_biv150_int8_kv_decode import dense_reference


def main(args):
    with socket.socket() as guard:assert guard.connect_ex(('127.0.0.1',9031))!=0,'Service running'
    torch.backends.cuda.matmul.allow_tf32=False
    source=load_module(args.source,'extended_split_reference')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    original=args.source.read_text()
    old='weights = tl.exp(tl.where(seg_valid, segm_max, 0.0) - overall_max)'
    new='weights = tl.exp(tl.where(seg_valid, segm_max - overall_max, float("-inf")))'
    assert original.count(old)==1
    corrected=original.replace(old,new)
    fixed_path=args.out.parent/'fixed-reduction.py';fixed_path.write_text(corrected)
    fixed=load_module(fixed_path,'extended_split_fixed_reduction')
    records=[]
    cases=[(b,kv,False) for b in [1,2,4,8] for kv in [1024,4096,16384,32768]]
    cases += [(1,100,False),(4,127,True),(8,1024,True)]
    for batch,kv,mixed in cases:
        c=inputs(batch,kv,mixed=mixed)
        baseline_segments=source._pick_segments(batch,2,16)
        full_call(source,c,baseline_segments);reference=c['out'].clone()
        before=graph_time(lambda:full_call(source,c,baseline_segments),repeats=15,unroll=4)[0]
        cc=dict(c,scratch=torch.empty(batch,16,128,128,device='cuda',dtype=torch.float32),
                maxima=torch.empty(batch,16,128,device='cuda',dtype=torch.float32),
                sums=torch.empty(batch,16,128,device='cuda',dtype=torch.float32),out=torch.empty_like(c['out']))
        fixed_base_ms=graph_time(lambda:full_call(fixed,c,baseline_segments),repeats=15,unroll=4)[0]
        for width in [16,32,64,128]:
            row=dict(batch=batch,kv=kv,mixed=mixed,baseline_segments=baseline_segments,
                     segments=width,baseline_ms=before,fixed_reduce_baseline_ms=fixed_base_ms,
                     scratch_bytes=cc['scratch'].nbytes+cc['maxima'].nbytes+cc['sums'].nbytes)
            try:
                full_call(fixed,cc,width);result=cc['out'].clone()
                torch.testing.assert_close(result,reference,atol=.005,rtol=.005)
                ms=graph_time(lambda:full_call(fixed,cc,width),repeats=15,unroll=4)[0]
                row.update(candidate_ms=ms,speedup=before/ms,speedup_vs_fixed_reduce=fixed_base_ms/ms,bit_exact=torch.equal(result,reference),
                    max_abs_diff=(result.float()-reference.float()).abs().max().item(),dense_checks=[])
                for i in sorted({0,min(1,batch-1),batch-1}):
                    gold=dense_reference(c,c['k'],c['v'],i).to(torch.bfloat16)
                    torch.testing.assert_close(result[i],gold,atol=.005,rtol=.005)
                    row['dense_checks'].append(dict(row=i,passed=True))
            except Exception as error:row['error']=str(error)[:1200]
            records.append(row);print(json.dumps(row),flush=True)
            args.out.write_text(json.dumps(dict(source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                fixed_source_sha256=hashlib.sha256(corrected.encode()).hexdigest(),records=records,service_installed=False),indent=2))
        after=graph_time(lambda:full_call(source,c,baseline_segments),repeats=15,unroll=4)[0]
        print(json.dumps(dict(batch=batch,kv=kv,baseline_before_ms=before,baseline_after_ms=after)),flush=True)
        del c,cc,reference
        torch.cuda.empty_cache()


    # Independently inspect inactive-segment weights when true logits are very
    # negative. The reference softmax subtracts its maximum and stays finite.
    c=inputs(4,127);c['q'].fill_(-2);c['k'].fill_(4)
    full_call(source,c,16)
    old_nonfinite=int((~torch.isfinite(c['out'])).sum().item())
    stress=[]
    cc=dict(c,scratch=torch.empty(4,16,128,128,device='cuda',dtype=torch.float32),
            maxima=torch.empty(4,16,128,device='cuda',dtype=torch.float32),
            sums=torch.empty(4,16,128,device='cuda',dtype=torch.float32),out=torch.empty_like(c['out']))
    for width in [16,32,64,128]:
        full_call(fixed,cc,width)
        row=dict(segments=width,finite=bool(torch.isfinite(cc['out']).all().item()),dense_checks=[])
        try:
            for i in range(4):
                gold=dense_reference(c,c['k'],c['v'],i).to(torch.bfloat16)
                torch.testing.assert_close(cc['out'][i],gold,atol=.005,rtol=.005)
                row['dense_checks'].append(dict(row=i,passed=True))
        except Exception as error:row['error']=str(error)[:800]
        stress.append(row)
    result=json.loads(args.out.read_text())
    result['negative_logits_stress']=dict(query_value=-2,key_value=4,baseline_nonfinite=old_nonfinite,candidates=stress)
    args.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result['negative_logits_stress']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
