"""Compare full deferred BF16 decode chain, including score scratch traffic."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
from types import SimpleNamespace
import torch
from biv150_benchmark_utils import load_module,inputs,full_call,graph_time,check_reference


def main(args):
    with socket.socket() as guard:assert guard.connect_ex(('127.0.0.1',9031))!=0,'Service running'
    base=load_module(args.baseline,'deferred_bf16_baseline')
    deferred=load_module(args.source,'deferred_bf16_candidate')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    records=[]
    for b,kv,mixed in [(8,127,True),(64,4096,False),(32,16384,False),(31,16896,False)]:
        c=inputs(b,kv,mixed=mixed)
        base_segments=base._pick_segments(b,2,16)
        full_call(base,c,base_segments);reference=c['out'].clone()
        base_ms=graph_time(lambda:full_call(base,c,base_segments),repeats=11,unroll=4)[0]
        scores=torch.empty(b,16,kv,device='cuda',dtype=torch.float32)
        local_max=torch.empty(b,16,16,device='cuda',dtype=torch.float32)
        for tile,warps in [(32,1),(64,1),(64,2),(128,2),(128,4)]:
            for stages in [1,2]:
                for segments in sorted({base_segments,min(16,base_segments*2)}):
                    launch=deferred.DeferredLaunch(scores,local_max,tile,warps,stages)
                    mod=SimpleNamespace(_gqa_decode_attention=launch,
                        _reduce_partials=deferred.DeferredReduce(base._reduce_partials,tile),LOG2E=base.LOG2E)
                    row=dict(batch=b,kv=kv,mixed=mixed,tile=tile,warps=warps,stages=stages,
                             segments=segments,baseline_segments=base_segments,baseline_ms=base_ms,
                             scratch_bytes=scores.nbytes+local_max.nbytes)
                    try:
                        full_call(mod,c,segments)
                        torch.testing.assert_close(c['out'],reference,atol=.005,rtol=.005)
                        check_reference(c)
                        ms=graph_time(lambda:full_call(mod,c,segments),repeats=11,unroll=4)[0]
                        row.update(candidate_ms=ms,speedup=base_ms/ms,
                            max_abs_diff=(c['out'].float()-reference.float()).abs().max().item(),
                            qk_regs=launch.last_qk.n_regs,pv_regs=launch.last_pv.n_regs,
                            qk_shared=launch.last_qk.metadata.shared,pv_shared=launch.last_pv.metadata.shared)
                    except Exception as error:row['error']=str(error)[:1200]
                    records.append(row);print(json.dumps(row),flush=True)
                    args.out.write_text(json.dumps(dict(source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                        baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),records=records),indent=2))
        del c,reference,scores,local_max
        torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
