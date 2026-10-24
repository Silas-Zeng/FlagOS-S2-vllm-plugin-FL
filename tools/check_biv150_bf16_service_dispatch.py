"""Actual registered MM/MM-out route, caller output, fullgraph and replay checks."""
import argparse,inspect,json,socket
from pathlib import Path
import torch
from bench_biv150_bf16_real_oracle import weights
from audit_biv150_bf16_saved_ulp_cpu import describe


def main(a):
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',9031))!=0
    import flag_gems
    from vllm_fl.utils import use_flaggems,get_flag_gems_whitelist_blacklist
    assert use_flaggems() and not any(get_flag_gems_whitelist_blacklist())
    flag_gems.enable(record=False,once=True)
    from vllm_fl.patches.biv150_bf16_small_splitk import install_worker_launcher,run_into,_supported
    namespace=inspect.unwrap(flag_gems.mm_out).__globals__;before=namespace['_launch_mm']
    meta=install_worker_launcher();launch=namespace['_launch_mm']
    cell=dict(zip(launch.__code__.co_freevars,launch.__closure__))['original'];assert cell.cell_contents is before
    result=dict(installation=meta,dispatch_table=torch._C._dispatch_dump_table('aten::mm.out'),records=[])
    torch.set_num_threads(2);torch.manual_seed(20261025)
    def compare(x,y):torch.testing.assert_close(x.cpu(),y.cpu(),atol=0,rtol=0)
    for layer,name,host,ids in weights(a.model,[4]):
        w=host.to('cuda');k=w.shape[1];n=w.shape[0];wt=w.t()
        assert wt.untyped_storage().data_ptr()==w.untyped_storage().data_ptr()
        for m in (1,8,16):
            x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
            out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
            expected=torch.empty_like(out);scratch=torch.empty((4,m,n),device='cuda',dtype=torch.float32)
            returned=torch.mm(x,wt,out=out);assert returned.data_ptr()==out.data_ptr()
            run_into(x,w,expected,scratch,(16,128,64,4,4));compare(out,expected)
            def call(t,v):return torch.mm(t,v.t())
            compiled=torch.compile(call,fullgraph=True)
            compare(compiled(x,w),expected)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):torch.mm(x,wt,out=out)
            for factor in (0.,.5,4.):
                x.normal_().mul_(factor);graph.replay();torch.cuda.synchronize()
                run_into(x,w,expected,scratch,(16,128,64,4,4));compare(out,expected);compare(compiled(x,w),expected)
            gold=(x.cpu().double()@host.double().t()).to(torch.bfloat16)
            result['records'].append(dict(geometry=name,m=m,weight_identity=ids,caller_output_same=True,
                original_weight_storage=True,hook_matches_tested_kernel_bit_exact=True,
                fullgraph_and_fake=True,changed_input_graph=True,cpu_diagnostic=describe(out.cpu(),gold)))
        # Verify actual fallback entry identity; no numeric pass is inferred for
        # an original backend that may itself be nondeterministic.
        for m in (32,64):
            x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16);out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
            assert not _supported(x,wt,out,m,n,k)
            seen=[]
            def spy(*args):seen.append(args);return before(*args)
            cell.cell_contents=spy
            try:torch.mm(x,wt,out=out)
            finally:cell.cell_contents=before
            assert len(seen)==1 and all(u._cdata==v._cdata for u,v in zip(seen[0][:3],(x,wt,out)))
            assert seen[0][3:]==(m,n,k)
            result['records'].append(dict(geometry=name,m=m,original_fallback_handle_and_tensor_impls=True))
    result.update(completed=True,selected_cases=9,fallback_cases=6)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2));print(json.dumps(dict(completed=True,selected=9,fallback=6)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,default=Path('/workspace/MiniCPM5-2B'));p.add_argument('--out',type=Path,required=True)
    with torch.inference_mode():main(p.parse_args())
