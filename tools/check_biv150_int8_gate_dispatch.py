"""Check the real custom-op route, unchanged fallbacks, fake/fullgraph and replay."""

import argparse
import hashlib
import json
import os
import socket
from pathlib import Path

import torch


def compare(actual,expected,**kwargs):
    # Keep production FlagGems CUDA policy enabled; compare on CPU because its
    # CUDA isclose/reduction path cannot validate empty M0 outputs.
    assert actual.device==expected.device
    torch.testing.assert_close(actual.cpu(),expected.cpu(),**kwargs)


def main(args):
    with socket.socket() as port:assert port.connect_ex(('127.0.0.1',9031))!=0
    os.environ['BIV150_NATIVE_W8A8_LIBRARY']=str(args.library.resolve())
    from vllm_fl.patches.biv150_native_w8a8 import load_library,quantize_weight
    from vllm_fl.patches.biv150_int8_gemm_silu import gate_m32_m64_or_auto
    from vllm_fl.utils import use_flaggems,get_flag_gems_whitelist_blacklist
    import flag_gems
    assert use_flaggems()
    whitelist,blacklist=get_flag_gems_whitelist_blacklist()
    assert not whitelist and not blacklist
    flag_gems.enable(record=False,once=True)
    load_library();torch.manual_seed(20261025)
    weight=torch.randn(12288,2048,device='cuda',dtype=torch.bfloat16)*.02
    q,sw=quantize_weight(weight);marker=torch.zeros((),device='cuda',dtype=weight.dtype).expand_as(weight)
    report=dict(library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),records=[],inference_mode=torch.is_inference_mode_enabled(),
        int_mm_dispatch=torch._C._dispatch_dump_table("aten::_int_mm"))
    for rows in (0,1,8,16,17,31,32,33,40,63,64,65,128,256,2048,2049):
        x=torch.randn(rows,2048,device='cuda',dtype=torch.bfloat16)
        for name,w in (('compressed',marker),('bf16',weight)):
            original=torch.ops.biv150_native.gate_silu_auto(x,q,sw,w,17)
            observed=gate_m32_m64_or_auto(x,q,sw,w,17)
            selected=rows in (32,64) and name=='compressed'
            compare(observed,original,atol=.005 if selected else 0,rtol=.005 if selected else 0)
            report['records'].append(dict(m=rows,weight=name,selected=selected,passed=True,fallback_bit_exact=not selected))
    report['shape_checks']={}
    for rows in (32,64):
        x=torch.randn(rows,2080,device='cuda',dtype=torch.bfloat16)[:,:2048]
        compare(gate_m32_m64_or_auto(x,q,sw,marker,17),
            torch.ops.biv150_native.gate_silu_auto(x,q,sw,marker,17),atol=0,rtol=0)
        x=torch.randn(rows,2048,device='cuda',dtype=torch.bfloat16)
        compare(gate_m32_m64_or_auto(x,q,sw,marker,33),
            torch.ops.biv150_native.gate_silu_auto(x,q,sw,marker,33),atol=0,rtol=0)
        def call(t):return gate_m32_m64_or_auto(t,q,sw,marker,17)
        compiled=torch.compile(call,fullgraph=True,backend='inductor')
        compare(compiled(x),torch.ops.biv150_native.gate_silu_auto(x,q,sw,marker,17),atol=.005,rtol=.005)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):output=call(x)
        for factor in (0.,.5,4.):
            x.normal_().mul_(factor);graph.replay();torch.cuda.synchronize()
            compare(output,torch.ops.biv150_native.gate_silu_auto(x,q,sw,marker,17),atol=.005,rtol=.005)
            compare(compiled(x),torch.ops.biv150_native.gate_silu_auto(x,q,sw,marker,17),atol=.005,rtol=.005)
        report['shape_checks'][str(rows)]=dict(noncontiguous_fallback_bit_exact=True,
            unsupported_min_rows_fallback_bit_exact=True,fullgraph_and_fake=True,changed_input_graph=True)
    report.update(completed=True,fullgraph_and_fake=True,changed_input_graph=True)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(completed=True,rows=len(report['records']),fullgraph=True,changed_input_graph=True)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    with torch.inference_mode():main(p.parse_args())
