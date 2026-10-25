"""Postcompile producer/consumer integration with learned-weight SDK proxies.

The unchanged SDK forward is executed with candidate/reference call routing; attention/input-norm are
identity proxies. This is an integration check, not an actual model score.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import socket
import subprocess
import traceback
from unittest.mock import patch

import torch
from safetensors import safe_open


def snapshot(values):
    return [v.detach().cpu() for v in values]


def compare(a,b):
    error=(a.float()-b.float()).abs()
    return dict(bit_exact=torch.equal(a,b),finite=bool(torch.isfinite(a).all()),
                violations=int((error>.005+.005*b.float().abs()).sum()),max_abs=float(error.max()))


def weights(model,layer):
    keys=[f'model.layers.{layer}.{s}' for s in ('post_attention_layernorm.weight',
        'mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight')]
    keys.append(f'model.layers.{layer+1}.input_layernorm.weight')
    found={}
    for p in model.glob('*.safetensors'):
        with safe_open(str(p),framework='pt',device='cpu') as handle:
            for key in keys:
                if key in handle.keys():found[key]=handle.get_tensor(key)
    assert set(found)==set(keys)
    identities={key:hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest() for key,value in found.items()}
    return [found[key].cuda() for key in keys],identities


def main(args):
    with socket.socket() as port:assert port.connect_ex(('127.0.0.1',9031))!=0
    assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
    root=Path(__file__).resolve().parents[1]
    assert not subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip()
    args.out.mkdir(parents=True,exist_ok=False)
    os.environ['BIV150_NATIVE_W8A8_LIBRARY']=str(args.library.resolve())
    import flag_gems
    from vllm_fl.utils import use_flaggems,get_flag_gems_whitelist_blacklist
    white,black=get_flag_gems_whitelist_blacklist();assert use_flaggems() and not white and not black
    flag_gems.enable(record=False,once=True)
    from vllm_fl.patches.biv150_native_w8a8 import load_library,quantize_weight
    from vllm_fl.patches import biv150_live_carry_gate as own
    from vllm_fl.patches import biv150_live_carry_codegen as codegen
    from vllm.model_executor.models.llama import LlamaDecoderLayer
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.config import VllmConfig,set_current_vllm_config
    from vllm.ir.ops.layernorm import fused_add_rms_norm
    load_library()
    native=fused_add_rms_norm.impls['native'].impl_fn
    original=LlamaDecoderLayer.forward
    os.environ['VLLM_FL_BIV150_LIVE_CARRY_GATE']='1'
    installed=codegen.install();assert installed.get('installed') and not installed.get('already')
    patched=LlamaDecoderLayer.forward
    assert patched is original and installed['original_decoder_forward_unchanged']
    # Reference calls execute the exact original producer/consumer callables,
    # including when Inductor reuses the same loaded module for both functions.
    def reference_call(fn,*values):
        with patch.object(codegen,'supported',return_value=False):return fn(*values)
    test_config=VllmConfig()

    class IdentityNorm(torch.nn.Module):
        def forward(self,x,residual=None):return x if residual is None else (x,residual)
    class Attention(torch.nn.Module):
        def forward(self,positions,hidden_states):return hidden_states
    class Norm(torch.nn.Module):
        def __init__(self,w):
            super().__init__();self.weight=torch.nn.Parameter(w,requires_grad=False)
            self.variance_epsilon=1e-6;self.variance_size_override=None;self.has_weight=True
        def forward(self,x,residual):return native(x,residual,self.weight,1e-6)
    class Gate(torch.nn.Module):
        def __init__(self,wq,sw):
            super().__init__();self.register_buffer('_biv150_native_wq',wq)
            self.register_buffer('_biv150_native_sw',sw)
            self.weight=torch.nn.Parameter(torch.zeros((),device='cuda',dtype=torch.bfloat16).expand(12288,2048),requires_grad=False)
            self.bias=None
    class Down(torch.nn.Module):
        def __init__(self,w):super().__init__();self.weight=torch.nn.Parameter(w,requires_grad=False);self.bias=None
        def forward(self,x):return torch.nn.functional.linear(x,self.weight),None
    class MLP(torch.nn.Module):
        def __init__(self,wq,sw,down):
            super().__init__();self.gate_up_proj=Gate(wq,sw);self.down_proj=Down(down)
            with set_current_vllm_config(test_config):self.act_fn=SiluAndMul()
        def forward(self,x):
            g=self.gate_up_proj
            return self.down_proj(torch.ops.biv150_native.gate_silu_auto(x,g._biv150_native_wq,g._biv150_native_sw,g.weight,17))[0]
    class Layer(torch.nn.Module):
        def __init__(self,nw,wq,sw,down,nextw):
            super().__init__();self.input_layernorm=IdentityNorm();self.self_attn=Attention()
            self.post_attention_layernorm=Norm(nw);self.mlp=MLP(wq,sw,down)
            self.next_norm=Norm(nextw)

    result=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        integration_proxy_only=True,actual_sdk_decoder_forward=True,actual_model=False,
        reference_forces_original_producer_consumer=True,
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        installation=installed,records=[],device=torch.cuda.get_device_name())
    save=lambda:(args.out/'result.json').write_text(json.dumps(result,indent=2))
    torch.manual_seed(20261025);torch.set_num_threads(2)
    for layer_id in (4,20,37):
        (nw,gw,uw,dw,nextw),identities=weights(args.model,layer_id)
        wq,sw=quantize_weight(torch.cat((gw,uw)));del gw,uw
        layer=Layer(nw,wq,sw,dw,nextw)
        positions=torch.arange(2048,device='cuda',dtype=torch.int64)
        def reference(x,residual):
            down,carry=original(layer,positions,x,residual)
            y,r=layer.next_norm(down,carry);return y,r,down
        def candidate(x,residual):
            down,carry=patched(layer,positions,x,residual)
            y,r=layer.next_norm(down,carry);return y,r,down
        def promoted(x,residual):
            assert x.shape[0]==64
            _,carry=layer.post_attention_layernorm(x,residual)
            gate=own.gate_m64(x,residual,layer.post_attention_layernorm.weight,wq,sw)
            down,_=layer.mlp.down_proj(gate);y,r=layer.next_norm(down,carry);return y,r,down
        bf=torch.compile(reference,fullgraph=True,dynamic=True)
        cf=torch.compile(candidate,fullgraph=True,dynamic=True)
        pf=torch.compile(promoted,fullgraph=True,dynamic=True)
        # Match the real generic-profile trace before guarded64 capture.
        warm=torch.zeros((2048,2048),device='cuda',dtype=torch.bfloat16)
        warm_r=warm.clone();torch._dynamo.mark_static(warm,1);torch._dynamo.mark_static(warm_r,1)
        reference_call(bf,warm,warm_r);cf(warm,warm_r);del warm,warm_r
        for m in (64,32):
            x=torch.empty((m,2048),device='cuda',dtype=torch.bfloat16);r=torch.empty_like(x)
            torch._dynamo.mark_static(x,1);torch._dynamo.mark_static(r,1)
            record=dict(layer=layer_id,m=m,weight_sha256=identities,patterns=[],selected=m==64)
            result['records'].append(record)
            selections_before=codegen._SELECTIONS
            try:
                for pattern in ('zero','normal','large','cancel'):
                    x.normal_();r.normal_()
                    if pattern=='zero':x.zero_();r.zero_()
                    elif pattern=='large':x.mul_(100);r.mul_(100)
                    elif pattern=='cancel':r.copy_(-x);r[:,::17]+=.00390625
                    before=snapshot((x,r));gold=snapshot(reference_call(bf,x,r));actual=snapshot(cf(x,r))
                    if m==64:assert codegen._SELECTIONS>selections_before,dict(error='paired route not selected',patches=codegen._PATCHED)
                    expected=snapshot(pf(x,r)) if m==64 else gold
                    checks=[compare(a,b) for a,b in zip(actual,expected)]
                    assert all(v['bit_exact'] and v['finite'] for v in checks),checks
                    assert torch.equal(x.cpu(),before[0]) and torch.equal(r.cpu(),before[1])
                    record['patterns'].append(dict(name=pattern,promotion_or_fallback_bit_exact=True,
                        original_strict_comparison=[compare(a,b) for a,b in zip(actual,gold)]))
                from torch._subclasses.fake_tensor import FakeTensorMode
                with FakeTensorMode(allow_non_fake_inputs=True) as mode:
                    fake=own.gate_m64(mode.from_tensor(x),mode.from_tensor(r),mode.from_tensor(nw),mode.from_tensor(wq),mode.from_tensor(sw)) if m==64 else None
                    if fake is not None:assert fake.shape==(64,6144) and fake.dtype==torch.bfloat16
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):captured=cf(x,r)
                for factor in (0.,.5,4.):
                    x.normal_().mul_(factor);r.normal_().mul_(factor)
                    before=snapshot((x,r));graph.replay();torch.cuda.synchronize()
                    actual=snapshot(captured);eager=snapshot(cf(x,r));expected=snapshot(pf(x,r)) if m==64 else snapshot(reference_call(bf,x,r))
                    assert all(torch.equal(a,b) for a,b in zip(actual,eager))
                    assert all(torch.equal(a,b) for a,b in zip(actual,expected))
                    assert torch.equal(x.cpu(),before[0]) and torch.equal(r.cpu(),before[1])
                del graph,captured
                selected_calls=codegen._SELECTIONS-selections_before
                assert bool(selected_calls)==(m==64),dict(m=m,selected_host_calls=selected_calls,patches=codegen._PATCHED)
                record.update(passed=True,fullgraph=True,fake_checked=m==64,changed_input_graph=True,
                    selected_host_calls_not_graph_replays=selected_calls)
            except Exception as error:
                result['generated_pairs']=codegen._PATCHED
                record.update(passed=False,error=repr(error),traceback=traceback.format_exc());save();raise
            save()
    result.update(completed=True,passed=all(r['passed'] for r in result['records']),
        generated_pairs=codegen._PATCHED,decoder_forward_unchanged=LlamaDecoderLayer.forward is original,
        generated_cache=os.environ.get('TORCHINDUCTOR_CACHE_DIR'))
    save();print(json.dumps(dict(completed=True,passed=result['passed'],cases=len(result['records']))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path('/workspace/MiniCPM5-2B'))
    p.add_argument('--out',type=Path,required=True)
    with torch.inference_mode():main(p.parse_args())
