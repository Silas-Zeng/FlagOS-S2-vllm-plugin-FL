"""Isolate repeated native/custom-op results under exact worker policy."""
import argparse,hashlib,json,os,socket
from pathlib import Path
import torch

p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--native-only',action='store_true');a=p.parse_args()
with socket.socket() as s:assert s.connect_ex(('127.0.0.1',9031))!=0
os.environ['BIV150_NATIVE_W8A8_LIBRARY']=str(a.library)
from vllm_fl.patches.biv150_native_w8a8 import load_library,quantize_weight
if not a.native_only:
 from vllm_fl.patches.biv150_int8_gemm_silu import gate_m32_m64_or_auto as wrapped
from vllm_fl.utils import use_flaggems,get_flag_gems_whitelist_blacklist
import flag_gems
assert use_flaggems() and not any(get_flag_gems_whitelist_blacklist())
flag_gems.enable(record=False,once=True);load_library();torch.set_num_threads(2)
a.out.mkdir(exist_ok=False,parents=True)
def delta(x,y):
 x=x.float();y=y.float();d=(x-y).abs();return dict(exact=torch.equal(x,y),mismatches=int((d!=0).sum()),max_abs=float(d.max()),violations=int((d>.005+.005*y.abs()).sum()))
with torch.inference_mode():
 torch.manual_seed(20261025)
 weight=torch.randn(12288,2048,device='cuda',dtype=torch.bfloat16)*.02
 q,sw=quantize_weight(weight);marker=torch.zeros((),device='cuda',dtype=weight.dtype).expand_as(weight)
 wh=weight.cpu();report=dict(native_only=a.native_only,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),records=[])
 native=torch.ops.biv150_native.gate_silu_auto
 if a.native_only:wrapped=native
 for rows in (1,8,16,17,32,64):
  x=torch.randn(rows,2048,device='cuda',dtype=weight.dtype)
  for name,w in (('compressed',marker),('bf16',weight)):
   results=[]
   for step in range(4):
    raw=native(x,q,sw,w,17);observed=wrapped(x,q,sw,w,17)
    torch.cuda.synchronize();results.append((raw.cpu(),observed.cpu()))
   item=dict(rows=rows,weight=name,pairs=[delta(v,u) for u,v in results],native_repeat=[delta(v[0],results[-1][0]) for v in results],wrapped_repeat=[delta(v[1],results[-1][1]) for v in results])
   if name=='bf16' and rows<=16:
    z=(x.cpu().double()@wh.double().t()).to(torch.bfloat16).float()
    gold=(torch.nn.functional.silu(z[:,:6144])*z[:,6144:]).to(torch.bfloat16)
    item['native_cpu']=[delta(v[0],gold) for v in results];item['wrapped_cpu']=[delta(v[1],gold) for v in results]
   report['records'].append(item);(a.out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(item),flush=True)
