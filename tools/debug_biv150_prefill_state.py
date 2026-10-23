"""Read private score/max/sum buffers from a cloned pinned prefill kernel."""
import hashlib,importlib,importlib.util,json,socket,sys
from pathlib import Path
import torch


def main():
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',9031))!=0,'Service running'
    original=importlib.import_module('vllm.v1.attention.ops.triton_unified_attention')
    zero='--zero-init' in sys.argv
    text=Path(original.__file__).read_text()
    assert hashlib.sha256(text.encode()).hexdigest()=='54c495dcea404582a1526a96ae7e472965cbd03c8fa0931c71ef72223bf04e1f'
    if zero:
        old_sum='L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)'
        assert text.count(old_sum)==1
        text=text.replace(old_sum,'L = tl.zeros([BLOCK_M], dtype=tl.float32)')
    anchor='        M, L, P, alpha = softmax_step(S, M, L)'
    store='''        debug_offset = (query_offset_0[:,None]*num_query_heads + query_offset_1[:,None])*DEBUG_KV + seq_offset[None,:]
        tl.store(DEBUG_S+debug_offset,S,mask=query_mask_0[:,None]&query_mask_1[:,None]&(seq_offset[None,:]<seq_len))
'''
    assert text.count(anchor)==1;text=text.replace(anchor,store+anchor)
    anchor='    # ---- Epilogue ---------------------------------------------------------'
    assert text.count(anchor)==1
    text=text.replace(anchor,'''    debug_row=query_offset_0*num_query_heads+query_offset_1
    tl.store(DEBUG_M+debug_row,M,mask=query_mask_0&query_mask_1)
    tl.store(DEBUG_L+debug_row,L,mask=query_mask_0&query_mask_1)

'''+anchor)
    text=text.replace('DEBUG_S','qq_bias_ptr').replace('DEBUG_M','segm_max_ptr').replace('DEBUG_L','segm_expsum_ptr').replace('DEBUG_KV','128')
    path=Path('/workspace/experiments/minicpm-s2/prefill-diagnostics-20261023/prefill-state-generated.py');path.write_text(text)
    spec=importlib.util.spec_from_file_location('prefill_state_private',path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    torch.manual_seed(20261023);qlen=16;kv=128
    q=torch.randn(qlen,16,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(kv//16,16,2,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    table=torch.randperm(kv//16,device='cuda',dtype=torch.int32)[None,:]
    cu=torch.tensor([0,qlen],device='cuda',dtype=torch.int32);lens=torch.tensor([kv],device='cuda',dtype=torch.int32)
    scores=torch.full((qlen,16,kv),float('nan'),device=q.device);maxima=torch.empty((qlen,16),device=q.device);sums=torch.empty_like(maxima)
    raw=module.kernel_unified_attention
    class Capture:
        def __getitem__(self,grid):
            def invoke(**kwargs):
                kwargs['use_sme']=0
                kwargs.update(qq_bias_ptr=scores,segm_max_ptr=maxima,segm_expsum_ptr=sums)
                return raw[grid](**kwargs)
            return invoke
    module.kernel_unified_attention=Capture()
    def call(fn,out):fn(q=q,k=k,v=v,out=out,cu_seqlens_q=cu,max_seqlen_q=qlen,seqused_k=lens,max_seqlen_k=kv,softmax_scale=128**-.5,causal=True,window_size=(-1,-1),block_table=table,softcap=0.,q_descale=1.,k_descale=1.,v_descale=1.)
    old=torch.empty_like(q);debug=torch.empty_like(q);call(original.unified_attention,old);call(module.unified_attention,debug)
    kd=k[table[0].long()].reshape(kv,2,128).float().repeat_interleave(8,1);vd=v[table[0].long()].reshape(kv,2,128).float().repeat_interleave(8,1)
    gold=torch.einsum('qhd,khd->qhk',q.float(),kd)*128**-.5
    valid=torch.arange(kv,device=q.device)[None,:]<=kv-qlen+torch.arange(qlen,device=q.device)[:,None]
    gold=gold.masked_fill(~valid[:,None,:],-float('inf'))
    gm=gold.max(-1).values;gl=(gold-gm[:,:,None]).exp().sum(-1)
    dense=torch.einsum('qhk,khd->qhd',gold.softmax(-1).to(q.dtype).float(),vd).to(q.dtype)
    result=dict(zero_init=zero,score_mask_mismatches=(torch.isfinite(scores)!=torch.isfinite(gold)).sum().item(),score_max_abs_diff=(scores[torch.isfinite(gold)]-gold[torch.isfinite(gold)]).abs().max().item(),max_diff=(maxima-gm).abs().max().item(),sum_diff=(sums-gl).abs().max().item(),original_vs_debug=(old.float()-debug.float()).abs().max().item(),debug_vs_dense=(debug.float()-dense.float()).abs().max().item(),first_state=dict(maxima=maxima[0,0].item(),gold_max=gm[0,0].item(),sum=sums[0,0].item(),gold_sum=gl[0,0].item()),kernel_type=str(type(original.kernel_unified_attention)))
    print(json.dumps(result),flush=True)
    (path.parent/('prefill-state-zero-debug.json' if zero else 'prefill-state-debug.json')).write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
