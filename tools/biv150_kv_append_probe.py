"""Experimental fused per-token/head KV quantization + paged-cache append.

Used only by an isolated operator probe; does not register serving hooks.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _append_quantized(
    key, value, slots, kc, vc, ks, vs,
    key_token:tl.constexpr,key_head:tl.constexpr,
    value_token:tl.constexpr,value_head:tl.constexpr,
    kc_block:tl.constexpr,kc_slot:tl.constexpr,kc_head:tl.constexpr,
    vc_block:tl.constexpr,vc_slot:tl.constexpr,vc_head:tl.constexpr,
    ks_block:tl.constexpr,ks_slot:tl.constexpr,ks_head:tl.constexpr,
    vs_block:tl.constexpr,vs_slot:tl.constexpr,vs_head:tl.constexpr,
    LIMIT:tl.constexpr,IS_INT:tl.constexpr,
):
    token=tl.program_id(0)
    slot=tl.load(slots+token).to(tl.int64)
    if slot < 0:
        return
    block=slot//16
    position=slot%16
    head=tl.arange(0,2)
    dim=tl.arange(0,128)
    kval=tl.load(key+token*key_token+head[:,None]*key_head+dim[None,:]).to(tl.float32)
    vval=tl.load(value+token*value_token+head[:,None]*value_head+dim[None,:]).to(tl.float32)
    kscale=tl.maximum(tl.max(tl.abs(kval),1),1e-8)/LIMIT
    vscale=tl.maximum(tl.max(tl.abs(vval),1),1e-8)/LIMIT
    kn=tl.minimum(tl.maximum(kval/kscale[:,None],-LIMIT),LIMIT)
    vn=tl.minimum(tl.maximum(vval/vscale[:,None],-LIMIT),LIMIT)
    if IS_INT:
        # Explicit round-to-nearest/even; no target-specific libdevice dependency.
        ka=tl.abs(kn); va=tl.abs(vn)
        kf=tl.floor(ka); vf=tl.floor(va)
        kr=ka-kf; vr=va-vf
        ki=kf+((kr>0.5)|((kr==0.5)&((kf.to(tl.int32)&1)==1))).to(tl.float32)
        vi=vf+((vr>0.5)|((vr==0.5)&((vf.to(tl.int32)&1)==1))).to(tl.float32)
        kn=tl.where(kn<0,-ki,ki)
        vn=tl.where(vn<0,-vi,vi)
    tl.store(kc+block*kc_block+position*kc_slot+head[:,None]*kc_head+dim[None,:],kn)
    tl.store(vc+block*vc_block+position*vc_slot+head[:,None]*vc_head+dim[None,:],vn)
    tl.store(ks+block*ks_block+position*ks_slot+head*ks_head,kscale)
    tl.store(vs+block*vs_block+position*vs_slot+head*vs_head,vscale)


def append_quantized(key,value,slots,kc,vc,ks,vs):
    assert key.shape[1:]==value.shape[1:]==(2,128)
    assert key.dtype==value.dtype==torch.bfloat16
    assert kc.dtype==vc.dtype and kc.dtype in (torch.int8,torch.float8_e4m3fn)
    limit=127.0 if kc.dtype==torch.int8 else torch.finfo(kc.dtype).max
    return _append_quantized[(key.shape[0],)](
        key,value,slots,kc,vc,ks,vs,
        *key.stride()[:2],*value.stride()[:2],
        *kc.stride()[:3],*vc.stride()[:3],*ks.stride(),*vs.stride(),
        LIMIT=limit,IS_INT=kc.dtype==torch.int8,num_warps=1,num_stages=1)
