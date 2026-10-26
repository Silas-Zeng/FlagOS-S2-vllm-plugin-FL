"""Optional Inductor-owned output buffer for the existing pure BF16 custom op.

No imports of Torch/vLLM or GPU work until explicit install/call. The callback
is used by generated code; eager and unsupported compiled inputs keep the old op.
"""
from __future__ import annotations
import os

_INSTALLED = False
_CHOICE = None
_RECORD_SELECTION = False
_SELECTED = set()
_PROVIDER_CALLS = {}
_LARGE_LINEAR = False
_PROVIDED_LINEAR = None
_PROVENANCE = None
_GEOMETRIES = ((2048,2048),(12288,2048),(2560,2048),(2048,6144))


def small_linear_out(x, weight, *, out):
    import torch
    from vllm_fl.patches.biv150_stable_small_o import _small_o_into
    eligible = (x.ndim == weight.ndim == 2 and x.shape[1] == weight.shape[1]
                and tuple(weight.shape) in _GEOMETRIES
                and x.dtype == weight.dtype == torch.bfloat16
                and x.device == weight.device == out.device and x.device.type == 'cuda'
                and out.dtype == x.dtype and tuple(out.shape) == (x.shape[0],weight.shape[0])
                and x.stride(1) == weight.stride(1) == 1
                and x.stride(0) >= x.shape[1] and weight.stride(0) >= weight.shape[1])
    if not eligible:
        out.copy_(torch.nn.functional.linear(x, weight))
        return out
    with torch.cuda.device(x.device):
        if 0 < x.shape[0] <= 8:
            if _RECORD_SELECTION:
                key=(x.shape[0],weight.shape[0],weight.shape[1])
                if key not in _SELECTED and not torch.cuda.is_current_stream_capturing():
                    _SELECTED.add(key)
                    print('BIV150_SMALL_OUT_SELECTED',{'rows':key[0],'output_columns':key[1],
                          'input_columns':key[2],'allocation':'compiler output buffer'},flush=True)
            return _small_o_into(x, weight, out)
        use_linear = (_LARGE_LINEAR and 128 <= x.shape[0] <= 2048
            and weight.stride(0) == weight.shape[1]
            and out.stride(1) == 1 and out.stride(0) >= out.shape[1]
            and not x.requires_grad and not weight.requires_grad and not out.requires_grad
            and out.untyped_storage().data_ptr() not in
                (x.untyped_storage().data_ptr(), weight.untyped_storage().data_ptr()))
        if _RECORD_SELECTION and not torch.cuda.is_current_stream_capturing():
            key=(x.shape[0],weight.shape[0],weight.shape[1],bool(use_linear))
            count=_PROVIDER_CALLS.get(key,0)+1;_PROVIDER_CALLS[key]=count
            if count & (count-1) == 0:
                print('BIV150_SMALL_OUT_PROVIDER_SELECTED',{'rows':key[0],
                      'output_columns':key[1],'input_columns':key[2],
                      'provider':'provided_linear' if use_linear else 'mm_out',
                      'call_count':count},flush=True)
        if use_linear:
            return _PROVIDED_LINEAR(x, weight, out)
        return torch.mm(x, weight.t(), out=out)


def install():
    global _CHOICE, _INSTALLED, _RECORD_SELECTION, _LARGE_LINEAR, _PROVIDED_LINEAR, _PROVENANCE
    requested = os.environ.get('VLLM_FL_BIV150_LARGE_LINEAR','0') == '1'
    if _INSTALLED:
        if requested != _LARGE_LINEAR:
            raise RuntimeError('Large linear policy changed after installation')
        return {'installed':True,'reason':'already installed','allocation':'Inductor out buffer',
                'large_linear':_LARGE_LINEAR,'provider_provenance':_PROVENANCE}
    if requested:
        from vllm_fl.patches import biv150_provided_linear as provided
        provenance = provided.initialize()
    else:
        provided = None;provenance = None
    import torch
    import sympy
    from torch._inductor import ir
    from torch._inductor.lowering import fallback_handler, lowerings, register_lowering
    from torch._inductor.select_algorithm import ExternKernelChoice
    target = torch.ops.biv150_stable_small_o.linear.default
    previous = lowerings.get(target)
    if previous is not None and not getattr(previous, '_is_fallback_handler', False):
        raise RuntimeError('Another lowering already owns the BF16 operator')
    previous = previous or fallback_handler(target, add_to_fallback_set=False)
    choice = ExternKernelChoice(small_linear_out, name='biv150_small_linear_out',
                               has_out_variant=True, op_overload=target)

    @register_lowering(target, type_promotion_kind=None)
    def lower(x, weight):
        xs, ws = x.get_size(), weight.get_size()
        if (len(xs) != 2 or len(ws) != 2
                or x.get_dtype() != torch.bfloat16 or weight.get_dtype() != torch.bfloat16
                or x.get_device() != weight.get_device() or x.get_device().type != 'cuda'
                or not all(isinstance(v,(int,sympy.Integer)) for v in (xs[1],ws[0],ws[1]))
                or xs[1] != ws[1] or tuple(ws) not in _GEOMETRIES):
            return previous(x,weight)
        x.realize();weight.realize()
        layout = ir.FixedLayout(x.get_device(),x.get_dtype(),[xs[0],ws[0]],[ws[0],1])
        return choice.bind([x,weight],layout).output_node()

    _RECORD_SELECTION = os.environ.get("VLLM_FL_BIV150_SMALL_OUT_RECORD_SELECTION","0")=="1"
    _LARGE_LINEAR = requested
    _PROVIDED_LINEAR = provided.linear_into if provided is not None else None
    _PROVENANCE = provenance
    _CHOICE = choice
    _INSTALLED = True
    return {'installed':True,'allocation':'Inductor out buffer',
            'small_rows':'unchanged masked BF16 kernel M1-8',
            'larger_rows':'provided linear M128-2048 when eligible; otherwise original MM' if requested else 'original torch.mm out dispatch',
            'large_linear':requested,'provider_provenance':provenance,
            'unsupported':'previous functional-op fallback'}
