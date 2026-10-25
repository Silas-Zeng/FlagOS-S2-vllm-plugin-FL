"""Preserve strict BF16 numerical failures and describe ULP error on CPU."""
import torch

def describe(actual,reference):
    assert actual.dtype==reference.dtype==torch.bfloat16
    assert actual.device.type==reference.device.type=='cpu'
    finite=torch.isfinite(actual)&torch.isfinite(reference)
    delta=(actual.float()-reference.float()).abs()
    strict=(delta>.005+.005*reference.float().abs())|~finite
    def ordered(x):
        bits=x.contiguous().view(torch.int16).to(torch.int32)&65535
        return torch.where((bits&32768)!=0,65535-bits,bits+32768)
    ulp=(ordered(actual)-ordered(reference)).abs()
    ulp=torch.where(actual==reference,0,ulp)
    bad_finite=strict&finite
    return dict(elements=actual.numel(),nonfinite=int((~finite).sum()),
        strict_violations=int(strict.sum()),strict_passed=not bool(strict.any()),
        max_abs=float(delta.max()),max_finite_ulp=int(ulp[finite].max()) if bool(finite.any()) else None,
        strict_finite_violations=int(bad_finite.sum()),
        strict_finite_max_ulp=int(ulp[bad_finite].max()) if bool(bad_finite.any()) else 0,
        strict_violations_single_ulp=int((bad_finite&(ulp==1)).sum()),
        strict_violations_above_one_ulp=int((bad_finite&(ulp>1)).sum()))
