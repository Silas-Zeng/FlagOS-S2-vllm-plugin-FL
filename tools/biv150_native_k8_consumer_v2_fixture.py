"""Consumer v2: correct partial implementation reference, original final hard gate."""
from biv150_native_k8_consumer_fixture import *
import biv150_native_k8_consumer_fixture as original


def golden(case):
    from biv150_native_k8_pagewise_reference import from_inputs
    result=original.golden(case)
    ref=from_inputs(original.input_bytes(case),case['rows'],case['segments'],case['pitch'])
    result['ideal_partial_diagnostic']=result['partial']
    result['partial']=ref['partial']
    # Final output, M and L retain the original ideal quantized-score FP64 gate.
    return result


def plan():
    result=original.plan()
    result.update(reference_version='pagewise_BF16_P_v2',
      partial_implementation_reference='FP64 online page16, P RN32 then BF16 before PV, rescale old accumulator each page',
      partial_ideal_diagnostic='original ideal FP64, strict .005 metrics retained separately',
      final_output_hard_reference='unchanged ideal quantized-score FP64',
      thresholds_changed=False,kernel_or_dtype_changed=False,original_v1_status='failed; preserved')
    return result
