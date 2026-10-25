"""Exhaustive CPU conformance for the experimental scale1 V codec."""
import hashlib
import json
import math
from biv150_fp8_v_reference import (bf16_value,decode_e4m3,encode_nearest_reference,
                                    encode_integer_candidate,decode_integer_bf16_bits,
                                    bf16_bits_of_exact_value)


def main():
    finite=0;special=0;saturated=0;encoded=bytearray()
    for bits in range(65536):
        reference=encode_nearest_reference(bits)
        actual=encode_integer_candidate(bits)
        assert actual==reference,(hex(bits),actual,reference)
        encoded.append(actual)
        value=bf16_value(bits)
        if math.isfinite(value):
            finite+=1
            assert (actual&127)!=127
            saturated+=abs(value)>448
        else:special+=1
    for code in range(256):
        value=decode_e4m3(code)
        bits=decode_integer_bf16_bits(code)
        if math.isnan(value):assert math.isnan(bf16_value(bits))
        else:assert bits==bf16_bits_of_exact_value(value),(code,hex(bits),value)
    assert decode_e4m3(126)==448 and decode_e4m3(1)==2**-9
    assert encode_integer_candidate(0x8000)==128
    assert decode_integer_bf16_bits(128)==0x8000
    # Midpoints exist exactly in BF16. Both even directions are exercised.
    for lo in range(126):
        mid=(decode_e4m3(lo)+decode_e4m3(lo+1))/2
        bits=bf16_bits_of_exact_value(mid)
        assert bf16_value(bits)==mid
        expected=lo if lo%2==0 else lo+1
        assert encode_integer_candidate(bits)==expected
    print(json.dumps({'passed':True,'gpu_executed':False,'bf16_patterns':65536,
                      'finite_inputs':finite,'special_inputs':special,'fp8_decode_patterns':256,
                      'exact_midpoint_ties':126,'finite_saturated_inputs':saturated,
                      'encoding_sha256':hashlib.sha256(encoded).hexdigest(),
                      'rounding':'nearest-even','finite_overflow':'saturate to448',
                      'nonfinite_input':'signed canonical FP8 NaN','scale':1,
                      'limitation':'codec conformance only; no attention/model/performance/capacity claim'},indent=2))


if __name__=='__main__':main()
