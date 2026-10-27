"""CPU targets for on-load quantization of one stock BF16 K page, not a GPU simulator."""
import math

from biv150_konly_decode_quant_reference import quantize_v2
from biv150_konly_i8_qk_reference import f32, validate_vector


def tile_reference(query, keys, valid):
    if len(query) != 8 or len(keys) != 16 or type(valid) is not int or not 0 <= valid <= 16:
        raise ValueError('Logical 8x128 Q and 16x128 K rows, bounded tail required')
    qc, qs = zip(*(quantize_v2(row) for row in query))
    # The original tail load supplies zero for all 128 dimensions of an invalid
    # column. Do not inspect a poisoned/unmapped invalid stock row.
    kc, ks = zip(*(quantize_v2(validate_vector(keys[column]) if column < valid else [0.] * 128)
                   for column in range(16)))
    integer, target, closed = [], [], []
    for row in range(8):
        ii, ff, dd = [], [], []
        for column in range(16):
            raw = sum(int(a) * int(b) for a, b in zip(qc[row], kc[column]))
            assert abs(raw) <= 128 * 127 ** 2 < 2 ** 24
            ii.append(raw)
            ff.append(f32(f32(float(raw) * qs[row]) * ks[column]) if column < valid else -math.inf)
            dd.append(math.fsum((float(a) * qs[row]) * (float(b) * ks[column])
                                for a, b in zip(qc[row], kc[column])) if column < valid else -math.inf)
        integer.append(ii)
        target.append(ff)
        closed.append(dd)
    return dict(integer=integer, fp32=target, quantized_operand_fp64=closed,
                query_codes=qc, query_scales=qs, key_codes=kc, key_scales=ks)
