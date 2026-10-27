"""Standard-library contracts for a private K-only I8 QK primitive.

No framework imports. The encoder allocates a new destination; it never compresses
an existing K cache in place. CPU IEEE arithmetic is not a GPU/FTZ simulation.
"""
import math
import struct

PAGE_BYTES = 16384
K_SCALE_OFFSET = 4096
V_OFFSET = 8192
MIN_NORMAL = 2.0 ** -126


def f32(x):
    try:
        return struct.unpack('<f', struct.pack('<f', x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def f32_bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def bf16_bits(x):
    b = f32_bits(f32(x))
    if b & 0x7f800000 == 0x7f800000:
        raise ValueError('finite BF16 inputs required')
    return ((b + 0x7fff + ((b >> 16) & 1)) >> 16) & 0xffff


def bf16(x):
    return struct.unpack('<f', struct.pack('<I', bf16_bits(x) << 16))[0]


def validate_vector(values):
    values = tuple(values)
    if len(values) != 128 or any(not math.isfinite(x) or bf16(x) != x for x in values):
        raise ValueError('expected 128 finite, already-BF16 values')
    return values


def round_ratio_even(value, scale):
    """Exact rational nearest/even, independent of an FP32 divide instruction."""
    a, b = abs(value).as_integer_ratio()
    c, d = scale.as_integer_ratio()
    numerator, denominator = a * d, b * c
    q, r = divmod(numerator, denominator)
    if 2 * r > denominator or (2 * r == denominator and q & 1):
        q += 1
    return -q if value < 0 else q


def quantize(values):
    """One token/head scale: zero->1; otherwise RN32(max(peak/127,2^-126)).

    Codes use exact-ratio nearest/even and [-127,127]. This fixes a representation
    contract; no GPU quantizer or equivalence to GPU approximate division is claimed.
    """
    values = validate_vector(values)
    peak = max(abs(x) for x in values)
    scale = 1.0 if peak == 0 else f32(max(peak / 127.0, MIN_NORMAL))
    codes = tuple(max(-127, min(127, round_ratio_even(x, scale))) for x in values)
    assert all(abs(x - c * scale) <= scale / 2 for x, c in zip(values, codes))
    return codes, scale


def indices(page, slot, head, dim=0):
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (page, slot, head, dim)):
        raise ValueError('integer indices required')
    if not (0 <= page < 2**31 and 0 <= slot < 16 and 0 <= head < 2 and 0 <= dim < 128):
        raise ValueError('index outside packed page contract')
    base = page * PAGE_BYTES
    return (base + slot * 256 + head * 128 + dim,
            base + K_SCALE_OFFSET + 4 * (slot * 2 + head),
            base + V_OFFSET + 2 * (slot * 256 + head * 128 + dim))


def encode_pages(keys, values):
    """Sources are nested [page][slot][head][128] BF16 numbers; destination is new."""
    if len(keys) != len(values) or not keys:
        raise ValueError('matching nonempty source pages required')
    output = bytearray([0xa5]) * (len(keys) * PAGE_BYTES)
    for page in range(len(keys)):
        if len(keys[page]) != 16 or len(values[page]) != 16:
            raise ValueError('each page has exactly 16 slots')
        for slot in range(16):
            if len(keys[page][slot]) != 2 or len(values[page][slot]) != 2:
                raise ValueError('each slot has exactly two KV heads')
            for head in range(2):
                codes, scale = quantize(keys[page][slot][head])
                vv = validate_vector(values[page][slot][head])
                ko, so, vo = indices(page, slot, head)
                output[ko:ko + 128] = bytes(c & 255 for c in codes)
                struct.pack_into('<f', output, so, scale)
                for dim, value in enumerate(vv):
                    struct.pack_into('<H', output, vo + 2 * dim, bf16_bits(value))
    return bytes(output)


def decode_key(packed, page, slot, head):
    ko, so, _ = indices(page, slot, head)
    if ko + 128 > len(packed) or so + 4 > len(packed):
        raise ValueError('page outside source allocation')
    codes = struct.unpack_from('<128b', packed, ko)
    scale = struct.unpack_from('<f', packed, so)[0]
    if not math.isfinite(scale) or scale <= 0 or any(abs(c) > 127 for c in codes):
        raise ValueError('invalid code/scale payload')
    return codes, scale


def references(query_codes, query_scales, packed, page, head, valid):
    """Independent integer dot and reconstructed-operand FP64 sum.

    FP32 target order is RN32(RN32(INT32 * sQ) * sK). Invalid columns store raw
    integer zero and score -inf; the zero-length primitive has no softmax.
    """
    if len(query_codes) != 8 or len(query_scales) != 8 or not 0 <= valid <= 16:
        raise ValueError('logical M8 / N16 metadata required')
    if any(len(row) != 128 or any(isinstance(c, bool) or not isinstance(c, int) or abs(c) > 127 for c in row)
           for row in query_codes):
        raise ValueError('signed I8 [-127,127] rows required')
    if any(not math.isfinite(s) or s <= 0 or f32(s) != s for s in query_scales):
        raise ValueError('positive finite FP32 query scales required')
    integer, fp32, fp64 = [], [], []
    for row, qs in zip(query_codes, query_scales):
        ir, sr, rr = [], [], []
        for slot in range(16):
            if slot >= valid:
                ir.append(0); sr.append(-math.inf); rr.append(-math.inf)
                continue
            codes, ks = decode_key(packed, page, slot, head)
            raw = sum(a * b for a, b in zip(row, codes))
            assert abs(raw) <= 128 * 127**2 < 2**24
            value = f32(f32(float(raw) * qs) * ks)
            reference = math.fsum((float(a) * qs) * (float(b) * ks) for a, b in zip(row, codes))
            if not math.isfinite(value):
                raise ValueError('scaled score outside finite FP32 fixture domain')
            absolute_products = math.fsum(abs((float(a) * qs) * (float(b) * ks)) for a, b in zip(row, codes))
            assert abs(value - reference) <= (2**-22 * absolute_products + 2**-148)
            ir.append(raw); sr.append(value); rr.append(reference)
        integer.append(ir); fp32.append(sr); fp64.append(rr)
    return dict(integer=integer, fp32=fp32, quantized_operand_fp64=fp64)
