"""Owned source/slot fixtures, independent logical vectors and pitched bytes."""
import struct

from biv150_konly_i8_qk_reference import bf16, bf16_bits
from biv150_konly_strided_writer_contract import PAGE, SlotSnapshot, View

GUARD = 256
SPECIAL_V = (0, 0x8000, 1, 0x8001, 0x7f, 0x807f, 0x80, 0x8080,
             0x3f80, 0xbf80, 0x7f7f, 0xff7f, 0x7f80, 0xff80,
             0x7f81, 0x7fc1, 0xff81, 0xffc1)
SPECS = (
    ('empty', 256, 256, 0, 0, 0, False),
    ('one_slot_extra_rows', 256, 256, 1, 4, 3, False),
    ('tail17_k384_v640', 384, 640, 17, 20, 19, False),
    ('tail15_k640_v384', 640, 384, 15, 18, 20, False),
    ('k2560_v256', 2560, 256, 18, 20, 21, False),
    ('k256_v2560', 256, 2560, 16, 20, 19, False),
    ('shared_qkv_2560', 2560, 2560, 17, 20, 20, True),
    ('negative_only', 384, 640, 5, 8, 9, False),
    ('codec141', 384, 640, 141, 144, 145, False),
)


def make_case(spec, phase=0):
    name, kp, vp, count, kr, vr, shared = spec
    pages = max(3, (count+15)//16)
    values = tuple(((-2**63 if t % 2 else -1) if t % 7 == 3 else
                    (37*t+11*phase) % (pages*16)) for t in range(count))
    if name == 'negative_only':
        values = (-1, -2, -2**63, -17, -1)
    if name == 'codec141':
        values = tuple((37*t+11*phase) % (pages*16) for t in range(count))
    vectors = codec_vectors()
    keys, vals = [], []
    for t in range(count):
        keys.append([[bf16_bits(x) for x in vectors[(2*t+h+phase) % len(vectors)]]
                     if values[t] >= 0 else [0x7fc1]*128 for h in range(2)])
        vals.append([[SPECIAL_V[(t*256+h*128+d+phase) % len(SPECIAL_V)] for d in range(128)] for h in range(2)])
    if shared:
        total = GUARD+max(kr, vr)*2560*2+GUARD
        backing = bytearray(b'\xe1'*total)
        kb = vb = backing
        ko, vo = GUARD+2048*2, GUARD+2304*2
        kbase = vbase = 0x1000000
    else:
        kb = bytearray(b'\xe1'*(GUARD+kr*kp*2+GUARD))
        vb = bytearray(b'\xe1'*(GUARD+vr*vp*2+GUARD))
        ko = vo = GUARD
        kbase, vbase = 0x1000000, 0x2000000
    for t in range(count):
        for h in range(2):
            struct.pack_into('<128H', kb, ko+2*(t*kp+h*128), *keys[t][h])
            struct.pack_into('<128H', vb, vo+2*(t*vp+h*128), *vals[t][h])
    key = View('bfloat16', (kr, 2, 128), (kp, 128, 1), kbase, len(kb), ko)
    value = View('bfloat16', (vr, 2, 128), (vp, 128, 1), vbase, len(vb), vo)
    slots = View('int64', (count,), (1,), 0x3000000, GUARD+count*8+GUARD, GUARD)
    cache = View('uint8', (pages*PAGE,), (1,), 0x4000000, GUARD+pages*PAGE+GUARD, GUARD)
    return dict(name=name, pages=pages, key=key, value=value, slots=slots, cache=cache,
                snapshot=SlotSnapshot(values, 'component-owned-cpu-preupload', phase),
                buffers=dict(key=bytes(kb), value=bytes(vb)), logical_keys=keys, logical_values=vals,
                before=bytes((i*17+0xa5) % 256 for i in range(pages*PAGE)), shared_readonly_sources=shared)


def codec_vectors():
    # Finite BF16 zeros, half boundaries, exponent extremes and mixed signs.
    from biv150_konly_i8_qk_reference import bf16
    import random
    rows=[[0.]*128,[-0.]*128,[2.**-133]*128,[-2.**-133]*128,
          [struct.unpack('<f',struct.pack('<I',0x7f7f0000))[0]]*128]
    for peak in range(128,256):
        rows.append([float(peak),peak/2,-peak/2,-0.]+[bf16((-1)**d*peak*((d%31)/32)) for d in range(124)])
    rng=random.Random(20261027)
    for exponent in (-133,-126,-100,-10,0,10,100,120):
        rows.append([bf16(rng.uniform(-1,1)*2.**exponent) for _ in range(128)])
    return rows

