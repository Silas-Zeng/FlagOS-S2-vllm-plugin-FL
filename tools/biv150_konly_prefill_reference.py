"""Independent stdlib metadata, address, mask and closed-attention oracles."""
from bisect import bisect_right
from collections import Counter
import math

from biv150_konly_decode_quant_reference import quantize_v2
from biv150_konly_i8_qk_reference import indices, f32, validate_vector
from biv150_konly_prefill_factory import FORMAT, PROFILES


def integer(value, lower=0, upper=2**63-1):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError('Bounded integer metadata required')
    return value


def lengths(query_lengths, key_lengths):
    if not query_lengths or len(query_lengths) != len(key_lengths):
        raise ValueError('Matching nonempty per-sequence metadata required')
    for q, k in zip(query_lengths, key_lengths):
        integer(q, 0, 2**31-1); integer(k, q, 8192*16)
    if sum(query_lengths) >= 2**31:
        raise ValueError('Query prefix must fit the original SDK I32 metadata')


def grid(profile, query_lengths, segments=1):
    if profile not in PROFILES:
        raise ValueError('Unknown private profile; never use a BF16 fallback')
    is3d, _, bq, *_ = PROFILES[profile]
    integer(segments, 1, 16)
    if segments not in (1, 2, 4, 8, 16):
        raise ValueError('Power-of-two segment shape required')
    if not is3d and segments != 1:
        raise ValueError('2D cannot use segmented scratch')
    return (sum(query_lengths)//bq + len(query_lengths), 2) + ((segments,) if is3d else ())


def rows(profile, query_lengths, key_lengths, segments=1):
    """Transcribe the pinned prefix search/row masks, not its numerical recurrence."""
    lengths(query_lengths, key_lengths)
    is3d, bm, bq, tile, *_ = PROFILES[profile]
    shape = grid(profile, query_lengths, segments)
    cu = [0]
    for q in query_lengths:
        cu.append(cu[-1]+q)
    starts = [q//bq+i for i, q in enumerate(cu)]
    for program in range(shape[0]):
        seq = bisect_right(starts, program)-1
        local = program-starts[seq]
        qlen, klen = query_lengths[seq], key_lengths[seq]
        if local*bq >= qlen:
            continue
        context = klen-qlen
        prefix = min(context+local*bq+(bm-1)//8+1, klen)
        ntiles = (prefix+tile-1)//tile
        per_segment = (klen+segments*tile-1)//(segments*tile) if is3d else 0
        for seg in range(segments if is3d else 1):
            if is3d and seg*per_segment*tile >= klen:
                continue
            lo, hi = (seg*per_segment, min((seg+1)*per_segment, ntiles)) if is3d else (0, ntiles)
            for kvhead in range(2):
                for m in range(bm):
                    qpos = local*bq+m//8
                    if qpos >= qlen:
                        continue
                    head = kvhead*8+m%8
                    keys = tuple(k for k in range(lo*tile, hi*tile)
                                 if k < prefix and k <= context+qpos)
                    yield dict(sequence=seq, query_position=qpos, token=cu[seq]+qpos,
                               head=head, segment=seg, keys=keys)


def coverage(profile, query_lengths, key_lengths, segments=1):
    seen = {}
    pairs = 0
    for row in rows(profile, query_lengths, key_lengths, segments):
        key = (row['sequence'], row['query_position'], row['head'])
        pairs += len(row['keys'])
        if profile == 'raw3d':
            seen.setdefault(key, Counter()).update(row['keys'])
        else:
            seq, qpos, _ = key
            allowed = tuple(range(key_lengths[seq]-query_lengths[seq]+qpos+1))
            if row['keys'] != allowed or key in seen:
                raise AssertionError('2D causal row duplicated or missing keys')
            seen[key] = 1
    expected = sum(query_lengths)*16
    if len(seen) != expected:
        raise AssertionError('Live query/head rows dropped or duplicated by grid')
    if profile == 'raw3d':
        for (seq, qpos, _), keys in seen.items():
            allowed = range(key_lengths[seq]-query_lengths[seq]+qpos+1)
            if keys != Counter(allowed):
                raise AssertionError('Causal key coverage changed across blocks/segments')
    return dict(query_head_rows=expected, grid=grid(profile, query_lengths, segments),
                causal_key_pairs=pairs)


def address(page, slot, head, dim):
    """Compare the device helper's mixed pointer units against packed byte oracle."""
    ko, so, vo = indices(page, slot, head, dim)
    computed = (page*16384+slot*256+head*128+dim,
                4*(page*4096+1024+slot*2+head),
                2*(page*8192+4096+slot*256+head*128+dim))
    if computed != (ko, so, vo):
        raise AssertionError('Byte/BF16/FP32 pointer units disagree')
    return computed


def tile_addresses(table, start, valid_prefix, head, tile=64):
    """Invalid tile lanes may fetch padded table metadata but never K/V/scales."""
    integer(start); integer(valid_prefix); integer(tile, 1, 64)
    result = []
    for token in range(start, start+tile):
        if token//16 >= len(table):
            raise ValueError('Unmasked SDK table load lacks physical padding')
        page = table[token//16]
        result.append(None if token >= valid_prefix else
                      tuple(address(page, token%16, head, d) for d in range(128)))
    return result


def validate_binding(profile, *, format_id, query_lengths, key_lengths, pages,
                     tensors, segments=1, table_pitch=8192, kv_quant_mode=0, use_td=False):
    """A CPU descriptor contract, not a runtime Tensor/codec validation claim."""
    if profile not in PROFILES or format_id != FORMAT:
        raise ValueError('Unknown encoded format/profile: stop; BF16 fallback forbidden')
    if type(kv_quant_mode) is not int or kv_quant_mode != 0 or type(use_td) is not bool or use_td:
        raise ValueError('Built-in dual-KV quant modes and TD are outside scope')
    lengths(query_lengths, key_lengths); integer(pages, 1, 2**31-1)
    if table_pitch != 8192 or type(table_pitch) is not int:
        raise ValueError('Pinned padded table pitch required')
    shape = grid(profile, query_lengths, segments)
    is3d, _, _, _, _, _, sequences = PROFILES[profile]
    total = sum(query_lengths)
    if total <= 0 or sequences is not None and len(query_lengths) != sequences:
        raise ValueError('Wrong sequence profile')
    if is3d and max(query_lengths) > 1 or not is3d and max(query_lengths) <= 1:
        raise ValueError('Raw3D and normalized prefill profiles are distinct')
    if profile in ('q32', 'multi2') and total < 1024:
        raise ValueError('Canonical BM256 parent requires at least1024 query tokens')
    expected = {
        'q': ('bfloat16', (total,16,128), (2048,128,1), None),
        'out': ('bfloat16', (total,16,128), (2048,128,1), None),
        'k': ('int8', (pages,16,2,128), (16384,256,128,1), 0),
        'kscale': ('float32', (pages*4096,), (1,), 0),
        'v': ('bfloat16', (pages,16,2,128), (8192,256,128,1), 4096),
    }
    if is3d:
        expected.update(scratch=('float32', (total,16,segments,128), (16*segments*128,segments*128,128,1), 0),
                        maxima=('float32', (total,16,segments), (16*segments,segments,1), 0),
                        sums=('float32', (total,16,segments), (16*segments,segments,1), 0))
    if set(tensors) != set(expected):
        raise ValueError('Missing/extra descriptor roles')
    for role, (dtype, dims, strides, offset) in expected.items():
        d = tensors[role]
        for value in (*d.get('shape', ()), *d.get('stride', ())):
            integer(value, 1)
        if (d.get('dtype'), tuple(d.get('shape', ())), tuple(d.get('stride', ()))) != (dtype, dims, strides):
            raise ValueError('Descriptor rank/shape/stride/units mismatch: '+role)
        integer(d.get('storage_offset')); integer(d.get('storage_bytes'), 1)
        if offset is not None and d['storage_offset'] != offset:
            raise ValueError('Pointer already offset or wrong view origin: '+role)
        if not isinstance(d.get('storage'), str) or not d['storage']:
            raise ValueError('Known storage identity required')
        itemsize = {'int8':1, 'bfloat16':2, 'float32':4}[dtype]
        end = d['storage_offset']+sum((n-1)*s for n,s in zip(dims,strides))+1
        if end*itemsize > d['storage_bytes']:
            raise ValueError('Descriptor exceeds storage')
    storage = {k:d['storage'] for k,d in tensors.items()}
    if not storage['k'] == storage['v'] == storage['kscale']:
        raise ValueError('Kcode/scale/BF16 V must describe one packed allocation')
    if any(tensors[k]['storage_bytes'] != pages*16384 for k in ('k','kscale','v')):
        raise ValueError('The physical page allocation remains16KiB')
    readers = {storage['q'], storage['k']}
    writes = [storage[k] for k in ('out','scratch','maxima','sums') if k in storage]
    if storage['q'] == storage['k'] or readers.intersection(writes) or len(set(writes)) != len(writes):
        raise ValueError('Illegal query/output/scratch alias')
    return dict(profile=profile, grid=shape, CPU_metadata_contract=True,
                actual_tensor_binding_verified=False, fallback='reject', capacity_reclaimed=False)


def attention(query, key_rows, value_rows, query_position, context, quantized):
    """Closed global FP64 references from original BF16; BF16-P/PV not emulated."""
    q = validate_vector(query)
    integer(query_position); integer(context)
    if len(key_rows) != len(value_rows) or context+query_position >= len(key_rows):
        raise ValueError('Independent original-row geometry mismatch')
    if quantized:
        qc, qs = quantize_v2(q)
    scores = []
    for key in key_rows[:context+query_position+1]:
        key = validate_vector(key)
        if quantized:
            kc, ks = quantize_v2(key)
            raw = sum(a*b for a,b in zip(qc,kc))
            assert abs(raw) <= 128*127**2 < 2**24
            scaled = math.fsum((a*qs)*(b*ks) for a,b in zip(qc,kc))
            assert abs(scaled-raw*qs*ks) <= 2**-46*math.fsum(abs((a*qs)*(b*ks)) for a,b in zip(qc,kc))+2**-1074
        else:
            scaled = math.fsum(a*b for a,b in zip(q,key))
        scores.append(scaled*128**-.5)
    anchor = max(scores)
    weight = [math.exp(s-anchor) for s in scores]
    denominator = math.fsum(weight)
    for row in value_rows:
        validate_vector(row)
    return [math.fsum(w*v[d] for w,v in zip(weight,value_rows))/denominator for d in range(128)]


def score_order(raw, qs, ks, scale):
    integer(raw, -128*127**2, 128*127**2)
    return f32(f32(f32(float(raw)*qs)*ks)*scale)
