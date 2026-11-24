"""Stdlib mixed-reader metadata/address/math contracts; no GPU behavior simulation."""
import math

from biv150_konly_decode_quant_reference import FORMAT, quantize_v2

STOCK = 'stock-bf16-v0'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metadata(cu_q, kv_lengths, total_query_capacity):
    require(type(total_query_capacity) is int and 0 <= total_query_capacity < 2**31, 'int32 query capacity required')
    require(1 <= len(kv_lengths) <= 64 and len(cu_q) == len(kv_lengths) + 1, 'Sequence scope1..64 and cu_q length required')
    require(all(type(x) is int and 0 <= x < 2**31 for x in cu_q), 'Valid int32 cumulative Q offsets required')
    require(cu_q[0] == 0 and cu_q[-1] <= total_query_capacity and all(a <= b for a, b in zip(cu_q, cu_q[1:])), 'Nondecreasing bounded cumulative Q required')
    require(all(type(x) is int and 0 <= x <= 131072 for x in kv_lengths), 'Audited KV capacity required')
    return [dict(sequence=i, q_start=a, q_end=b, q_len=b-a, kv_length=kv_lengths[i], decode=b-a == 1)
            for i, (a, b) in enumerate(zip(cu_q, cu_q[1:]))]


def cumulative(lengths):
    result = [0]
    for length in lengths:
        require(type(length) is int and length >= 0, 'Nonnegative query lengths required')
        result.append(result[-1] + length)
    return result


def validate_binding(*, mode, format_id, num_seqs, total_queries, query_strides, output_strides,
                     key_strides, value_strides, scale_strides=None, table_row_stride=8192,
                     query_dtype='bfloat16', value_dtype='bfloat16', query_storage='q', output_storage='out',
                     initial_source_storage='source-kv', encoded_destination_storage='packed-kv'):
    require(mode in ('bf16', 'k8') and format_id == (STOCK if mode == 'bf16' else FORMAT), 'Explicit producer-owned format ID required, never BF16 wrapper inference')
    require(type(num_seqs) is int and type(total_queries) is int and 1 <= num_seqs <= 64 and 0 <= total_queries < 2**31, 'Bounded mixed descriptors required')
    require(query_dtype == value_dtype == 'bfloat16', 'Original Q/V BF16 required')
    for strides in (query_strides, output_strides):
        require(len(strides) == 3 and strides[0] >= 2048 and tuple(strides[1:]) == (128, 1), 'Contiguous heads/dim with nonoverlapping query/output rows required')
    require(tuple(value_strides) == (8192, 256, 128, 1), 'Original V strides are BF16 elements, not bytes')
    require(tuple(key_strides) == ((8192, 256, 128, 1) if mode == 'bf16' else (16384, 256, 128, 1)), 'K strides must match their BF16/byte units')
    require(mode == 'bf16' and scale_strides is None or mode == 'k8' and tuple(scale_strides or ()) == (4096, 2, 1), 'Kscale is FP32 with field offset1024 from full-page base')
    require(type(table_row_stride) is int and table_row_stride >= 8192, 'Original block table backing stride required')
    require(query_storage is not None and output_storage is not None and query_storage != output_storage, 'Q and output cannot alias')
    if mode == 'k8':
        require(initial_source_storage is not None and encoded_destination_storage is not None and initial_source_storage != encoded_destination_storage,
                'Initial encoder source/destination must be separate; never in-place compression')
    return dict(descriptor_contract_passed=True, payload_format_authenticated=False,
                production_activation=False, capacity_reclaimed=False)


def addresses(*, mode, q_start, sequence, physical_page, slot, kv_head, query_head, dim,
              query_stride=2048, output_stride=2048, table_stride=8192):
    require(mode in ('bf16', 'k8'), 'Unknown cache arm')
    require(all(type(x) is int for x in (q_start, sequence, physical_page, slot, kv_head, query_head, dim)), 'Integer address operands required')
    require(0 <= q_start < 2**31 and 0 <= sequence < 64 and 0 <= physical_page < 2**31, 'Valid int32 metadata indices required')
    require(0 <= slot < 16 and 0 <= kv_head < 2 and 0 <= query_head < 16 and 0 <= dim < 128, 'Valid tile coordinates required')
    result = dict(query_bytes=2 * (q_start * query_stride + query_head * 128 + dim),
        output_bytes=2 * (q_start * output_stride + query_head * 128 + dim),
        partial_sequence_row=sequence, table_element=sequence * table_stride,
        v_bytes=physical_page * 16384 + 8192 + slot * 512 + kv_head * 256 + dim * 2)
    if mode == 'bf16':
        result['k_bytes'] = physical_page * 16384 + slot * 512 + kv_head * 256 + dim * 2
    else:
        result['k_bytes'] = physical_page * 16384 + slot * 256 + kv_head * 128 + dim
        result['k_scale_bytes'] = physical_page * 16384 + 4096 + slot * 8 + kv_head * 4
    return result


def writer_partition(cu_q, kv_lengths, total_queries, prefill_skips_decode):
    require(prefill_skips_decode is True, 'Future paired prefill reader must skip decode rows')
    rows = metadata(cu_q, kv_lengths, total_queries)
    decode = {r['q_start'] for r in rows if r['decode']}
    prefill = {q for r in rows if r['q_len'] > 1 for q in range(r['q_start'], r['q_end'])}
    require(not decode & prefill, 'Reader output ownership overlap')
    require(decode | prefill == set(range(cu_q[-1])), 'Incomplete output partition')
    return dict(decode_tokens=sorted(decode), prefill_tokens=sorted(prefill), padded_query_rows=list(range(cu_q[-1], total_queries)))


def attention(query, keys, values):
    """One head's original BF16/quantized-operand FP64 goals, with exact INT64 dots."""
    require(len(query) == 128 and len(keys) == len(values), 'Attention vector geometry')
    qcode, qscale = quantize_v2(query)
    originals, quantized, integers = [], [], []
    for key, value in zip(keys, values):
        require(len(key) == len(value) == 128 and all(math.isfinite(x) for x in value), 'Finite KV geometry required')
        kcode, kscale = quantize_v2(key)
        integer = sum(a*b for a, b in zip(qcode, kcode)); require(abs(integer) <= 128*127**2 < 2**24, 'Exact integer bound')
        reconstructed = math.fsum((a*qscale)*(b*kscale) for a, b in zip(qcode, kcode))
        exact_scaled = integer*qscale*kscale
        absolute = math.fsum(abs((a*qscale)*(b*kscale)) for a, b in zip(qcode, kcode))
        require(abs(reconstructed-exact_scaled) <= absolute*2**-44 + 2**-1074, 'INT64 vs independent FP64 operand goal')
        originals.append(math.fsum(a*b for a, b in zip(query, key)) * 128**-.5)
        quantized.append(reconstructed * 128**-.5); integers.append(integer)
    def combine(scores):
        if not scores:
            return [0.]*128
        peak=max(scores);weights=[math.exp(s-peak) for s in scores];denom=math.fsum(weights)
        return [math.fsum(p*v[d] for p, v in zip(weights, values))/denom for d in range(128)]
    return dict(original=combine(originals), quantized=combine(quantized), integer_scores=integers,
                cpu_fp64_goal_only=True, gpu_behavior_simulated=False)
