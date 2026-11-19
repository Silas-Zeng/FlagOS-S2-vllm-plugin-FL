"""CPU arithmetic for a conditional research card; no kernel or serving policy."""
import json
import math
from pathlib import Path
import struct

PAGE = 16384
KEEP = 7
ROOT = Path(__file__).resolve().parents[1]


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def bf16(value):
    """Finite FP32 -> BF16 round-to-nearest, ties-to-even; no device FTZ model."""
    word = struct.unpack('<I', struct.pack('<f', f32(value)))[0]
    rounded = ((word + 0x7fff + ((word >> 16) & 1)) >> 16) << 16
    return struct.unpack('<f', struct.pack('<I', rounded))[0]


def bytes_at(start, width):
    return set(range(start, start + width))


def addresses(page, slot, head, dimension):
    base = page * PAGE
    return dict(k=base + slot * 512 + head * 256 + dimension * 2,
                code=base + 8192 + slot * 256 + head * 128 + dimension,
                scale=base + 12288 + slot * 8 + head * 4,
                saved=base + 12416 + slot * 512 + head * 256 + dimension * 2)


def active_reads(valid_slots):
    fields = {key: set() for key in ('k', 'code', 'scale', 'saved')}
    for slot in range(valid_slots):
        for head in range(2):
            for dimension in range(128):
                a = addresses(0, slot, head, dimension)
                fields['k'].update(bytes_at(a['k'], 2))
                if slot < KEEP:
                    fields['saved'].update(bytes_at(a['saved'], 2))
                else:
                    fields['code'].add(a['code'])
                    fields['scale'].update(bytes_at(a['scale'], 4))
    return fields


def main():
    regions = dict(k=(0, 8192), code=(8192, 12288), scale=(12288, 12416),
                   saved=(12416, 16000), padding=(16000, PAGE))
    seen = set()
    for first, last in regions.values():
        current = set(range(first, last))
        assert not seen.intersection(current)
        seen.update(current)
    assert seen == set(range(PAGE))
    for page in (0, 1, 3):
        for slot in range(16):
            for head in range(2):
                for dimension in range(128):
                    for kind, start in addresses(page, slot, head, dimension).items():
                        if kind == 'saved' and slot >= KEEP:
                            continue  # The future load must mask these addresses.
                        width = {'k': 2, 'code': 1, 'scale': 4, 'saved': 2}[kind]
                        low, high = regions[kind]
                        assert page * PAGE + low <= start < start + width <= page * PAGE + high
                        assert start % width == 0
    assert 12416 % 128 == 0 and 12416 % 256 == 128
    partial = []
    for count in range(17):
        fields = active_reads(count)
        useful = len(set().union(*fields.values()))
        expected = 512 * count + 512 * min(count, KEEP) + 264 * max(0, count - KEEP)
        assert useful == expected
        partial.append(dict(valid_slots=count, preserved=min(count, KEEP),
                            quantized=max(0, count - KEEP), useful_bytes=useful))
    fields = active_reads(16)
    useful = set().union(*fields.values())
    assert {key: len(value) for key, value in fields.items()} == dict(k=8192, code=2304, scale=72, saved=3584)
    lines = {str(width): len({byte // width for byte in useful}) * width for width in (32, 64, 128, 256)}
    assert lines['128'] == 14208 and lines['256'] == 14336
    # Separate loads may fetch the shared scale/saved 256B line more than once.
    separate256 = sum(len({byte // 256 for byte in value}) * 256 for value in fields.values())
    assert separate256 == 14592
    assert 7 * 1024 + 9 * 776 == len(useful) == 14152
    assert 8192 + 4096 + 128 + 3584 == 16000
    assert all(bf16(code) == code for code in range(-127, 128))
    probabilities = [f32(x) for x in (0., 2**-126, 2**-100, 1/3, .5,
                     .5 + 2**-9 - 2**-24, .5 + 2**-9, .5 + 2**-9 + 2**-24, .75, 1.)]
    values = [-127., -2., -.5, -0., 0., .5, 2., 127.]
    preserved_cases = 0
    for slot in range(KEEP):
        for probability in probabilities:
            for value in values:
                selected_scale = 1. if slot < KEEP else float('nan')
                assert bf16(f32(probability * selected_scale)) == bf16(probability)
                assert bf16(f32(probability * selected_scale)) * value == bf16(probability) * value
                preserved_cases += 1
    # One BF16 vector contains 2.0 and 0.5; its maxabs is 2.0.
    probability = f32(.5 + 2**-9)
    scale = f32(f32(2.) / 127.)
    value = .5
    code = math.floor(f32(value / scale) + .5)
    baseline = bf16(probability) * value
    approximate = bf16(f32(probability * scale)) * code
    assert code == 32 and baseline == .25 and approximate == .25390625
    bound = (abs(code) * abs(bf16(f32(probability * scale)) - probability * scale)
             + abs(probability) * abs(scale * code - value)
             + abs(probability - bf16(probability)) * abs(value))
    assert abs(approximate - baseline) <= bound + 1e-15
    result = dict(cpu_only=True, arithmetic_checks_passed=True, kernel_implemented=False,
                  gpu_executed=False, queued=False, precision_gate=False, model_evaluated=False,
                  model_source_still_frozen='e08c79456f9b943dbc4d9d02d0a4e030b4eceac1',
                  physical_bytes_per_page=PAGE, physical_capacity_reclaimed=False,
                  regions_half_open=regions, full_page_useful_read_bytes=len(useful),
                  ideal_unique_line_bytes=lines, separate_256B_region_load_bytes=separate256,
                  conditional_cache_write_bytes=14152, all_v8_plus_saved_cache_write_bytes=16000,
                  partial_pages=partial, preserved_operand_cases=preserved_cases,
                  nonpreserved_counterexample=dict(probability=probability, scale=scale, original_v=value,
                      code=code, original_pv_term=baseline, approximate_pv_term=approximate,
                      absolute_error_bound=bound),
                  limitation='Finite scalar operand arithmetic and ideal line coverage only; no device FTZ, '
                             'dot accumulation, actual transactions, speed or model precision claim.')
    out = ROOT / 'docs/evidence/prefill-multi2-20261027/v8-prefix7-research-cpu.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('cpu_only', 'arithmetic_checks_passed',
          'full_page_useful_read_bytes', 'ideal_unique_line_bytes', 'preserved_operand_cases', 'precision_gate')}))


if __name__ == '__main__':
    main()
