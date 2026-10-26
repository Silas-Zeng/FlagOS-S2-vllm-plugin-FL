#!/usr/bin/env python3
"""Bounded <=0 exp2 component factory and CPU-only numerical falsifier.

No production module is edited or imported. The generated Triton helper only
supports original GQA max-subtracted inputs. Positive-domain conditional-rescale
experiments must not be combined with this factory.
"""
import argparse
import ast
import ctypes
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import sys

BASELINE = '241220723a09061c182764f8f738e5ec01190367'
SOURCE_SHA = '2596cc09caf9b5b38b49755ae9af41ebdcf59a1d5ad889fc8e9319d1584438d1'
# Independently generated degree-5 Chebyshev-node interpolant on [-1,0],
# with c0 constrained to exactly 1 and coefficients rounded to binary32.
COEFF = (1.0, 0.6931431293487549, 0.24017895758152008,
         0.05529811978340149, 0.009209180250763893, 0.000946877000387758)


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def from_bits(x):
    return struct.unpack('<f', struct.pack('<I', x))[0]


def fmaf_cpu():
    fn = ctypes.CDLL(None).fmaf
    fn.argtypes = [ctypes.c_float] * 3
    fn.restype = ctypes.c_float
    # The separate multiply/add rounds to zero on this cancellation example.
    assert fn(f32(1 + 2**-23), f32(1 - 2**-23), -1.0) == -2**-46
    return fn


def software_exp2(x, fma):
    x = f32(x)
    if math.isnan(x) or x > 0:
        return float('nan')
    safe = max(x, -126.0)  # clamp before int conversion, including -inf/-FLT_MAX
    n = int(safe)  # round toward zero: r in (-1,0]
    r = f32(safe - n)
    p = COEFF[-1]
    for c in reversed(COEFF[:-1]):
        p = fma(p, r, c)
    out = f32(p * from_bits((n + 127) << 23))
    return 0.0 if x < -126 else out


def coefficient_derivation():
    """90-digit interpolation at six algebraic Chebyshev nodes.

    cos(15/45/75 degrees) have radical forms, avoiding dependence on a
    particular libm cos implementation. Decimal radicals are rounded at 90 digits.
    """
    with localcontext() as ctx:
        ctx.prec = 90
        sqrt2, sqrt6 = Decimal(2).sqrt(), Decimal(6).sqrt()
        cosines = [(sqrt6+sqrt2)/4, sqrt2/2, (sqrt6-sqrt2)/4]
        cosines += [-x for x in reversed(cosines)]
        nodes = [(x-1)/2 for x in cosines]
        a = [[x**j for j in range(6)] + [(x*Decimal(2).ln()).exp()] for x in nodes]
        for i in range(6):
            p = max(range(i, 6), key=lambda k: abs(a[k][i]))
            a[i], a[p] = a[p], a[i]
            scale = a[i][i]
            a[i] = [x / scale for x in a[i]]
            for k in range(6):
                if k != i:
                    scale = a[k][i]
                    a[k] = [u - scale*v for u, v in zip(a[k], a[i])]
        original = [a[k][-1] for k in range(6)]
        constrained = [Decimal(1), *original[1:]]
        rounded = tuple(f32(float(x)) for x in constrained)
        assert rounded == COEFF, (rounded, COEFF)
        ln2 = Decimal(2).ln()
        # Exact Cheb-node monic product bound is 2^-11. The 90-digit
        # Decimal radicals have absolute error well below 1e-80. A
        # telescoping product adds at most 6*delta*(1+delta)^5.
        delta = Decimal('1e-80')
        product_bound = Decimal(2)**-11 + 6*delta*(1+delta)**5
        interpolation = ln2**6 / Decimal(720) * product_bound
        zero_constraint = abs(original[0] - 1)
        coefficient_rounding = sum(abs(Decimal.from_float(c)-real)
                                   for c, real in zip(COEFF, constrained))
        u = Decimal(2)**-24
        gamma5 = 5*u / (1-5*u)
        fma_rounding = gamma5 * sum(abs(Decimal.from_float(c)) for c in COEFF)
        absolute = interpolation + zero_constraint + coefficient_rounding + fma_rounding
        return dict(nodes=[str(x) for x in nodes], coefficients_unrounded=[str(x) for x in constrained],
            coefficients_fp32=list(COEFF), polynomial_degree=5, fma_operations=5,
            node_absolute_perturbation_allowance=str(delta),
            interpolation_absolute_bound=float(interpolation),
            c0_constraint_shift=float(zero_constraint),
            coefficient_rounding_absolute_bound=float(coefficient_rounding),
            horner_fma_absolute_bound=float(fma_rounding),
            reduced_interval_absolute_bound=float(absolute),
            normal_domain_relative_bound=float(2*absolute),
            assumptions=['binary32 round-to-nearest-even FMA; no reassociation',
                'finite x in [-126,0]; RZ int conversion and exact range reduction',
                '90-digit algebraic Chebyshev nodes within 1e-80 of exact nodes',
                'subnormal inputs may flush; extra absolute effect <=2^-149',
                'x<-126 flushes explicitly with absolute error <2^-126; relative error is 1'])


def source_factory(text, scope='probability'):
    if scope not in ('probability', 'both'):
        raise ValueError('Only probability or both scope is supported')
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != SOURCE_SHA:
        raise ValueError('Original GQA source hash differs; max-subtracted domain must be re-audited')
    tree = ast.parse(text)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == '_gqa_decode_attention']
    if len(functions) != 1:
        raise ValueError('Unique original GQA function required')
    fn = functions[0]
    lines = text.splitlines(keepends=True)
    start = min([fn.lineno] + [d.lineno for d in fn.decorator_list])-1
    original = ''.join(lines[start:fn.end_lineno])
    substitutions = [('tl.math.exp2(scores - new_max[:, None])',
                      '_software_exp2_nonpositive(scores - new_max[:, None])')]
    if scope == 'both':
        substitutions.append(('tl.math.exp2(running_max - new_max)',
                              '_software_exp2_nonpositive(running_max - new_max)'))
    candidate = original
    for before, after in substitutions:
        if candidate.count(before) != 2:
            raise ValueError('Full/tail exp2 call sites changed')
        candidate = candidate.replace(before, after)
    restored = candidate
    for before, after in reversed(substitutions):
        restored = restored.replace(after, before)
    if restored != original:
        raise ValueError('Exact byte inverse failed')
    polynomial = '    p = ' + repr(COEFF[-1]) + '\n'
    polynomial += ''.join('    p = tl.fma(p, r, ' + repr(c) + ')\n'
                          for c in reversed(COEFF[:-1]))
    helper = ('@triton.jit\ndef _software_exp2_nonpositive(x):\n'
        '    # Restrict range before FP-to-int: -inf and -FLT_MAX are safe.\n'
        '    safe = tl.maximum(tl.minimum(x, 0.0), -126.0)\n'
        '    n = safe.to(tl.int32)\n'
        '    r = safe - n.to(tl.float32)\n' + polynomial +
        '    scale = ((n + 127) << 23).to(tl.float32, bitcast=True)\n'
        '    out = p * scale\n'
        '    out = tl.where(x < -126.0, 0.0, out)\n'
        '    # Fail visibly if a future caller violates this factory domain.\n'
        '    return tl.where(x <= 0.0, out, float("nan"))\n\n')
    primitive = ('@triton.jit\ndef _software_exp2_probe(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):\n'
        '    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n'
        '    x = tl.load(X + i, mask=i < N, other=0.0)\n'
        '    tl.store(Y + i, _software_exp2_nonpositive(x), mask=i < N)\n\n')
    generated = ('import triton\nimport triton.language as tl\n'
                 'LN2 = tl.constexpr(0.6931471805599453)\n\n' + helper + primitive + candidate)
    compile(generated, '<gqa-software-exp2>', 'exec')
    return generated, dict(original_source_sha256=digest,
        generated_source_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        exact_byte_inverse=True, scope=scope, exp2_domain='[-inf,0]',
        replaced_full_and_tail_call_sites=2*len(substitutions),
        original_while_page16_qhoist_layout_and_reducer=True,
        fp32_running_sums_preserved=True, positive_conditional_rescale_compatible=False,
        polynomial_coefficients=list(COEFF), explicit_flush_below=-126)


def probe_values(samples=8192):
    values = [-float('inf'), -from_bits(0x7f7fffff), -2147483648.0,
              -1000.0, -150.0, -149.0, -127.0, -126.0, -0.0, 0.0]
    for integer in range(127):
        b = bits(-float(integer))
        values.extend(from_bits(v) for v in (max(0, b-1), b, b+1)
                      if not math.isnan(from_bits(v)) and from_bits(v) <= 0)
    values.extend(f32(-126*i/samples) for i in range(samples+1))
    return values


def cpu_check(source, samples):
    fma = fmaf_cpu()
    derivation = coefficient_derivation()
    # Dyadic reduced grid + independent random bit-patterns + every integer
    # boundary in the actual input domain. Decimal is independent of libm exp2.
    rng = random.Random(20261027)
    values = probe_values(samples)
    values.extend(-i/samples for i in range(samples+1))
    values.extend(from_bits(rng.randrange(0x80000000, bits(-126.0)+1))
                  for _ in range(samples))
    worst_rel = (0.0, 0.0)
    worst_abs = (0.0, 0.0)
    normal_count = flushed_count = 0
    taylor4_worst = 0.0
    with localcontext() as ctx:
        ctx.prec = 65
        ln2 = Decimal(2).ln()
        for x in values:
            got = software_exp2(x, fma)
            assert math.isfinite(got) and got >= 0, (x, got)
            reference = Decimal(0) if x == -float('inf') else (Decimal.from_float(x)*ln2).exp()
            error = abs(Decimal.from_float(got)-reference)
            if x < -126:
                flushed_count += 1
                assert got == 0 and error < Decimal(2)**-126
                continue
            normal_count += 1
            rel = float(error/reference)
            if rel > worst_rel[0]:
                worst_rel = (rel, x)
            if float(error) > worst_abs[0]:
                worst_abs = (float(error), x)
            assert rel <= derivation['normal_domain_relative_bound'], (x, rel)
        for i in range(samples+1):
            r = -i/samples
            z = r*math.log(2)
            p = sum(z**k/math.factorial(k) for k in range(5))
            taylor4_worst = max(taylor4_worst, abs(p/math.exp2(r)-1))
    exact_zero_one = all(bits(software_exp2(x, fma)) == bits(1.0) for x in (-0.0, 0.0))
    exact_integers = all(bits(software_exp2(-float(i), fma)) == bits(math.exp2(-i))
                         for i in range(127))
    assert exact_zero_one and exact_integers
    assert math.isnan(software_exp2(1.0, fma)) and math.isnan(software_exp2(float('nan'), fma))
    proofs = {}
    text = (source/'vllm_fl/patches/biv150_gqa_decode.py').read_text()
    for scope in ('probability', 'both'):
        _, proofs[scope] = source_factory(text, scope)
    try:
        source_factory(text.replace('new_max = tl.maximum', 'new_max = tl.minimum', 1))
    except ValueError:
        rejects_modified_domain = True
    else:
        raise AssertionError('Changed max-update domain was accepted')
    return dict(cpu_only=True, gpu_executed=False, torch_imported='torch' in sys.modules,
        official_score=False, precision_gate=False, production_unchanged=True,
        samples_requested=samples, tested_normal_inputs=normal_count, tested_flushed_inputs=flushed_count,
        max_relative_error=worst_rel[0], max_relative_error_at=worst_rel[1],
        max_absolute_error=worst_abs[0], max_absolute_error_at=worst_abs[1],
        exact_zero_one=exact_zero_one, exact_integer_powers=exact_integers,
        positive_and_nan_fail_visible=True, rejects_modified_domain=rejects_modified_domain,
        taylor4_reduced_interval_max_relative_error=taylor4_worst,
        derivation=derivation, source_transforms=proofs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=32768)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.samples < 1024 or args.samples & (args.samples-1):
        parser.error('--samples must be a power of two >=1024')
    result = cpu_check(args.source, args.samples)
    data = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.out:
        args.out.write_text(data)
    print(data, end='')
