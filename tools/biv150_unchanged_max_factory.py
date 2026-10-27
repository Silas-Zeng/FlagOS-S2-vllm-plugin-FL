"""CPU-only reversible source factory. Importing this module never loads a GPU SDK."""
import ast
import hashlib
import math

MAIN = '241220723a09061c182764f8f738e5ec01190367'
SOURCE_SHA = '2596cc09caf9b5b38b49755ae9af41ebdcf59a1d5ad889fc8e9319d1584438d1'
KERNEL_SHA = 'b505e712ea21e10563ffb4a14f01a13a9c77fa32308095a80ebbc417b96e69ce'
NAME = '_gqa_decode_unchanged_max'
PROLOGUE = 'import triton\nimport triton.language as tl\nLN2 = tl.constexpr(0.6931471805599453)\n\n'
OLD = '''        new_max = tl.maximum(running_max, tl.max(scores, 1))
        probability = tl.math.exp2(scores - new_max[:, None])
        rescale = tl.math.exp2(running_max - new_max)
        running_sum = running_sum * rescale + tl.sum(probability, 1)
        accumulator = accumulator * rescale[:, None] + tl.dot(
            probability.to(V.dtype), V
        )
        running_max = new_max
'''
NEW = '''        new_max = tl.maximum(running_max, tl.max(scores, 1))
        probability = tl.math.exp2(scores - new_max[:, None])
        probability_sum = tl.sum(probability, 1)
        same_finite_max = (new_max == running_max) & (tl.abs(new_max) < float("inf"))
        need_rescale = tl.min(same_finite_max.to(tl.int32), 0) == 0
        if need_rescale:
            rescale = tl.math.exp2(running_max - new_max)
            running_sum = tl.fma(running_sum, rescale, probability_sum)
            accumulator = accumulator * rescale[:, None]
        else:
            running_sum = running_sum + probability_sum
        accumulator = accumulator + tl.dot(
            probability.to(V.dtype), V
        )
        running_max = new_max
'''


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def kernel_source(text):
    need(digest(text) == SOURCE_SHA, 'Unaudited GQA source')
    fn = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)
          and n.name == '_gqa_decode_attention']
    need(len(fn) == 1, 'Unique original kernel required')
    fn = fn[0]
    begin = min([fn.lineno] + [n.lineno for n in fn.decorator_list]) - 1
    result = ''.join(text.splitlines(keepends=True)[begin:fn.end_lineno])
    need(digest(result) == KERNEL_SHA, 'Original kernel bytes changed')
    return result


def restore(candidate):
    need(candidate.startswith(PROLOGUE), 'Private prologue changed')
    body = candidate[len(PROLOGUE):]
    need(body.count(NEW) == 2, 'Expected exactly full and tail transforms')
    need(body.count('def ' + NAME + '(') == 1, 'Private function name changed')
    restored = body.replace(NEW, OLD).replace('def ' + NAME + '(', 'def _gqa_decode_attention(', 1)
    need(digest(restored) == KERNEL_SHA, 'Inverse contains another kernel edit')
    return restored


def skip_contract(old, new):
    """Predicate algebra only: this does not simulate GPU max/exp/FTZ/dot."""
    need(len(old) == len(new) == 8, 'Exactly eight real Q heads required')
    return all(math.isfinite(b) and a == b for a, b in zip(old, new))


def pv_call_spans(tree):
    """Exact original PV call AST, excluding QK and unrelated dot operations."""
    expected = ast.dump(ast.parse('tl.dot(probability.to(V.dtype), V)', mode='eval').body,
                        include_attributes=False)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and
             ast.dump(node, include_attributes=False) == expected]
    need(len(calls) == 2, 'Expected exactly the full/tail original PV dot AST')
    return [dict(start=node.lineno, end=node.end_lineno,
                 argument_lines=sorted({arg.lineno for arg in node.args}),
                 call_ast_sha256=digest(expected)) for node in sorted(calls, key=lambda n: n.lineno)]


def build(text):
    original = kernel_source(text)
    need(original.count(OLD) == 2, 'Original recurrence anchors changed')
    occupied = {n.id for n in ast.walk(ast.parse(original)) if isinstance(n, ast.Name)}
    need(not occupied.intersection({'probability_sum', 'same_finite_max', 'need_rescale'}),
         'Temporary name collision')
    candidate = PROLOGUE + original.replace(OLD, NEW).replace(
        'def _gqa_decode_attention(', 'def ' + NAME + '(', 1)
    need(restore(candidate) == original, 'Exact source inverse failed')
    old_tree, new_tree = ast.parse(original), ast.parse(candidate)
    old_fn, new_fn = old_tree.body[0], new_tree.body[-1]
    need(ast.dump(old_fn.args) == ast.dump(new_fn.args), 'Raw parameter schema changed')
    pv_spans = pv_call_spans(new_fn)
    need([p['call_ast_sha256'] for p in pv_spans] ==
         [p['call_ast_sha256'] for p in pv_call_spans(old_fn)], 'PV call AST changed')
    lines = dict(alpha_exp=[], probability_exp=[], running_sum_fma=[], pv=[],
                 accumulator_scale=[], branch=[], qk=[])
    for node in ast.walk(new_fn):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == 'need_rescale':
            lines['branch'].append(node.lineno)
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            name = target.id if isinstance(target, ast.Name) else ''
            if name == 'rescale':
                lines['alpha_exp'].append(node.lineno)
            if name == 'probability':
                lines['probability_exp'].append(node.lineno)
            if name == 'running_sum' and isinstance(node.value, ast.Call) and \
                    isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'fma':
                lines['running_sum_fma'].append(node.lineno)
            if name == 'accumulator' and any(isinstance(n, ast.Call) and
                    isinstance(n.func, ast.Attribute) and n.func.attr == 'dot'
                    for n in ast.walk(node.value)):
                lines['pv'].append(node.lineno)
            if name == 'accumulator' and isinstance(node.value, ast.BinOp) and \
                    isinstance(node.value.op, ast.Mult):
                lines['accumulator_scale'].append(node.lineno)
            if name == 'scores' and any(isinstance(n, ast.Call) and
                    isinstance(n.func, ast.Attribute) and n.func.attr == 'dot'
                    for n in ast.walk(node.value)):
                lines['qk'].append(node.lineno)
    need(all(len(v) == 2 for v in lines.values()), 'Full/tail source map incomplete')
    proof = dict(main=MAIN, source_sha256=SOURCE_SHA, kernel_sha256=KERNEL_SHA,
                 generated_sha256=digest(candidate), exact_source_inverse=True,
                 same_raw_parameter_schema=True, source_lines=lines, pv_call_spans=pv_spans,
                 original_new_max_probability_and_pv_dot_preserved=True,
                 corrected_sum_explicit_fma=True, actual_ir_guard_pending=True,
                 ieee_identity_target_only=True, device_bitwise_equivalence_proven=False,
                 ftz_subnormal_and_signed_zero_device_falsifier_pending=True,
                 gpu_executed=False, production_installed=False, precision_gate=False)
    return candidate, proof
