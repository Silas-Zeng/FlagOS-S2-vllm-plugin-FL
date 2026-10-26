"""A source-checked conditional online-rescaling experiment; CPU import only."""
import ast
import hashlib


def rescale_source(text, threshold):
    if threshold not in (0.0, 8.0):
        raise ValueError("Only unchanged-max control or eight-bit anchor slack")
    tree = ast.parse(text)
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name == "_gqa_decode_attention"]
    if len(found) != 1:
        raise ValueError("Unique original GQA required")
    fn = found[0]
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    if names & {"proposed_max", "need_rescale", "probability_sum", "RESCALE_THRESHOLD"}:
        raise ValueError("Unexpected temporary collision")
    lines = text.splitlines(keepends=True)
    begin = min([fn.lineno] + [d.lineno for d in fn.decorator_list]) - 1
    original = "".join(lines[begin:fn.end_lineno])
    old = """        new_max = tl.maximum(running_max, tl.max(scores, 1))
        probability = tl.math.exp2(scores - new_max[:, None])
        rescale = tl.math.exp2(running_max - new_max)
        running_sum = running_sum * rescale + tl.sum(probability, 1)
        accumulator = accumulator * rescale[:, None] + tl.dot(
            probability.to(V.dtype), V
        )
        running_max = new_max
"""
    new = """        proposed_max = tl.maximum(running_max, tl.max(scores, 1))
        need_rescale = tl.max(proposed_max - running_max, 0) > RESCALE_THRESHOLD
        new_max = tl.where(need_rescale, proposed_max, running_max)
        probability = tl.math.exp2(scores - new_max[:, None])
        probability_sum = tl.sum(probability, 1)
        if need_rescale:
            rescale = tl.math.exp2(running_max - new_max)
            running_sum = running_sum * rescale + probability_sum
            accumulator = accumulator * rescale[:, None]
        else:
            running_sum = running_sum + probability_sum
        accumulator = accumulator + tl.dot(probability.to(V.dtype), V)
        running_max = new_max
"""
    if original.count(old) != 2:
        raise ValueError("Expected full and partial-page recurrences")
    changed = original.replace(old, new)
    if changed.replace(new, old) != original:
        raise ValueError("Exact inverse failed")
    generated = ("import triton\nimport triton.language as tl\n"
                 "LN2 = tl.constexpr(0.6931471805599453)\n"
                 + f"RESCALE_THRESHOLD = tl.constexpr({float(threshold)!r})\n\n"
                 + changed)
    compile(generated, "<gqa-rescale-skip>", "exec")
    return generated, dict(
        original_source_sha256=hashlib.sha256(text.encode()).hexdigest(),
        generated_source_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        exact_source_inverse=True, rescale_threshold_log2=threshold,
        probability_exponent_upper_bound=threshold,
        maximum_anchor_slack_log2=threshold,
        original_while_qhoist_pages_strides_chooser_reducer=True,
        floating_bitwise_equivalence_claim=False, model_acceptance_pending=True)
