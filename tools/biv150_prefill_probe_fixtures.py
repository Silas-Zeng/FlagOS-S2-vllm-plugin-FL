"""Shared same-input paged-KV fixtures, CPU FP64 reference and accepted baseline.
Derived from the reviewed PR45 diagnostic; imports perform no device work.
"""
import math
import hashlib
import importlib


def cumulative(values):
    result = [0]
    for value in values:
        result.append(result[-1] + value)
    return result


def row_indices(qlen, all_rows):
    if all_rows or qlen <= 64:
        return list(range(qlen))
    return sorted({min(i, qlen - 1) for i in (0, 1, 15, 16, 31, 32, qlen // 2, qlen - 1)})


def describe(tensor):
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype), "device": str(tensor.device),
            "storage_offset": tensor.storage_offset()}


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def cpu_inputs(qlens, klens, seed, pattern):
    """Allocate unused canary pages and shuffle unique physical pages per sequence."""
    layout_generator = torch.Generator(device="cpu").manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed + 100 * pattern)
    counts = [(k + 15) // 16 for k in klens]
    blocks = 2 * sum(counts) + 5
    ids = torch.randperm(blocks - 1, generator=layout_generator)[:sum(counts)]
    table = torch.full((len(qlens), 8192), blocks - 1, dtype=torch.int32)
    packed = torch.full((blocks, 2, 16, 2, 128), 37.0, dtype=torch.bfloat16)
    offset = 0
    for seq, (count, length) in enumerate(zip(counts, klens)):
        pages = ids[offset:offset + count]
        table[seq, :count] = pages.to(torch.int32)
        values = torch.randn((count, 2, 16, 2, 128), generator=generator).to(torch.bfloat16)
        if pattern == 0:
            # An exact uniform-softmax case detects the causal boundary over ALL
            # query rows. Each sequence/head/channel has an identifiable value.
            position = torch.arange(count * 16).reshape(count, 16, 1, 1)
            marker = (2.0 * position / length - 1.0 + seq * 0.25
                      + torch.arange(2).reshape(1, 1, 2, 1) * 0.125
                      + (torch.arange(128) % 2).reshape(1, 1, 1, 128) * 0.05)
            values[:, 1] = marker.to(torch.bfloat16)
        # A partially used final page must obey seqused_k, not its full capacity.
        if length % 16:
            values[-1, :, length % 16:] = 43.0
        packed[pages] = values
        offset += count
    q = torch.randn((sum(qlens), 16, 128), generator=generator).to(torch.bfloat16)
    if pattern == 0:
        q.zero_()
    return {"q": q, "packed": packed, "table": table,
            "cu_q": torch.tensor(cumulative(qlens), dtype=torch.int32),
            "cu_k": torch.tensor(cumulative(klens), dtype=torch.int32),
            "lengths": torch.tensor(klens, dtype=torch.int32),
            "k_scale": torch.tensor(1.0, dtype=torch.float32),
            "v_scale": torch.tensor(1.0, dtype=torch.float32)}


def reference(cpu, qlens, klens, uniform, all_rows):
    """FP64 CPU reference: all rows for uniform Q, selected rows otherwise."""
    positions, outputs = [], []
    starts = cumulative(qlens)
    for seq, (qlen, kvlen) in enumerate(zip(qlens, klens)):
        ids = cpu["table"][seq, :(kvlen + 15) // 16].long()
        pages = cpu["packed"][ids]
        keys = pages[:, 0].reshape(-1, 2, 128)[:kvlen].double()
        values = pages[:, 1].reshape(-1, 2, 128)[:kvlen].double()
        rows = row_indices(qlen, all_rows or uniform)
        visible = kvlen - qlen + torch.tensor(rows, dtype=torch.int64) + 1
        assert min(visible) > 0 and max(visible) <= kvlen
        if uniform:
            means = values.cumsum(0)[visible - 1] / visible[:, None, None]
            answer = means.repeat_interleave(8, dim=1)
        else:
            query = cpu["q"][starts[seq] + torch.tensor(rows)].double()
            answer = torch.empty((len(rows), 16, 128), dtype=torch.float64)
            mask = torch.arange(kvlen)[None, :] < visible[:, None]
            for group in range(2):
                q = query[:, group * 8:(group + 1) * 8].permute(1, 0, 2)
                scores = (q @ keys[:, group].T) * (128 ** -0.5)
                probability = scores.masked_fill(~mask[None], -math.inf).softmax(-1)
                answer[:, group * 8:(group + 1) * 8] = (
                    probability @ values[:, group]).permute(1, 0, 2)
        positions.extend(starts[seq] + i for i in rows)
        outputs.append(answer)
    return torch.tensor(positions), torch.cat(outputs).to(torch.bfloat16)


def compare(actual, expected, atol, rtol):
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    diff = (actual.float() - expected.float()).abs()
    passed = finite and bool(torch.allclose(actual, expected, atol=atol, rtol=rtol))
    return {"finite": finite, "passed": passed,
            "numeric_equal": bool(torch.equal(actual, expected)),
            "max_abs_diff": float(diff.max()) if finite else None,
            "mismatched_elements": int((actual != expected).sum()),
            "outside_tolerance": int((diff > atol + rtol * expected.float().abs()).sum()) if finite else None,
            "elements": actual.numel(), "atol": atol, "rtol": rtol}


def install_baseline():
    states = {}
    for name in ("biv150_decode_launch_runtime", "biv150_gqa_decode",
                 "biv150_prefill_normalized", "biv150_q_fragment_service"):
        module = importlib.import_module("vllm_fl.patches." + name)
        states[name] = module.install()
        if not states[name].get("installed"):
            raise RuntimeError(f"Accepted baseline installation failed: {name}: {states[name]}")
    attention = importlib.import_module("vllm.v1.attention.ops.triton_unified_attention")
    manager = importlib.import_module("vllm_fl.patches.biv150_q_fragment_service")._MANAGER
    return attention.unified_attention, manager, states
