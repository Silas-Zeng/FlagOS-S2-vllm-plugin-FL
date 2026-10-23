"""Split mixed prefill/decode attention inside the operator, without scheduling.

Standalone generated kernels only. BF16, causal MiniCPM geometry; no model hook.
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import socket
import sys
from functools import partial
from pathlib import Path

import torch
from biv150_benchmark_utils import full_call, graph_time

EXPECTED_ATTENTION = "54c495dcea404582a1526a96ae7e472965cbd03c8fa0931c71ef72223bf04e1f"
EXPECTED_GQA = "1a81409ba23a149081f7fc257460ea1512cbfdceeb0af32a2698994958749b9a"


def generate(attention, gqa):
    old = "L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)"
    assert attention.count(old) == 1
    normalized = attention.replace(old, "L = tl.zeros([BLOCK_M], dtype=tl.float32)")
    anchor = (
        "    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:\n        return\n"
    )
    assert normalized.count(anchor) == 1
    skip = normalized.replace(
        anchor, anchor + "\n    if cur_batch_query_len == 1:\n        return\n"
    )
    assert gqa.count("    seq_lens_ptr,\n") == 2
    mixed = gqa.replace("    seq_lens_ptr,\n", "    seq_lens_ptr,\n    cu_query_ptr,\n")
    for index in ["seq_idx", "token_idx"]:
        needle = f"    seq_len = tl.load(seq_lens_ptr + {index})"
        assert mixed.count(needle) == 1
        before = (
            f"    query_start = tl.load(cu_query_ptr + {index}).to(tl.int64)\n"
            f"    query_end = tl.load(cu_query_ptr + {index} + 1).to(tl.int64)\n"
            "    if query_end - query_start != 1:\n        return\n"
        )
        mixed = mixed.replace(needle, before + needle)
    for old, new in [
        ("token_idx * query_stride_token", "query_start * query_stride_token"),
        (
            "output_ptr + token_idx * output_stride_token",
            "output_ptr + query_start * output_stride_token",
        ),
        (
            "weights = tl.exp(tl.where(seg_valid, segm_max, 0.0) - overall_max)",
            'weights = tl.exp(tl.where(seg_valid, segm_max - overall_max, float("-inf")))',
        ),
    ]:
        assert mixed.count(old) == 1, old
        mixed = mixed.replace(old, new)
    return normalized, skip, mixed


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SequenceGrid:
    def __init__(self, kernel, cu):
        self.kernel, self.cu = kernel, cu
        self.last = None

    def __getitem__(self, grid):
        def invoke(**kwargs):
            self.last = self.kernel[(self.cu.numel() - 1, *grid[1:])](
                cu_query_ptr=self.cu, **kwargs
            )
            return self.last

        return invoke


def make_inputs(qlens, klens):
    counts = [math.ceil(k / 16) for k in klens]
    pages = sum(counts)
    q = torch.randn(sum(qlens), 16, 128, device="cuda", dtype=torch.bfloat16)
    packed = torch.randn(pages, 2, 16, 2, 128, device="cuda", dtype=torch.bfloat16)
    k, v = packed.unbind(1)
    table = torch.full(
        (len(qlens), max(counts) + 32), -1, device="cuda", dtype=torch.int32
    )
    shuffled = torch.randperm(pages, device="cuda", dtype=torch.int32)
    offset = 0
    for row, count in enumerate(counts):
        table[row, :count] = shuffled[offset : offset + count]
        offset += count
    starts = [0]
    for length in qlens:
        starts.append(starts[-1] + length)
    cu = torch.tensor(starts, device="cuda", dtype=torch.int32)
    lengths = torch.tensor(klens, device="cuda", dtype=torch.int32)
    scratch = torch.empty(len(qlens), 16, 16, 128, device="cuda", dtype=torch.float32)
    maxima = torch.empty(len(qlens), 16, 16, device="cuda", dtype=torch.float32)
    return dict(
        q=q,
        k=k,
        v=v,
        table=table,
        cu=cu,
        lengths=lengths,
        scratch=scratch,
        maxima=maxima,
        sums=torch.empty_like(maxima),
        out=torch.empty_like(q),
    ), starts


def kwargs_for(c, qlens, klens):
    return dict(
        q=c["q"],
        k=c["k"],
        v=c["v"],
        out=c["out"],
        cu_seqlens_q=c["cu"],
        max_seqlen_q=max(qlens),
        seqused_k=c["lengths"],
        max_seqlen_k=max(klens),
        softmax_scale=128**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=c["table"],
        softcap=0.0,
        q_descale=1.0,
        k_descale=1.0,
        v_descale=1.0,
    )


def check_dense(c, qlens, klens, starts):
    checked = []
    selected = {0, len(qlens) - 1}
    selected.update(i for i, n in enumerate(qlens) if n == 1)
    # Cover every decode row in small scattered cases, sampled long rows otherwise.
    if len(selected) > 8:
        selected = {0, 1, len(qlens) // 2, len(qlens) - 2, len(qlens) - 1}
    for row in sorted(selected):
        qlen, klen = qlens[row], klens[row]
        if not qlen:
            continue
        ids = c["table"][row, : (klen + 15) // 16].long()
        k = c["k"][ids].reshape(-1, 2, 128)[:klen].float()
        v = c["v"][ids].reshape(-1, 2, 128)[:klen].float()
        for lo in sorted({0, max(0, qlen - 4)}):
            hi = min(qlen, lo + 4)
            q = (
                c["q"][starts[row] + lo : starts[row] + hi]
                .float()
                .reshape(hi - lo, 2, 8, 128)
            )
            scores = torch.einsum("qhgd,khd->hgqk", q, k) * 128**-0.5
            valid = (
                torch.arange(klen, device="cuda")[None, :]
                <= klen - qlen + torch.arange(lo, hi, device="cuda")[:, None]
            )
            probability = scores.masked_fill(~valid[None, None], -float("inf")).softmax(
                -1
            )
            gold = (
                torch.einsum("hgqk,khd->qhgd", probability, v)
                .reshape(hi - lo, 16, 128)
                .to(torch.bfloat16)
            )
            torch.testing.assert_close(
                c["out"][starts[row] + lo : starts[row] + hi],
                gold,
                atol=0.005,
                rtol=0.005,
            )
            checked.append(dict(sequence=row, lo=lo, hi=hi))
    return checked


def main(args):
    with socket.socket() as guard:
        assert guard.connect_ex(("127.0.0.1", 9031)) != 0, "Service active"
    torch.backends.cuda.matmul.allow_tf32 = False
    original = importlib.import_module("vllm.v1.attention.ops.triton_unified_attention")
    attention = Path(original.__file__).read_text()
    gqa = args.source.read_text()
    assert hashlib.sha256(attention.encode()).hexdigest() == EXPECTED_ATTENTION
    assert hashlib.sha256(gqa.encode()).hexdigest() == EXPECTED_GQA
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sources = generate(attention, gqa)
    modules = []
    for name, source in zip(["normalized", "skip_decode", "mixed_gqa"], sources):
        path = args.out.parent / f"{name}.py"
        path.write_text(source)
        modules.append(load(path, "biv150_mixed_probe_" + name))
    baseline, prefill, mixed = modules
    raw_main, raw_reduce = mixed._gqa_decode_attention, mixed._reduce_partials
    result = dict(
        attention_sha256=EXPECTED_ATTENTION,
        gqa_sha256=EXPECTED_GQA,
        generated_hashes=[hashlib.sha256(s.encode()).hexdigest() for s in sources],
        service_installed=False,
        records=[],
    )
    cases = [
        ([1] * 31 + [2017], [16384] * 32),
        ([1] * 63 + [1985], [4096] * 64),
        ([1] * 15 + [2033], [8192] * 16),
        ([1] * 31 + [17], [16384] * 32),
        ([1, 15, 0, 33], [100, 257, 0, 4097]),
        ([1024, 1024], [4096, 16384]),
        ([256] * 8, [4096] * 8),
        ([0, 1, 7, 1, 0, 2019, 1, 19], [0, 17003, 300, 17003, 0, 16384, 17003, 257]),
        ([2017] + [1] * 31, [16384] * 32),
        ([1000, 1, 0, 1, 1046], [4000, 16000, 0, 511, 16384]),
    ]
    for index, (qlens, klens) in enumerate(cases):
        torch.manual_seed(20261023 + index)
        c, starts = make_inputs(qlens, klens)
        kwargs = kwargs_for(c, qlens, klens)
        if index == 0:
            kwargs["max_seqlen_q"] = 2048  # fixed upper bound for graph input mutations
        mixed._gqa_decode_attention = SequenceGrid(raw_main, c["cu"])
        mixed._reduce_partials = SequenceGrid(raw_reduce, c["cu"])
        width = mixed._pick_segments(len(qlens), 2, 16)
        old = partial(baseline.unified_attention, **kwargs)

        def candidate(context=c, call_kwargs=kwargs, nsegments=width):
            prefill.unified_attention(**call_kwargs)
            full_call(mixed, context, nsegments)

        row = dict(
            qlens=qlens,
            klens=klens,
            segments=width,
            k_stride=list(c["k"].stride()),
            block_table_stride=c["table"].stride(0),
        )
        try:
            old()
            reference = c["out"].clone()
            c["out"].fill_(float("nan"))
            c["scratch"].fill_(float("nan"))
            c["maxima"].fill_(float("nan"))
            c["sums"].fill_(float("nan"))
            candidate()
            torch.cuda.synchronize()
            torch.testing.assert_close(c["out"], reference, atol=0.005, rtol=0.005)
            row["dense_checks"] = check_dense(c, qlens, klens, starts)
            rounds = []
            for number in range(3):
                order = [("baseline", old), ("candidate", candidate)]
                if number % 2:
                    order.reverse()
                times = {
                    name: graph_time(fn, repeats=11, unroll=4)[0] for name, fn in order
                }
                rounds.append(
                    dict(**times, speedup=times["baseline"] / times["candidate"])
                )
            row.update(
                passed=True,
                rounds=rounds,
                median_speedup=sorted(x["speedup"] for x in rounds)[1],
                max_abs_diff=(c["out"].float() - reference.float()).abs().max().item(),
                n_regs=mixed._gqa_decode_attention.last.n_regs,
                shared=mixed._gqa_decode_attention.last.metadata.shared,
            )
            if index == 0:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    candidate()
                mutations = []
                for n in [0, 15, 31]:
                    changed = [1] * n + [0] * (31 - n) + [2048 - n]
                    cu = [0]
                    for length in changed:
                        cu.append(cu[-1] + length)
                    c["cu"].copy_(torch.tensor(cu, device="cuda", dtype=torch.int32))
                    c["lengths"].copy_(
                        torch.tensor(
                            [16384 if q else 0 for q in changed],
                            device="cuda",
                            dtype=torch.int32,
                        )
                    )
                    c["out"].fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    got = c["out"].clone()
                    old()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(got, c["out"], atol=0.005, rtol=0.005)
                    c["out"].copy_(got)
                    check_dense(c, changed, [16384 if q else 0 for q in changed], cu)
                    mutations.append(dict(decode_rows=n, passed=True))
                row["changed_input_graph_replay"] = mutations
        except Exception as error:
            row["error"] = str(error)[:2200]
        result["records"].append(row)
        args.out.write_text(json.dumps(result, indent=2))
        print(json.dumps(row), flush=True)
        del c, kwargs, reference
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
