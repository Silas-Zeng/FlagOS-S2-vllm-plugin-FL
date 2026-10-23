"""Validate the production mixed-attention wrapper, fallback, and Graph inputs."""

import argparse
import importlib
import json
import math
import socket
from functools import partial
from pathlib import Path

import torch
from biv150_benchmark_utils import graph_time


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
    from vllm_fl.patches import (
        biv150_mixed_attention as mixed,
        biv150_prefill_normalized as norm,
    )

    baseline, _ = norm.make_normalized(original)
    assert norm.install()["installed"]
    metadata = mixed.install()
    assert metadata["installed"], metadata
    candidate_fn = original.unified_attention
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = dict(installation=metadata, service_installed=False, records=[])
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
        kwargs.update(
            softmax_segm_output=c["scratch"],
            softmax_segm_max=c["maxima"],
            softmax_segm_expsum=c["sums"],
        )
        selected = mixed.eligible(kwargs)
        old = partial(baseline, **kwargs)
        candidate = partial(candidate_fn, **kwargs)
        before_calls = mixed._state["selected_calls"]
        row = dict(
            qlens=qlens,
            klens=klens,
            selected=selected,
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
            assert mixed._state["selected_calls"] - before_calls == int(selected)
            if selected:
                assert mixed.scratch_for(kwargs)[0] is c["scratch"]
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
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
