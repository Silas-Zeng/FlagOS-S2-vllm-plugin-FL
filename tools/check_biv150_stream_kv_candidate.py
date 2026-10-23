"""Exercise the production GQA wrapper, including its batch-static cache policy."""

import argparse
import hashlib
import json
import socket
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
from biv150_benchmark_utils import (
    check_reference,
    full_call,
    graph_time,
    inputs,
    load_module,
)


class Capture:
    def __init__(self, kernel):
        self.kernel = kernel
        self.last = None
        self.policy = None

    def __getitem__(self, grid):
        def invoke(**kwargs):
            self.policy = kwargs.get("KV_CACHE_POLICY")
            self.last = self.kernel[grid](**kwargs)
            return self.last

        return invoke


def main(args):
    with socket.socket() as guard:
        assert guard.connect_ex(("127.0.0.1", 9031)) != 0, "Service running"
    baseline = load_module(args.baseline, "stream_kv_original")
    candidate = load_module(args.source, "stream_kv_candidate")
    captured = Capture(candidate._gqa_decode_attention)
    candidate._gqa_decode_attention = captured

    def fallback(**kwargs):
        raise RuntimeError("Unexpected fallback for audited eligible inputs")

    wrapper, _ = candidate._make_wrapper(
        SimpleNamespace(unified_attention=fallback, reduce_segments=None)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for batch, kv, mixed in [
        (64, 4096, False),
        (64, 5120, False),
        (32, 16384, False),
        (31, 16896, False),
        (16, 17408, False),
        (40, 16896, True),
        (8, 1024, True),
        (1, 100, False),
    ]:
        c = inputs(batch, kv, mixed=mixed)
        width = baseline._pick_segments(batch, 2, 16)
        full_call(baseline, c, width)
        ref = c["out"].clone()
        kwargs = dict(
            q=c["q"],
            k=c["k"],
            v=c["v"],
            out=c["out"],
            cu_seqlens_q=torch.arange(batch + 1, device="cuda", dtype=torch.int32),
            max_seqlen_q=1,
            seqused_k=c["lengths"],
            max_seqlen_k=kv,
            softmax_scale=128**-0.5,
            causal=True,
            block_table=c["table"],
            seq_threshold_3D=64,
            num_par_softmax_segments=16,
            softmax_segm_output=c["scratch"],
            softmax_segm_max=c["maxima"],
            softmax_segm_expsum=c["sums"],
            alibi_slopes=None,
            sinks=None,
            qq_bias=None,
            mm_prefix_range=None,
            output_scale=None,
            k_scale_cache=None,
            v_scale_cache=None,
            softcap=0.0,
            use_alibi_sqrt=False,
            window_size=(-1, -1),
            chunk_lookback=-1,
            use_td=False,
            kv_quant_mode=0,
        )
        before = candidate.stats()["calls"]
        wrapper(**kwargs)
        assert candidate.stats()["calls"] == before + 1
        expected = ".cg" if batch <= 8 else ".cv"
        assert captured.policy == expected
        torch.testing.assert_close(c["out"], ref, atol=0, rtol=0)
        check_reference(c)
        old = graph_time(partial(full_call, baseline, c, width), repeats=15, unroll=4)[
            0
        ]
        new = graph_time(partial(wrapper, **kwargs), repeats=15, unroll=4)[0]
        row = dict(
            batch=batch,
            kv=kv,
            mixed=mixed,
            segments=width,
            cache_policy=captured.policy,
            bit_exact=True,
            baseline_ms=old,
            candidate_ms=new,
            speedup=old / new,
            n_regs=captured.last.n_regs,
            n_spills=captured.last.n_spills,
            shared=captured.last.metadata.shared,
        )
        records.append(row)
        print(json.dumps(row), flush=True)
        args.out.write_text(
            json.dumps(
                dict(
                    baseline_sha256=hashlib.sha256(
                        args.baseline.read_bytes()
                    ).hexdigest(),
                    candidate_sha256=hashlib.sha256(
                        args.source.read_bytes()
                    ).hexdigest(),
                    records=records,
                ),
                indent=2,
            )
        )
        del c, ref, kwargs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
