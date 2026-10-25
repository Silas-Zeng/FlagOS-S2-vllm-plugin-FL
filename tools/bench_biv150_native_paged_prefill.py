"""Standalone native paged-prefill capability probe; never installed in serving.

Run only on the experiment owner's idle GPU. The native and accepted Triton
paths read the SAME shuffled, packed KV allocation and write the SAME out buffer.
No gather fallback is implemented: a rejected HND view is retained as evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import socket
import statistics
import time
import traceback
from pathlib import Path


CASES = {
    "chunk4k": ([2048], [4096]),
    "chunk16k": ([2048], [16384]),
    "ragged": ([1, 17, 33], [65, 130, 257]),
    "fresh": ([2048], [2048]),
}
BASE_COMMIT = "0bf9d2efeaa10144dd7f0ff4597298df029f9328"


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


def run_case(name, args, baseline, manager, native, record):
    compiled_before = set(manager.optimized)
    qlens, klens = CASES[name]
    seed = args.seed
    cpu = cpu_inputs(qlens, klens, seed, 0)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        device = {key: value.to("cuda") for key, value in cpu.items()}
        k, v = device["packed"].unbind(1)
        kh, vh = k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        out = torch.empty_like(device["q"])
    stream.synchronize()
    assert k.stride() == v.stride() == (8192, 256, 128, 1)
    assert kh.stride() == vh.stride() == (8192, 128, 256, 1)
    assert device["q"].stride() == (2048, 128, 1)
    assert device["table"].stride() == (8192, 1)
    assert stream.cuda_stream != torch.cuda.default_stream().cuda_stream
    record.update(qlens=qlens, klens=klens, stream=int(stream.cuda_stream),
                  layouts={"q": describe(device["q"]), "packed": describe(device["packed"]),
                           "k_nhd": describe(k), "k_hnd_view": describe(kh),
                           "v_hnd_view": describe(vh), "out": describe(out),
                           "table": describe(device["table"])},
                  gather_performed=False, unused_page_canary=37.0, tail_canary=43.0,
                  patterns=[], graph_replays=[])

    def generic():
        return baseline(q=device["q"], k=k, v=v, out=out,
                        cu_seqlens_q=device["cu_q"], seqused_k=device["lengths"],
                        max_seqlen_q=max(qlens), max_seqlen_k=max(klens),
                        softmax_scale=128 ** -0.5, causal=True, window_size=(-1, -1),
                        block_table=device["table"], softcap=0,
                        q_descale=None, k_descale=device["k_scale"],
                        v_descale=device["v_scale"])

    def candidate():
        return native(device["q"], kh, vh, device["cu_q"], device["cu_k"],
                      max(qlens), max(klens), dropout_p=0.0,
                      softmax_scale=128 ** -0.5, causal=True,
                      block_table=device["table"], out=out)

    def once(function, require_out=False):
        with torch.cuda.stream(stream):
            out.fill_(math.nan)
            result = function()
        stream.synchronize()
        if require_out:
            assert isinstance(result, torch.Tensor) and result.data_ptr() == out.data_ptr()
            assert result.shape == out.shape and result.stride() == out.stride()
        return out.cpu().clone()

    graph = None
    for pattern in range(2):
        # Separate layout/data RNGs preserve physical page assignment while
        # changing the contents of every captured Q/K/V input in pattern 1.
        cpu = cpu_inputs(qlens, klens, seed, pattern)
        with torch.cuda.stream(stream):
            for key, value in cpu.items():
                device[key].copy_(value)
        before = manager.selected_calls
        expected = once(generic)
        if manager.selected_calls <= before:
            raise RuntimeError("Current normalized/Q-fragment baseline was not selected")
        record["last_stage"] = f"pattern_{pattern}_native_eager"
        actual = once(candidate, require_out=True)
        indices, ref = reference(cpu, qlens, klens, pattern == 0, args.reference_all_rows)
        result = {"pattern": pattern, "uniform_q": pattern == 0,
                  "reference_rows": indices.tolist(), "reference_total_rows": len(indices),
                  "reference_all_rows": len(indices) == sum(qlens),
                  "q_sha256": digest(cpu["q"]), "packed_sha256": digest(cpu["packed"]),
                  "table_sha256": digest(cpu["table"]),
                  "native_vs_baseline_all_elements": compare(actual, expected, args.atol, args.rtol),
                  "native_vs_cpu_fp64": compare(actual[indices], ref, args.atol, args.rtol),
                  "baseline_vs_cpu_fp64": compare(expected[indices], ref, args.atol, args.rtol)}
        record["patterns"].append(result)
        if not all(result[key]["passed"] for key in
                   ("native_vs_baseline_all_elements", "native_vs_cpu_fp64", "baseline_vs_cpu_fp64")):
            raise RuntimeError("Numerical gate failed; no component timing")
        if graph is None:
            record["last_stage"] = "native_graph_capture"
            with torch.cuda.stream(stream):
                for _ in range(args.warmup):
                    candidate()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                candidate()
            stream.synchronize()
        for replay in range(args.graph_replays):
            record["last_stage"] = f"pattern_{pattern}_graph_{replay}"
            replay_out = once(graph.replay)
            check = compare(replay_out, actual, 0.0, 0.0)
            record["graph_replays"].append(dict(pattern=pattern, replay=replay, **check))
            if not check["passed"]:
                raise RuntimeError("Poisoned changed-input graph replay differs from native eager")
        # Input bytes, including unused/tail canaries, must remain intact.
        for key, value in cpu.items():
            got = device[key].cpu().contiguous().view(-1).view(torch.uint8)
            wanted = value.contiguous().view(-1).view(torch.uint8)
            if not torch.equal(got, wanted):
                raise RuntimeError("Native/baseline changed an input: " + key)
        result["input_bytes_unchanged"] = True

    # Keep Q-fragment eager dispatch active: its production wrapper intentionally
    # declines Graph capture. Capturing generic() would benchmark a different
    # baseline. Both timings therefore use warmed eager calls on this same stream.
    record["last_stage"] = "component_timing"
    values = {"baseline_ms": [], "native_ms": []}
    functions = {"baseline_ms": generic, "native_ms": candidate}
    selected_before = manager.selected_calls
    with torch.cuda.stream(stream):
        for _ in range(args.warmup):
            generic()
            candidate()
    stream.synchronize()
    events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    for repeat in range(args.repeats):
        order = list(functions) if repeat % 2 == 0 else list(reversed(functions))
        for key in order:
            with torch.cuda.stream(stream):
                events[0].record()
                functions[key]()
                events[1].record()
            events[1].synchronize()
            values[key].append(events[0].elapsed_time(events[1]))
    assert manager.selected_calls - selected_before == args.warmup + args.repeats
    # Final poisoned outputs also validate the actual timed input after timing.
    expected, actual = once(generic), once(candidate, require_out=True)
    record["post_timing_numeric"] = compare(actual, expected, args.atol, args.rtol)
    assert record["post_timing_numeric"]["passed"]
    record["timing"] = dict(values, baseline_median_ms=statistics.median(values["baseline_ms"]),
                            native_median_ms=statistics.median(values["native_ms"]))
    record["timing"]["speedup"] = (record["timing"]["baseline_median_ms"]
                                      / record["timing"]["native_median_ms"])
    record["baseline_new_compilations"] = {
        key: manager.optimized[key] for key in set(manager.optimized) - compiled_before}
    record["baseline_observed_candidate_hashes"] = sorted(manager.optimized)
    record.update(complete=True, component_correctness_passed=True, last_stage="complete")


def main(args):
    global torch
    if args.out.exists():
        raise FileExistsError("Use a new output path; prior evidence is never overwritten")
    report = {"schema": 1, "complete": False, "component_correctness_passed": False,
              "service_acceptance_passed": False, "diagnostic_only": True,
              "base_commit": BASE_COMMIT, "stage": "startup",
              "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "argv": vars(args) | {"out": str(args.out)},
              "timing_scope": "Alternating eager CUDA-event intervals on one nondefault stream; same preallocated out; includes dispatch gaps/internal copies. Hot component measurement, not service throughput.",
              "cases": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False))
    save()
    try:
        with socket.socket() as check:
            if check.connect_ex(("127.0.0.1", 9031)) == 0:
                raise RuntimeError("Port 9031 is occupied; coordinate the GPU owner first")
        report["stage"] = "imports_and_baseline_install"
        import torch
        torch.set_num_threads(args.cpu_threads)
        report["torch_version"] = torch.__version__
        baseline, manager, states = install_baseline()
        report["install_states"] = states
        native_module = importlib.import_module("ixformer.contrib.vllm_flash_attn.flash_attn_interface")
        native = native_module.flash_attn_varlen_func
        report.update(ixformer_version=importlib.metadata.version("ixformer"),
                      native_entrypoint=native.__module__ + "." + native.__name__,
                      native_source_sha256=hashlib.sha256(Path(native_module.__file__).read_bytes()).hexdigest())
        save()
        for name in args.case or list(CASES):
            report["stage"] = name
            record = {"name": name, "complete": False, "component_correctness_passed": False}
            report["cases"].append(record)
            save()
            print(json.dumps({"case": name, "stage": "starting"}), flush=True)
            started = time.monotonic()
            try:
                run_case(name, args, baseline, manager, native, record)
            except BaseException as error:
                record.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
                raise
            finally:
                record["elapsed_seconds"] = time.monotonic() - started
                report["baseline_q_fragment"] = manager.snapshot()
                save()
            print(json.dumps({"case": name, "complete": True, "timing": record["timing"]}), flush=True)
        report.update(complete=True, component_correctness_passed=True, stage="complete")
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--case", choices=CASES, action="append")
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--graph-replays", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--atol", type=float, default=0.005)
    parser.add_argument("--rtol", type=float, default=0.005)
    parser.add_argument("--reference-all-rows", action="store_true")
    parser.add_argument("--describe", action="store_true", help="Print CPU-only fixture plan without importing torch")
    args = parser.parse_args()
    if args.describe:
        print(json.dumps({"cases": CASES, "base_commit": BASE_COMMIT,
                          "nhd_stride": [8192, 256, 128, 1],
                          "hnd_view_stride": [8192, 128, 256, 1],
                          "block_table_stride": [8192, 1]}))
    else:
        if args.out is None or min(args.warmup, args.repeats, args.graph_replays, args.cpu_threads) < 1:
            parser.error("--out and positive warmup/repeats/graph-replays/cpu-threads are required")
        if not all(math.isfinite(v) and v >= 0 for v in (args.atol, args.rtol)):
            parser.error("atol/rtol must be finite and nonnegative")
        main(args)
