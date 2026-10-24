"""Measure delayed V loads and retain exact IR evidence; installs no service."""

import argparse
import hashlib
import json
import socket
import statistics
import subprocess
from functools import partial
from pathlib import Path

import torch
from biv150_benchmark_utils import full_call, graph_time, inputs, load_module

PIN = "1a81409ba23a149081f7fc257460ea1512cbfdceeb0af32a2698994958749b9a"


def delayed_v(text):
    """Move only both V loads to immediately before their corresponding PV dot."""
    full = "        V = tl.load(value_cache_ptr + v_offsets)\n"
    tail = "        V = tl.load(value_cache_ptr + v_offsets, mask=column_valid[:, None], other=0.0)\n"
    anchor = "        accumulator = accumulator * rescale[:, None] + tl.dot(\n"
    assert text.count(full) == text.count(tail) == 1
    assert text.count(anchor) == 2
    changed = text.replace(full, "").replace(tail, "")
    first, middle, last = changed.split(anchor)
    changed = first + full + anchor + middle + tail + anchor + last
    # Invert exactly: no floating arithmetic, mask or synchronization was removed.
    restored = changed.replace(full + anchor, anchor, 1)
    restored = restored.replace(tail + anchor, anchor, 1)
    restored = restored.replace(
        "        K = tl.load(key_cache_ptr + k_offsets)\n",
        "        K = tl.load(key_cache_ptr + k_offsets)\n" + full,
        1,
    )
    masked_k = "        K = tl.load(key_cache_ptr + k_offsets, mask=column_valid[None, :], other=0.0)\n"
    restored = restored.replace(masked_k, masked_k + tail, 1)
    assert restored == text
    return changed


def reference(case, row):
    length = int(case["lengths"][row].cpu())
    if not length:
        return torch.zeros((16, 128), dtype=torch.bfloat16)
    ids = case["table"][row, : (length + 15) // 16].long()
    k = case["k"][ids].reshape(-1, 2, 128)[:length].cpu().double()
    v = case["v"][ids].reshape(-1, 2, 128)[:length].cpu().double()
    q = case["q"][row].reshape(2, 8, 128).cpu().double()
    p = torch.einsum("hgd,thd->hgt", q, k).mul(128**-0.5).softmax(-1)
    return torch.einsum("hgt,thd->hgd", p, v).reshape(16, 128).bfloat16()


class Capture:
    def __init__(self, kernel):
        self.kernel, self.last = kernel, None

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.last = self.kernel[grid](*args, **kwargs)
            return self.last

        return launch

    def save(self, folder):
        folder.mkdir()
        k = self.last
        result = dict(
            n_regs=k.n_regs,
            n_spills=k.n_spills,
            shared=k.metadata.shared,
            metadata=str(k.metadata),
            ir={},
        )
        for name in ("ttir", "ttgir", "llir"):
            value = k.asm[name]
            assert isinstance(value, str)
            (folder / ("kernel." + name)).write_text(value)
            result["ir"][name] = dict(
                sha256=hashlib.sha256(value.encode()).hexdigest(), bytes=len(value)
            )
        (folder / "metadata.json").write_text(json.dumps(result, indent=2))
        return result


def main(args):
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 9031)) == 0:
            raise RuntimeError("Existing service")
    if "No running processes found" not in subprocess.check_output(
        ["/usr/local/corex/bin/ixsmi"], text=True
    ):
        raise RuntimeError("GPU occupied")
    text = args.source.read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == PIN
    args.out.mkdir(parents=True, exist_ok=False)
    modules = {}
    for name, source in (("baseline", text), ("late_v", delayed_v(text))):
        path = args.out / (name + ".py")
        path.write_text(source)
        module = load_module(path, "decode_v_liveness_" + name)
        capture = Capture(module._gqa_decode_attention)
        module._gqa_decode_attention = capture
        modules[name] = module, capture
    result = dict(source_sha256=PIN, service_installed=False, records=[])
    result_file = args.out / "result.json"
    for batch in (8, 32, 40, 64):
        for length in (4096, 16896):
            for ragged in (False, True):
                row = dict(batch=batch, kv=length, ragged=ragged)
                try:
                    c = inputs(batch, length, mixed=ragged)
                    packed = torch.empty(
                        (c["k"].shape[0], 2, 16, 2, 128),
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    k, v = packed.unbind(1)
                    k.copy_(c["k"])
                    v.copy_(c["v"])
                    c.update(k=k, v=v)
                    row["kv_stride"] = list(k.stride())
                    baseline, candidate = modules["baseline"][0], modules["late_v"][0]
                    segments = baseline._pick_segments(batch, 2, 16)
                    row["segments"] = segments
                    full_call(baseline, c, segments)
                    gold = c["out"].clone()
                    for key in ("scratch", "maxima", "sums"):
                        c[key].fill_(float("nan"))
                    full_call(candidate, c, segments)
                    torch.testing.assert_close(c["out"], gold, atol=0, rtol=0)
                    rows = sorted({0, 1, batch - 1})
                    for index in rows:
                        torch.testing.assert_close(
                            c["out"][index].cpu(),
                            reference(c, index),
                            atol=0.005,
                            rtol=0.005,
                        )
                    folder = args.out / f"b{batch}-k{length}-r{int(ragged)}"
                    folder.mkdir()
                    row["compiled"] = {
                        name: capture.save(folder / name)
                        for name, (_, capture) in modules.items()
                    }
                    rounds = []
                    for iteration in range(3):
                        calls = [("baseline_ms", baseline), ("candidate_ms", candidate)]
                        if iteration % 2:
                            calls.reverse()
                        timing = {
                            name: graph_time(
                                partial(full_call, module, c, segments),
                                repeats=9,
                                unroll=4,
                            )[0]
                            for name, module in calls
                        }
                        timing["speedup"] = (
                            timing["baseline_ms"] / timing["candidate_ms"]
                        )
                        rounds.append(timing)
                    row["rounds"] = rounds
                    row["median_speedup"] = statistics.median(
                        x["speedup"] for x in rounds
                    )
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        full_call(candidate, c, segments)
                    for scale in (0.0, 1.0):
                        c["q"].normal_().mul_(scale)
                        for key in ("scratch", "maxima", "sums"):
                            c[key].fill_(float("nan"))
                        graph.replay()
                        torch.cuda.synchronize()
                        replayed = c["out"].clone()
                        full_call(baseline, c, segments)
                        torch.testing.assert_close(replayed, c["out"], atol=0, rtol=0)
                        for index in rows:
                            torch.testing.assert_close(
                                replayed[index].cpu(),
                                reference(c, index),
                                atol=0.005,
                                rtol=0.005,
                            )
                    row.update(passed=True, bit_exact=True, changed_input_graph=True)
                    del graph, replayed, c, packed, k, v, gold
                except Exception as error:
                    row.update(passed=False, error=repr(error))
                result["records"].append(row)
                result_file.write_text(json.dumps(result, indent=2))
                print(json.dumps(row), flush=True)
    assert len(result["records"]) == 16 and all(
        r.get("passed") for r in result["records"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
