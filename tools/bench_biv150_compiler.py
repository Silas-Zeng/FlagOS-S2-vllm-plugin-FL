"""COREX register/SME options on the existing GQA algorithm (operator probe)."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from bench_biv150_fastpaths import (load_module, inputs, main_call, full_call,
                                   graph_time)


class Options:
    def __init__(self, kernel, extra):
        self.kernel, self.extra = kernel, extra

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def call(**kwargs):
            return launch(**kwargs, **self.extra)

        return call


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    base = load_module(args.source, "compiler_probe_base")
    options = {"baseline": {}}
    options.update({f"regs{n}": {"maxnreg": n} for n in [64, 80, 96, 128, 192]})
    options.update({f"sme{n}": {"use_sme": n} for n in [0, 15, 31, 47]})
    records = []
    for b, kv in [(64, 4096), (32, 16384)]:
        c = inputs(b, kv)
        seg = base._pick_segments(b, 2, 16)
        full_call(base, c, seg)
        ref = c["out"].clone()
        baseline_ms = graph_time(lambda: full_call(base, c, seg))[0]
        for name, extra in options.items():
            mod = SimpleNamespace(_gqa_decode_attention=Options(base._gqa_decode_attention, extra),
                                  _reduce_partials=base._reduce_partials, LOG2E=base.LOG2E)
            kernel = main_call(mod, c, seg)
            full_call(mod, c, seg)
            torch.testing.assert_close(c["out"], ref, atol=0.005, rtol=0.005)
            error = (c["out"].float()-ref.float()).abs().max().item()
            ms = graph_time(lambda: full_call(mod, c, seg))[0]
            row = dict(batch=b, kv=kv, segments=seg, name=name, options=extra,
                       n_regs=getattr(kernel, "n_regs", None),
                       spills=getattr(kernel, "n_spills", None),
                       baseline_ms=baseline_ms, median_ms=ms,
                       speedup=baseline_ms/ms, max_abs_diff=error)
            records.append(row)
            print(json.dumps(row), flush=True)
        del c, ref
        torch.cuda.empty_cache()
    (args.out / "results.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
