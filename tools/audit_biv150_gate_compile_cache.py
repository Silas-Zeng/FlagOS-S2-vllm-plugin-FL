"""Read existing Corex artifacts without importing Torch or touching the GPU."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def audit(root, out):
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for metadata_path in sorted(root.glob("*/fused_gate.json")):
        directory = metadata_path.parent
        metadata = json.loads(metadata_path.read_text())
        ir = (directory / "fused_gate.ttir").read_text()
        gpu_ir = (directory / "fused_gate.ttgir").read_text()
        assembly = (directory / "fused_gate.asm").read_text()
        match = re.search(r"arith.constant dense<0> : tensor<(\d+)x(\d+)xi32>", ir)
        if not match:
            raise ValueError(f"Cannot determine accumulator geometry: {metadata_path}")
        bm, bn = map(int, match.groups())
        aq_shapes = re.findall(rf"arith.constant dense<0> : tensor<{bm}x(\d+)xi8>", ir)
        if len(set(aq_shapes)) != 1:
            raise ValueError(f"Ambiguous AQ tile: {metadata_path}")
        bk = int(aq_shapes[0])
        constants = {name: (int(value), shape) for name, value, shape in re.findall(
            r"(%[\w]+) = arith.constant dense<([0-9]+)> : tensor<([0-9x]+)xi32>", ir)}
        row_limits = set()
        for rhs, shape in re.findall(r"arith.cmpi slt, %\w+, (%\w+) : tensor<([0-9x]+)xi32>", ir):
            value = constants.get(rhs)
            if value and value[0] in (17, 32, 40, 64, 256, 2048) and shape in (str(bm), f"{bm}x1"):
                row_limits.add(value[0])
        if len(row_limits) != 1:
            raise ValueError(f"Ambiguous row limit {row_limits}: {metadata_path}")
        resources = {}
        for key in ("vgpr_count", "vgpr_spill_count", "sgpr_spill_count", "private_segment_fixed_size"):
            values = re.findall(rf"\.{key}:\s*(\d+)", assembly)
            resources[key] = int(values[0]) if values else None
        record = dict(cache_key=directory.name, rows=row_limits.pop(),
            tile=[bm, bn, bk], warps=metadata["num_warps"],
            warp_size=metadata["warp_size"], shared_bytes=metadata["shared"],
            resources=resources, mma_layouts=re.findall(r"#mma[^\n]*", gpu_ir)[:2],
            layout_conversion_statements=gpu_ir.count("ttg.convert_layout"),
            static_only=True, runtime_occupancy_or_bandwidth_proven=False, files={})
        for path in directory.iterdir():
            if not path.is_file():
                continue
            data = path.read_bytes()
            record["files"][path.name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            # Keep raw binary artifacts in the export as well as readable IR.
            target = out / "raw" / directory.name / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        rows.append(record)
    result = dict(cache=str(root), gpu_not_accessed=True, rows=rows,
                  summary=dict(records=len(rows), warp_sizes=sorted({x["warp_size"] for x in rows}),
                      nonzero_vgpr_spills=sum(bool(x["resources"]["vgpr_spill_count"]) for x in rows)))
    (out / "audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.cache, args.out)
