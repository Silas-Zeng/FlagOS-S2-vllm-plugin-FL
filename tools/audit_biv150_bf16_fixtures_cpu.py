"""Arbitrate saved BF16 disagreement slices on CPU; no GPU operations.

This audits the preserved subset, not complete matrices or model accuracy.
The original comparator failures stay failures until a new controlled test.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def check(actual,reference):
    actual=actual.float();reference=reference.float()
    delta=(actual-reference).abs()
    bad=(delta > .005+.005*reference.abs()) | ~torch.isfinite(actual)
    return dict(violations=int(bad.sum()),count=bad.numel(),
                max_abs=float(delta.max()),passed=not bool(bad.any()))


def main(args):
    torch.set_num_threads(2)
    source=json.loads((args.probe/"result.json").read_text())
    report=dict(cpu_only=True,fixture_subset_only=True,full_output_verified=False,
                model_accuracy_verified=False,source_commit=source["commit"],
                original_summary=source.get("summary"),records=[])
    args.out.mkdir(parents=True,exist_ok=False)
    for record in source["records"]:
        fixture=record.get("fixture")
        if not fixture:continue
        path=args.probe/fixture["path"]
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        assert sha==fixture["sha256"]
        data=torch.load(path,map_location="cpu",weights_only=True)
        assert all(t.device.type=="cpu" for t in data.values() if isinstance(t,torch.Tensor))
        reference=(data["x_rows"].double()@data["weight_rows"].double().t()).to(torch.bfloat16)
        selected=check(data["actual"],reference)
        baseline=check(data["baseline"],reference)
        # The slice includes the Cartesian product of saved rows/columns. Also
        # explicitly report the original failing flat coordinates themselves.
        original_coordinates=[]
        for index in data["flat_indices"]:
            r,c=divmod(index,record["n"])
            i=data["rows"].index(r);j=data["columns"].index(c)
            a=float(data["actual"][i,j]);b=float(data["baseline"][i,j]);gold=float(reference[i,j])
            original_coordinates.append(dict(row=r,column=c,actual=a,baseline=b,cpu_fp64_bf16=gold))
        report["records"].append(dict(
            geometry=record["geometry"],m=record["m"],pitch=record["pitch"],config=record["config"],
            failed_pattern=record.get("failed_pattern"),failed_stage=record.get("failed_stage"),
            fixture=fixture,actual_cpu=selected,baseline_cpu=baseline,
            original_coordinates=original_coordinates))
    rows=report["records"]
    report["summary"]=dict(fixtures=len(rows),
        actual_pass_baseline_fail=sum(r["actual_cpu"]["passed"] and not r["baseline_cpu"]["passed"] for r in rows),
        actual_fail_baseline_pass=sum(not r["actual_cpu"]["passed"] and r["baseline_cpu"]["passed"] for r in rows),
        both_pass=sum(r["actual_cpu"]["passed"] and r["baseline_cpu"]["passed"] for r in rows),
        both_fail=sum(not r["actual_cpu"]["passed"] and not r["baseline_cpu"]["passed"] for r in rows))
    (args.out/"result.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report["summary"]),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--probe",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    main(parser.parse_args())
