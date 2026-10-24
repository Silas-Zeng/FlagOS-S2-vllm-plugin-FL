"""Small-M BF16 components, including full split/reduce/allocation cost.

No model hook, no service score, no weight packing or duplicate representation.
Graph timings are warmed microbenchmarks; actual service dispatch is not proved.
"""

import argparse
import hashlib
import inspect
import json
import socket
import statistics
import subprocess
import traceback
from pathlib import Path

import torch
import triton

from biv150_benchmark_utils import graph_time
from biv150_bf16_small_splitk import CONFIGS, allocate, matmul, run_into


GEOMETRIES = (
    ("qkv", 2560, 2048),
    ("o", 2048, 2048),
    ("down", 2048, 6144),
    ("bf16_edge_gate", 12288, 2048),
    ("lm_head", 130560, 2048),
)
ROWS = (1, 8, 16, 31, 32, 40, 63, 64)


def sampled_reference(x, weight):
    m, k = x.shape
    n = weight.shape[0]
    rows = sorted({0, m//2, m-1})
    columns = sorted({0,1,17,31,127,n//2,n-1})
    # Corex device FP64 conversion was previously found unsuitable as an oracle.
    answer = (x[rows].cpu().double() @ weight[columns].cpu().double().t()).to(torch.bfloat16)
    return rows, columns, answer


def preserve_fixture(folder,prefix,x,weight,output,gold):
    actual=output.detach().cpu()
    reference=gold.detach().cpu()
    bad=((actual.float()-reference.float()).abs() > .005+.005*reference.float().abs()) | ~torch.isfinite(actual)
    indices=bad.flatten().nonzero().flatten()[:16].tolist()
    if not indices: indices=[0,reference.numel()//2,reference.numel()-1]
    n=weight.shape[0]
    rows=sorted({i//n for i in indices})
    columns=sorted({i%n for i in indices})
    path=folder/f"{prefix}.pt"
    torch.save(dict(x_rows=x[rows].cpu(),weight_rows=weight[columns].cpu(),
                    rows=rows,columns=columns,flat_indices=indices,
                    actual=actual[rows][:,columns],baseline=reference[rows][:,columns]),path)
    return dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def failure_details(error):
    chain=[]
    current=error
    visited=set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        chain.append(dict(type=type(current).__name__,message=str(current),repr=repr(current)))
        current=current.__cause__ or current.__context__
    return dict(error=repr(error),exception_chain=chain,traceback=traceback.format_exc())


def main(args):
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1",9031)) != 0
    assert "No running processes found" in subprocess.check_output(
        ["/usr/local/corex/bin/ixsmi"],text=True
    )
    import flag_gems

    baseline_path=Path(inspect.getsourcefile(inspect.unwrap(flag_gems.mm)))
    expected="4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090"
    assert hashlib.sha256(baseline_path.read_bytes()).hexdigest()==expected, baseline_path
    assert "_iluvatar" in str(baseline_path)
    tuning_path=baseline_path.parent.parent/"tune_configs.yaml"
    tuning_sha=hashlib.sha256(tuning_path.read_bytes()).hexdigest()
    assert tuning_sha=="dd1c7432a84b72dde9fe7d48527f4b1744383fa188a6feb69adb1d8fe5748bc0"
    args.out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(20261024)
    result=dict(component_only=True,service_installed=False,service_dispatch_proven=False,
                commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                baseline=dict(module=flag_gems.mm.__module__,file=str(baseline_path),sha256=expected,
                              tuning_file=str(tuning_path),tuning_sha256=tuning_sha),
                dtype="BF16 inputs/output; FP32 partials; no atomic accumulation",
                runtime=dict(torch=torch.__version__,triton=triton.__version__,
                             device=torch.cuda.get_device_name(),
                             total_memory=torch.cuda.get_device_properties(0).total_memory),
                duplicate_weights=False,packing=False,warmed_graph_microbenchmark=True,
                records=[],baseline_errors=[],tail_cases=[])
    def save():
        (args.out/"result.json").write_text(json.dumps(result,indent=2))

    # Irregular tails prove that split boundaries cover K exactly, beyond the
    # model's divisible dimensions. These are correctness cases, not score work.
    for m,n,k in ((3,2591,2017),(17,2051,6175)):
        x=torch.randn(m,k+13,device="cuda",dtype=torch.bfloat16)[:,:k]
        weight=torch.randn(n,k,device="cuda",dtype=torch.bfloat16)*0.02
        gold=flag_gems.mm(x,weight.t())
        ri,ci,host=sampled_reference(x,weight)
        torch.testing.assert_close(gold[ri][:,ci].cpu(),host,atol=.005,rtol=.005)
        for config in CONFIGS:
            row=dict(m=m,n=n,k=k,config=list(config))
            try:
                out,scratch=allocate(x,weight,config)
                if config[3]>1: scratch.fill_(float("nan"))
                run_into(x,weight,out,scratch,config)
                torch.testing.assert_close(out,gold,atol=.005,rtol=.005)
                torch.testing.assert_close(out[ri][:,ci].cpu(),host,atol=.005,rtol=.005)
                assert bool(torch.isfinite(out).all())
                row["passed"]=True
            except Exception as error:
                row.update(passed=False,**failure_details(error))
                if isinstance(error,AssertionError):
                    row["fixture"]=preserve_fixture(args.out,f"tail-{m}-{n}-{CONFIGS.index(config)}",x,weight,out,gold)
            result["tail_cases"].append(row)
            save()
        if not any(r["passed"] for r in result["tail_cases"] if r["m"]==m):
            result.update(completed=False,fatal_error="No variant passed tail prerequisite; matrix not started")
            save()
            raise RuntimeError(result["fatal_error"])
        del x,weight,gold

    for name,n,k in GEOMETRIES:
        # A single native [N,K] BF16 weight is shared by the baseline and all
        # candidates. No transposed duplicate is retained, even for LM head.
        weight=torch.randn(n,k,device="cuda",dtype=torch.bfloat16)*0.02
        for m in ROWS:
            for pitch in (k,k+32):
                x=torch.empty(m,pitch,device="cuda",dtype=torch.bfloat16)[:,:k]
                cases=[]
                for config in CONFIGS:
                    cases.append(dict(geometry=name,m=m,n=n,k=k,pitch=pitch,
                                      config=list(config),patterns=[],passed=all(
                                          r["passed"] for r in result["tail_cases"]
                                          if r["config"]==list(config)),
                                      skipped_after_tail_failure=not all(
                                          r["passed"] for r in result["tail_cases"]
                                          if r["config"]==list(config)),
                                      tail_checks_passed=all(
                                          r["passed"] for r in result["tail_cases"]
                                          if r["config"]==list(config))))
                invalid_baseline=False
                for pattern in ("zero","tiny","normal","large","outlier"):
                    x.normal_()
                    if pattern=="zero": x.zero_()
                    elif pattern=="tiny": x.mul_(1e-4)
                    elif pattern=="large": x.mul_(100)
                    elif pattern=="outlier": x[:,k//2]=32
                    before=x.clone()
                    gold=flag_gems.mm(x,weight.t())
                    ri,ci,host=sampled_reference(x,weight)
                    try:
                        torch.testing.assert_close(gold[ri][:,ci].cpu(),host,atol=.005,rtol=.005)
                        assert bool(torch.isfinite(gold).all())
                    except Exception as error:
                        result["baseline_errors"].append(dict(geometry=name,m=m,pitch=pitch,
                                                              pattern=pattern,error=repr(error)))
                        invalid_baseline=True
                        break
                    for config,row in zip(CONFIGS,cases):
                        if not row["passed"]: continue
                        try:
                            output,scratch=allocate(x,weight,config)
                            if config[3]>1: scratch.fill_(float("nan"))
                            compiled=run_into(x,weight,output,scratch,config)
                            torch.testing.assert_close(output,gold,atol=.005,rtol=.005)
                            torch.testing.assert_close(output[ri][:,ci].cpu(),host,atol=.005,rtol=.005)
                            torch.testing.assert_close(x,before,atol=0,rtol=0)
                            assert bool(torch.isfinite(output).all())
                            row["patterns"].append(dict(name=pattern,cpu_reference=True,
                                                        original_tolerance_pass=True,inputs_unchanged=True))
                            row["resources"]=dict(n_regs=getattr(compiled,"n_regs",None),
                                                  n_spills=getattr(compiled,"n_spills",None),
                                                  shared=getattr(getattr(compiled,"metadata",None),"shared",None))
                            del output,scratch
                        except Exception as error:
                            row.update(passed=False,failed_pattern=pattern,**failure_details(error))
                            if isinstance(error,AssertionError):
                                row["fixture"]=preserve_fixture(args.out,
                                    f"{name}-{m}-{pitch}-{CONFIGS.index(config)}-{pattern}",x,weight,output,gold)
                    del gold
                if invalid_baseline:
                    for row in cases: row.update(passed=False,baseline_invalid=True)
                else:
                    for config,row in zip(CONFIGS,cases):
                        if not row["passed"]: continue
                        replayed=None
                        try:
                            x.normal_()
                            def baseline(): return flag_gems.mm(x,weight.t())
                            def candidate(): return matmul(x,weight,config)
                            graph=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                replayed=candidate()
                            for factor in (0.0,0.5,4.0):
                                x.normal_().mul_(factor)
                                graph.replay()
                                torch.cuda.synchronize()
                                torch.testing.assert_close(replayed,baseline(),atol=.005,rtol=.005)
                            del graph,replayed
                            x.normal_()
                            rounds=[]
                            for iteration in range(3):
                                calls=[("baseline_ms",baseline),("candidate_ms",candidate)]
                                if iteration%2: calls.reverse()
                                timing={label:graph_time(fn,repeats=9,unroll=4)[0] for label,fn in calls}
                                timing["speedup"]=timing["baseline_ms"]/timing["candidate_ms"]
                                rounds.append(timing)
                            split=config[3]
                            row.update(changed_input_graph=True,rounds=rounds,
                                median_speedup=statistics.median(t["speedup"] for t in rounds),
                                min_round_speedup=min(t["speedup"] for t in rounds),
                                extra_fp32_workspace_bytes=split*m*n*4 if split>1 else 0,
                                bf16_weight_bytes=n*k*2,
                                promising_component=(statistics.median(t["speedup"] for t in rounds)>=1.03
                                                     and min(t["speedup"] for t in rounds)>=1.0
                                                     and row["tail_checks_passed"]))
                        except Exception as error:
                            row.update(passed=False,failed_stage="graph_or_timing",**failure_details(error))
                            if isinstance(error,AssertionError) and "replayed" in locals() and replayed is not None:
                                row["fixture"]=preserve_fixture(args.out,
                                    f"{name}-{m}-{pitch}-{CONFIGS.index(config)}-graph",x,weight,replayed,baseline())
                result["records"].extend(cases)
                save()
                print(json.dumps(dict(geometry=name,m=m,pitch=pitch,
                    passed=sum(r["passed"] for r in cases),tested=len(cases),
                    promising=sum(r.get("promising_component",False) for r in cases))),flush=True)
        del weight
    result["summary"]=dict(records=len(result["records"]),expected=640,
        passed=sum(r["passed"] for r in result["records"]),
        skipped_after_tail_failure=sum(r["skipped_after_tail_failure"] for r in result["records"]),
        promising_components=sum(r.get("promising_component",False) for r in result["records"]),
        tail_passed=sum(r["passed"] for r in result["tail_cases"]),tail_total=16)
    result["completed"]=(len(result["records"])==640 and not result["baseline_errors"])
    save()
    print(json.dumps(result["summary"]),flush=True)
    if not result["completed"]:
        raise RuntimeError("Incomplete matrix or invalid comparator: no promotion")
    # Exploration records may include failed variants. completed does not mean
    # every variant passed; any shipping choice requires its own full validation.


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    main(parser.parse_args())
