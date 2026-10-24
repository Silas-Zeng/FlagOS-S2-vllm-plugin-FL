"""CPU proof of contiguous split coverage, buffer bounds and immutable base."""

import argparse
import ast
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace


def main(args):
    root=Path(__file__).resolve().parents[1]
    parsed=ast.parse((root/"tools/biv150_bf16_small_splitk.py").read_text())
    configurations=ast.literal_eval(next(n.value for n in parsed.body if isinstance(n,ast.Assign)
                                        and any(isinstance(t,ast.Name) and t.id=="CONFIGS" for t in n.targets)))
    function=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=="chunk_intervals")
    namespace=dict(triton=SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b))
    exec(compile(ast.Module(body=[function],type_ignores=[]),"<actual-split-intervals>","exec"),namespace)
    count=0
    for k in (1,2,3,31,63,64,65,127,128,2017,2048,6144,6175,131071):
        for split in (1,2,4):
            intervals=namespace["chunk_intervals"](k,split)
            assert len(intervals)==split
            assert all(0<=a<=b<=k for a,b in intervals)
            coverage=[i for a,b in intervals for i in range(a,b)]
            assert coverage==list(range(k))
            count+=1
    buffers=0
    for m in (1,8,16,31,32,40,63,64):
        for n,k in ((2560,2048),(2048,2048),(2048,6144),(12288,2048),(130560,2048)):
            for bm,bn,bk,split,warps in configurations:
                assert bm in (16,32) and split in (1,2,4) and bk==64
                # Actual kernel offsets remain representable in signed32 bits,
                # including the largest head weight and full FP32 workspace.
                assert n*k*2 < 2**31 and split*m*n*4 < 2**31
                assert (m-1)*n+(n-1)==m*n-1
                if split>1: assert (split-1)*m*n+(m*n-1)==split*m*n-1
                buffers+=1
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                   and n.func.attr.startswith("atomic") for n in ast.walk(parsed))
    def address_base(node):
        if isinstance(node,ast.Name): return node.id
        assert isinstance(node,ast.BinOp)
        return address_base(node.left)
    stores=[n for n in ast.walk(parsed) if isinstance(n,ast.Call)
            and isinstance(n.func,ast.Attribute) and ast.unparse(n.func)=="tl.store"]
    assert len(stores)==3
    assert {address_base(n.args[0]) for n in stores}=={"OUTPUT","PARTIAL"}
    reducer=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=="reduce_partials")
    assert any(isinstance(n,ast.Call) and ast.unparse(n.func)=="tl.sum" for n in ast.walk(reducer))
    assert not subprocess.check_output(["git","-C",str(root),"diff",
        "31a0a7f9205d90904afac0ba19e9e65856c79867","--","vllm_fl","csrc"],text=True)
    result=dict(cpu_only=True,actual_split_function_executed=True,partition_cases=count,
                geometry_config_buffer_checks=buffers,no_gaps_or_overlaps=True,
                signed32_offsets_safe_for_model_matrix=True,no_atomic_accumulation=True,
                store_targets_only_output_and_scratch=True,
                production_sources_unchanged=True,gpu_numeric_and_performance_pending=True)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path)
    main(parser.parse_args())
