#!/usr/bin/env python3
"""CPU-only review of fresh service Q32 native artifacts; never imports a framework."""
import argparse,json
from pathlib import Path
import biv150_konly_service_q32_fresh_binding as binding

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--compile-out',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    if a.compile_out:
        value=binding.load_compile(a.compile_out)
        result=dict(CPU_only=True,actual_fresh_native_audit_passed=True,audit=value['audit'],baseline_LLVM=value['baseline_LLVM'],
            compile_out=str(a.compile_out.resolve()),kernel_hashes={r:v['hash'] for r,v in value['records'].items()},
            actual_artifact_SHA256={r:{e:binding.h.sha(b) for e,b in d.items()} for r,d in value['actual'].items()},
            loaded_resources_verified=False,numerical_verified=False,precision_gate=False,service_verified=False)
    else:result=dict(CPU_only=True,enabled=False,actual_fresh_native_audit_passed=False,GPU_executed=False)
    text=json.dumps(result,sort_keys=True,indent=2)+'\n'
    if a.out:
        with a.out.open('x') as out:out.write(text)
    print(text,end='')
if __name__=='__main__':main()
