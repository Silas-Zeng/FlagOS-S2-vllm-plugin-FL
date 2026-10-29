#!/usr/bin/env python3
"""Default stdlib plan; explicit CPU preparation or Root-only numeric screen."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import biv150_konly_q32_w8_numeric as adapter

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',action='store_true');p.add_argument('--gpu',action='store_true')
    p.add_argument('--control-raw',type=Path);p.add_argument('--candidate-raw',type=Path);p.add_argument('--static-review',type=Path)
    p.add_argument('--preflight',type=Path);p.add_argument('--preflight-sha256')
    p.add_argument('--admission',type=Path);p.add_argument('--admission-sha256')
    p.add_argument('--root-owned-exclusive',action='store_true');p.add_argument('--numeric-diagnostic',action='store_true')
    p.add_argument('--commit');p.add_argument('--out',type=Path);a=p.parse_args()
    adapter.need(not(a.prepare and a.gpu),'choose CPU preparation or explicit Root GPU')
    f=adapter.framework();source=f.source;contract=f.contract
    if a.prepare:
        adapter.need(all((a.control_raw,a.candidate_raw,a.static_review,a.out)),'both actual raw archives/static review/new JSON output')
        adapter.need(not any((a.preflight,a.preflight_sha256,a.admission,a.admission_sha256,a.root_owned_exclusive,a.numeric_diagnostic,a.commit)),'CPU prepare rejects device options')
        adapter.need(a.out.is_absolute() and not a.out.exists(),'new absolute preflight JSON')
        value=contract.preflight(a.control_raw,a.candidate_raw,a.static_review)
        a.out.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
        value=dict(preflight=str(a.out),sha256=adapter.sha(a.out.read_bytes()),provenance=value['provenance'],
            hashes=contract.HASHES,cubins=contract.CUBINS,complete_typed_ABI_equal=True,Root_numeric_admission=False,GPU_executed=False)
    elif a.gpu:
        adapter.need(not any((a.control_raw,a.candidate_raw,a.static_review)),'GPU consumes hashed CPU-prepared proof')
        adapter.need(all((a.preflight,a.preflight_sha256,a.admission,a.admission_sha256,a.commit,a.out,a.root_owned_exclusive)),'Root-exclusive exact source/preflight/admission/new output required')
        data=a.preflight.read_bytes();adapter.need(adapter.sha(data)==a.preflight_sha256,'prepared proof SHA')
        value=source.run(a,json.loads(data))
    else:
        adapter.need(not any(v for k,v in vars(a).items()if k not in ('prepare','gpu')),'options require explicit mode')
        value=contract.plan()
    adapter.need(not {'torch','triton','numpy','vllm','vllm_fl'}.intersection(sys.modules) or a.gpu,'CPU path imported framework')
    print(json.dumps(value,sort_keys=True,indent=2,allow_nan=False))

if __name__=='__main__':main()
