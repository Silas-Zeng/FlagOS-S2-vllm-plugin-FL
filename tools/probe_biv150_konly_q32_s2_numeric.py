#!/usr/bin/env python3
"""Default stdlib plan; prepare actual N1 caches or run Root-only numeric/Graph/side."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import biv150_konly_q32_s2_numeric_source as source
import biv150_konly_q32_s2_numeric_contract as contract


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',action='store_true');p.add_argument('--gpu',action='store_true')
    p.add_argument('--s1-raw',type=Path);p.add_argument('--s2-raw',type=Path);p.add_argument('--static-review',type=Path)
    p.add_argument('--preflight',type=Path);p.add_argument('--preflight-sha256')
    p.add_argument('--admission',type=Path);p.add_argument('--admission-sha256')
    p.add_argument('--root-owned-exclusive',action='store_true');p.add_argument('--numeric-diagnostic',action='store_true')
    p.add_argument('--commit');p.add_argument('--out',type=Path);a=p.parse_args()
    source.need(not(a.prepare and a.gpu),'choose CPU prepare or explicit Root GPU')
    if a.prepare:
        source.need(all((a.s1_raw,a.s2_raw,a.static_review,a.out)),'both actual raw archives/static review/new JSON output')
        source.need(not any((a.preflight,a.preflight_sha256,a.admission,a.admission_sha256,a.root_owned_exclusive,a.numeric_diagnostic,a.commit)),'prepare rejects device options')
        source.need(a.out.is_absolute() and not a.out.exists(),'new absolute preflight JSON')
        value=contract.preflight(a.s1_raw,a.s2_raw,a.static_review)
        a.out.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
        value=dict(preflight=str(a.out),sha256=source.sha(a.out.read_bytes()),
            provenance=value['provenance'],hashes=contract.HASHES,cubins=contract.CUBINS,
            complete_typed_ABI_equal=True,Root_numeric_admission=False,GPU_executed=False)
    elif a.gpu:
        source.need(not any((a.s1_raw,a.s2_raw,a.static_review)),'GPU uses separately hashed CPU-prepared proof')
        source.need(all((a.preflight,a.preflight_sha256,a.admission,a.admission_sha256,a.commit,a.out,a.root_owned_exclusive)),'Root-exclusive exact source/preflight/admission/new output required')
        data=a.preflight.read_bytes();source.need(source.sha(data)==a.preflight_sha256,'prepared proof SHA')
        value=source.run(a,json.loads(data))
    else:
        source.need(not any(v for k,v in vars(a).items() if k not in ('prepare','gpu')),'options require explicit mode')
        value=contract.plan()
    source.need(not {'torch','triton','numpy','vllm','vllm_fl'}.intersection(sys.modules) or a.gpu,'CPU path imported framework')
    print(json.dumps(value,sort_keys=True,indent=2,allow_nan=False))


if __name__=='__main__':main()
