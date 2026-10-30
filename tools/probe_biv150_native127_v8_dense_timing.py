#!/usr/bin/env python3
"""Default CPU plan; explicit Root-only one-case dense reader ABBA."""
import argparse
import json
from pathlib import Path
import sys
import biv150_native127_v8_dense_timing as dense


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('gpu','timing','root-owned-exclusive','root-timing-approved'):p.add_argument('--'+name,action='store_true')
    for name in ('parent-source','compile-summary','static-review','control-library','library','codec-summary',
      'numeric-summary','numeric-review','timing-admission','dense-admission','out'):p.add_argument('--'+name,type=Path)
    for name in ('static-review-sha256','codec-proof-sha256','numeric-review-sha256','timing-admission-sha256','dense-admission-sha256'):p.add_argument('--'+name)
    a=p.parse_args()
    if a.gpu:
        if not all(v for k,v in vars(a).items() if k!='gpu'):p.error('Root timing/serial flags plus all original evidence and NEW dense admission required')
        return dense.execute(a)
    if any(v for k,v in vars(a).items() if k!='gpu'):p.error('Device arguments require --gpu')
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    print(json.dumps(dense.plan(),indent=2,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
