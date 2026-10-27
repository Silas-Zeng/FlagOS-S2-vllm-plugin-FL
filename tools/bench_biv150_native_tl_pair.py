#!/usr/bin/env python3
"""Default stdlib plan. Device implementation is an explicit Root-only probe."""
import argparse
import json
from biv150_native_tl_pair_contract import plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-device',action='store_true')
    a=p.parse_args()
    if a.run_device:raise RuntimeError('Device runner pending; this initial Draft is CPU plan only')
    print(json.dumps(plan(),indent=2))


if __name__=='__main__':main()
