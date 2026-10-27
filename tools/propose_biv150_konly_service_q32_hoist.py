#!/usr/bin/env python3
"""CPU-default/explicit proposal writer; never compiles or launches an override."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
from biv150_konly_service_q32_hoist import propose,need


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--enable',action='store_true')
    p.add_argument('--ttgir',type=Path);p.add_argument('--metadata',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    if not a.enable:
        need(not any((a.ttgir,a.metadata,a.out)),'inputs belong to explicit --enable')
        print(json.dumps(propose(None,{})[1],sort_keys=True));return
    need(a.ttgir and a.metadata and a.out and not a.out.exists(),'actual input+metadata+fresh output required')
    meta=json.loads(a.metadata.read_text());meta['enabled']=True
    override,proof=propose(a.ttgir.read_text(),meta)
    a.out.mkdir(parents=True);(a.out/'q32-service-hoist.ttgir').write_text(override)
    (a.out/'proposal.json').write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(CPU_proposal_created=True,actual_override_compiled=False,actual_I8_hoist_verified=False)))


if __name__=='__main__':main()
