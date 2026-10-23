"""Capture compiled IR/assembly identities for the actual BF16 decode kernel."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import socket
import torch
from biv150_benchmark_utils import load_module, inputs, main_call


def main(args):
    with socket.socket() as guard:
        assert guard.connect_ex(('127.0.0.1',9031)) != 0, 'Service running'
    args.out.mkdir(parents=True,exist_ok=False)
    source=load_module(args.source,'gqa_lowering_inspect')
    c=inputs(32,16384)
    kernel=main_call(source,c,source._pick_segments(32,2,16))
    torch.cuda.synchronize()
    summary=dict(source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                 n_regs=kernel.n_regs,n_spills=kernel.n_spills,metadata=str(kernel.metadata),asm_keys=list(kernel.asm))
    for kind in ['ttir','ttgir','llir','ptx','s','sass','asm']:
        if kind not in kernel.asm:continue
        value=kernel.asm[kind]
        if not isinstance(value,str):continue
        (args.out/('kernel.'+kind)).write_text(value)
        matches=[line.strip() for line in value.splitlines() if re.search('mma|tcu|bf16|bfloat|cvt|convert_layout|dot|ld.global|load ',line,re.I)]
        (args.out/(kind+'-relevant.txt')).write_text('\n'.join(matches))
        summary[kind]=dict(bytes=len(value),relevant_lines=len(matches))
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
