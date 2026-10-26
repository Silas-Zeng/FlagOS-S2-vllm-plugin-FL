"""Audit live Q32 off/on closure compatibility; no model or score is executed."""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess


def main(args):
    if not args.run_gpu:
        print('Prepared audit: strict source pins and actual mixed/GQA/normalized/Q32 closure identity; no service acceptance claim.')
        return
    with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',9031))!=0
    assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
    import torch
    from vllm_fl.patches import biv150_compute_order as order
    from vllm_fl.patches import biv150_decode_launch_runtime as launch
    from vllm_fl.patches import biv150_mixed_decode as mixed
    import biv150_prefill_probe_fixtures as fixture
    states={'launch':launch.install(),'mixed':mixed.install()}
    _,_,extra=fixture.install_baseline();states.update(extra)
    assert all(value['installed'] for value in states.values()),states
    order._main_stack(torch)
    os.environ['VLLM_FL_BIV150_PREFILL_Q32']='1'
    module=importlib.import_module('vllm_fl.patches.biv150_prefill_q32')
    installed=module.install();assert installed['installed'],installed
    order._main_stack(torch)
    pins={}
    for name,expected in order.SOURCES.items():
        spec=importlib.util.find_spec(name);assert spec and spec.origin,name
        actual=hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
        assert actual==expected,name
        pins[name]=actual
    root=Path(__file__).resolve().parents[1]
    assert hashlib.sha256((root/'vllm_fl/worker/model_runner.py').read_bytes()).hexdigest()==order.EXPECTED_RUNNER
    record=dict(passed=True,off_stack=True,on_stack=True,source_pins=pins,
                source_commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
                limits='Explicit live-module/closure audit only; no Worker cold-plan, numerical, concurrency or throughput acceptance.')
    args.out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-gpu',action='store_true');p.add_argument('--out',type=Path)
    args=p.parse_args()
    if args.run_gpu and (args.out is None or args.out.exists()):p.error('Need a new --out file')
    main(args)
