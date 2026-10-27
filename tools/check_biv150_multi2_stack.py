"""Opt-in live SDK/C3 closure audit; no model or score, schedule after scope gate."""
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
        print('Prepared actual SDK/C3 multi2 off/on closure audit; not executed.')
        return
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',9031))==0:raise RuntimeError('Service active')
    if 'No running processes found' not in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True):
        raise RuntimeError('GPU busy')
    os.environ['VLLM_FL_BIV150_PREFILL_MULTI2']='0'
    import torch
    package=importlib.import_module('vllm_fl')
    root=Path(__file__).resolve().parents[1]
    assert Path(package.__file__).resolve()==root/'vllm_fl/__init__.py','Loaded package is outside this source'
    import biv150_prefill_probe_fixtures as fixture
    from vllm_fl.patches import biv150_decode_launch_runtime as launch
    from vllm_fl.patches import biv150_mixed_decode as mixed
    from vllm_fl.patches import biv150_compute_order as order
    states={'launch':launch.install(),'mixed':mixed.install()}
    _,base,extra=fixture.install_baseline();states.update(extra)
    q32=importlib.import_module('vllm_fl.patches.biv150_prefill_q32')
    states['q32']=q32.install()
    assert all(s['installed'] for s in states.values()),states
    parent=base.holder[base.kernel_key]
    multi=importlib.import_module('vllm_fl.patches.biv150_prefill_multi2')
    os.environ['VLLM_FL_BIV150_PREFILL_MULTI2']='0'
    assert not multi.install()['installed'] and base.holder[base.kernel_key] is parent
    order._main_stack(torch)
    os.environ.pop('VLLM_FL_BIV150_PREFILL_MULTI2')
    states['multi2']=multi.install();assert states['multi2']['installed'],states
    assert multi._POLICY.base is parent and parent.manager is q32._MANAGER
    order._main_stack(torch)
    pins={}
    for name,expected in order.SOURCES.items():
        spec=importlib.util.find_spec(name);assert spec and spec.origin,name
        actual=hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
        assert actual==expected,name
        pins[name]=actual
    assert hashlib.sha256((root/'vllm_fl/worker/model_runner.py').read_bytes()).hexdigest()==order.EXPECTED_RUNNER
    result=dict(passed=True,precision_gate=False,off_stack=True,default_on_stack=True,
        parent_chain_preserved=True,installations=states,source_pins=pins,
        source_commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        limitations='Actual imported SDK/local source and C3 closures only; no Worker cold plan, kernel numerics, resources, model score or throughput proof.')
    args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-gpu',action='store_true');p.add_argument('--out',type=Path)
    args=p.parse_args()
    if args.run_gpu and (args.out is None or args.out.exists()):p.error('A new --out file is required')
    main(args)
