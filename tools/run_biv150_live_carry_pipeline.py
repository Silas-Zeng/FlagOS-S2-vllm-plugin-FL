"""Owned sequential CPU/device/full-model acceptance; no shared installation."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time


def idle():
    deadline=time.monotonic()+90
    while True:
        with socket.socket() as port:free=port.connect_ex(('127.0.0.1',9031))!=0
        gpu='No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
        if free and gpu:return
        if time.monotonic()>=deadline:raise RuntimeError('GPU/port busy; do not interrupt another job')
        time.sleep(2)


def main(args):
    source=args.source.resolve()
    assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
    idle();args.out.mkdir(parents=True,exist_ok=False)
    for name in ('native_w8a8.cpp','native_w8a8.cu'):
        assert (source/'csrc/iluvatar'/name).read_bytes()==(args.library_source/'csrc/iluvatar'/name).read_bytes()
    assert hashlib.sha256(args.library.read_bytes()).hexdigest()==args.library_sha256
    build=source/'build-native-w8a8';build.mkdir(exist_ok=False)
    target=build/'biv150_native_w8a8.so';shutil.copyfile(args.library,target)
    assert hashlib.sha256(target.read_bytes()).hexdigest()==args.library_sha256
    env=dict(os.environ,PYTHONPATH=str(source),PYTHONUNBUFFERED='1',MAX_JOBS='2')
    steps=[('cpu',[sys.executable,str(source/'tools/check_biv150_live_carry_cpu.py')]),
        ('row-guard-cpu',[sys.executable,str(source/'tools/check_biv150_row_guard_cpu.py'),'--out',str(args.out/'row-guard-cpu.json')]),
        ('gate9',[sys.executable,str(source/'tools/check_biv150_gate_auto.py'),'--library',str(target),'--out',str(args.out/'gate9.json')]),
        ('cached40',[sys.executable,str(source/'tools/check_biv150_quant_cached_dispatch.py'),'--library',str(target),'--out',str(args.out/'cached40.json')]),
        ('device',[sys.executable,'-u',str(source/'tools/check_biv150_live_carry_device.py'),'--library',str(target),'--out',str(args.out/'device')]),
        ('acceptance',[sys.executable,'-u',str(source/'tools/run_biv150_gate_team_acceptance.py'),'--source',str(source),'--out',str(args.out/'full'),'--mode','all'])]
    state=dict(driver_pid=os.getpid(),source=str(source),commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        library_sha256=args.library_sha256,library_reused_from=str(args.library),
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if args.native_checks_from:
        prior=json.loads((args.native_checks_from/'state.json').read_text())
        old=Path(prior['source'])
        assert prior['library_sha256']==args.library_sha256
        assert subprocess.check_output(['git','-C',str(old),'rev-parse','HEAD'],text=True).strip()==prior['commit']
        for file in ('csrc/iluvatar/native_w8a8.cpp','csrc/iluvatar/native_w8a8.cu',
            'tools/check_biv150_gate_auto.py','tools/check_biv150_quant_cached_dispatch.py',
            'vllm_fl/patches/biv150_native_w8a8.py'):
            assert (source/file).read_bytes()==(old/file).read_bytes(),file
        for name,count in (('gate9',9),('cached40',40)):
            report=json.loads((args.native_checks_from/(name+'.json')).read_text())
            assert report['library_sha256']==args.library_sha256 and len(report['records'])==count
            assert all(row.get('passed',row.get('bit_exact')) for row in report['records'])
            if name=='gate9':assert report['original_weight_fallback_unchanged']
            else:assert all(row['changed_input_graph'] for row in report['records'])
            shutil.copyfile(args.native_checks_from/(name+'.json'),args.out/(name+'-reused.json'))
        state['native_checks_reused_from']=dict(out=str(args.native_checks_from),source_commit=prior['commit'])
        steps=[item for item in steps if item[0] not in ('gate9','cached40')]
    for stage,command in steps:
        if stage not in ('cpu','row-guard-cpu'):idle()
        stage_env=dict(env,TORCHINDUCTOR_CACHE_DIR=str(args.out/(stage+'-inductor-cache')))
        with (args.out/(stage+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=source,env=stage_env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            state.update(stage=stage,pid=child.pid,command=command)
            (args.out/'state.json').write_text(json.dumps(state,indent=2))
            print(json.dumps(state),flush=True);code=child.wait()
        state['exit_code']=code;(args.out/'state.json').write_text(json.dumps(state,indent=2))
        if code:raise SystemExit(code)
        state.pop('exit_code')
    state['completed']=True;(args.out/'state.json').write_text(json.dumps(state,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--library',type=Path,required=True)
    p.add_argument('--library-source',type=Path,required=True);p.add_argument('--library-sha256',required=True)
    p.add_argument('--native-checks-from',type=Path)
    main(p.parse_args())
