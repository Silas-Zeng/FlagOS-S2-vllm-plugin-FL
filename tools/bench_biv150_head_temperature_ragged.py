"""Default stdlib plan; Root-only four-row continuation of the real old ten gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
ROWS=(31,41,42,43)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def plan():
    return dict(gpu_executed=False,baseline='actual F.linear + unchanged complete original Sampler',
        candidate='unchanged BF16 head kernel + original PreScaledSampler temperature ticket',
        rows={str(r):dict(BLOCK_M=32 if r==31 else 64,BLOCK_N=64 if r==31 else 256,
                          BLOCK_K=64 if r==31 else 32,num_warps=8 if r==31 else 16,
                          num_stages=1 if r==31 else 2,grid=[2040 if r==31 else 510,1]) for r in ROWS},
        numerical='old ten gates plus row-varying positive temperature; FP64 sample .001/.01 upgraded to all real rows x five original columns',
        tail_guards='full storage CPU authority SHA/boundary snapshots; no GPU negative/advanced canary indexing',
        timing='one warmed eager ABBA, 8 complete head+Sampler calls per arm; original timed_batch byte-identical; production-sized output allocations in timing',
        prerequisites=['Root exclusive acknowledgement','source exact commit/clean','port9031 closed','free memory>=30000MiB and visible existing process refusal','private copied exact tuning snapshot','all actual SDK source pins','actual baseline configuration matches one stated original tile'],
        pending=['fresh IR/hints/resources independent review','new-row numerical/sampling/RNG independent review','service','105','throughput'],
        historical_qualification_transferred=False)

def admit(root_ack,port_open,free_mib,visible_pids):
    if not root_ack:raise ValueError('Root-owned exclusive acknowledgement required')
    if port_open:raise ValueError('port9031 occupied')
    if type(free_mib) is not int or free_mib<30000:raise ValueError('insufficient/unknown free memory')
    if visible_pids:raise ValueError('visible pre-existing GPU process')
    return dict(root_owned_exclusive=True,port9031='closed',free_mib=free_mib,
                existing_visible_pids=[],process_visibility='no visible PID; visibility unavailable, not idle proof')

def preflight(args):
    if not args.root_owned_exclusive:raise ValueError('Root-owned exclusive acknowledgement required')
    port_open=False
    for family,host in ((socket.AF_INET,'127.0.0.1'),(socket.AF_INET6,'::1')):
        try:
            with socket.socket(family) as s:s.settimeout(1);port_open|=s.connect_ex((host,9031))==0
        except OSError:
            if family==socket.AF_INET:raise
    ix=shutil.which('ixsmi') or '/usr/local/corex/bin/ixsmi'
    full=subprocess.check_output([ix],text=True,timeout=15)
    free=subprocess.check_output([ix,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=15).strip()
    if not free.isdigit():raise ValueError('one numeric device free-memory record required')
    pids=[int(x) for x in re.findall(r'^\s*\|\s*\d+\s+(?:N/A\s+N/A\s+)?(\d+)\s+(?:C\+G|C|G)\s+',full,re.M)]
    receipt=admit(args.root_owned_exclusive,port_open,int(free),pids)
    receipt['ixsmi_observation']=full
    return receipt

def source_check(commit):
    actual=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if actual!=commit:raise ValueError('source commit mismatch')
    if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=no'],text=True).strip():raise ValueError('tracked source is modified')
    files=subprocess.check_output(['git','-C',str(ROOT),'ls-files','-z']).decode().split('\0')
    return {f:sha(ROOT/f) for f in files if f and (ROOT/f).is_file() and not (ROOT/f).is_symlink()}

def copy_tuning(source,target):
    manifest=json.loads((source/'manifest.json').read_text())
    records=manifest['databases']
    assert records and len({r['name'] for r in records})==len(records)
    target.mkdir(parents=True)
    for r in records:
        assert Path(r['name']).name==r['name'] and r['name'].endswith('.db')
        p=source/r['name'];assert sha(p)==r['sha256']
        shutil.copyfile(p,target/r['name']);assert sha(target/r['name'])==r['sha256']
    return records

def run(a):
    source=source_check(a.commit)
    admission=preflight(a)
    a.out.mkdir(parents=True,exist_ok=False)
    result=dict(plan(),completed=False,errors=[],source_commit=a.commit,source_sha256=source,
                admission=admission,cases=[],model_or_official_qualified=False)
    def save():(a.out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    save()
    try:
        for rows in a.row or ROWS:
            d=a.out/('rows'+str(rows));d.mkdir();(d/'source').mkdir()
            tuning=copy_tuning(a.tuning_snapshot,d/'flaggems-cache/config_cache')
            env={k:v for k,v in os.environ.items() if not k.startswith(('VLLM_FL_BIV150_','BIV150_','MINICPM_NATIVE_','MINICPM_DECODE_GAP_','CJJ_NATIVE_GRAPH'))}
            env.update(PYTHONDONTWRITEBYTECODE='1',VLLM_PLUGINS='fl',PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'tools'),
                TRITON_CACHE_DIR=str(d/'triton-cache'),FLAGGEMS_CACHE_DIR=str(d/'flaggems-cache'),TORCHINDUCTOR_CACHE_DIR=str(d/'inductor-cache'),
                VLLM_FL_BIV150_HEAD_TEMPERATURE='1',VLLM_FL_BIV150_HEAD_TEMPERATURE_RAGGED='1',
                VLLM_FL_BIV150_NATIVE_W8A8='0',VLLM_FL_BIV150_NATIVE_GQA='0')
            command=[sys.executable,str(Path(__file__).resolve()),'--worker','--run-gpu','--root-owned-exclusive','--commit',a.commit,'--row',str(rows),'--out',str(d),'--model',str(a.model)]
            with (d/'worker.log').open('x') as log:
                p=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
            raw=json.loads((d/'result.json').read_text()) if (d/'result.json').exists() else None
            result['gpu_executed']=result['gpu_executed'] or bool(raw and raw.get('gpu_executed'))
            case=dict(rows=rows,exit_code=p.returncode,tuning_snapshot=tuning,result=raw);result['cases'].append(case);save()
            if p.returncode or not raw or not raw.get('requested_component_gates_passed'):raise ValueError('component failed; do not continue remaining rows')
        assert source_check(a.commit)==source
        result['completed']=True;result['gpu_executed']=True
    except BaseException as e:
        result['errors'].append(repr(e));save();raise
    finally:save()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-gpu',action='store_true');p.add_argument('--root-owned-exclusive',action='store_true');p.add_argument('--commit');p.add_argument('--out',type=Path);p.add_argument('--model',type=Path,default=Path('/workspace/MiniCPM5-2B'));p.add_argument('--tuning-snapshot',type=Path);p.add_argument('--row',type=int,choices=ROWS,action='append');p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
    if a.worker:
        if not all((a.run_gpu,a.root_owned_exclusive,a.commit,a.out,a.row)) or len(a.row)!=1:p.error('internal worker still requires explicit GPU/Root/source admission')
        source_check(a.commit)
        preflight(a)
        from biv150_head_temperature_ragged_worker import worker
        raise SystemExit(worker(a.out,a.model,a.row[0]))
    if not a.run_gpu:
        if any((a.root_owned_exclusive,a.commit,a.out,a.tuning_snapshot,a.row)):p.error('device arguments require --run-gpu')
        print(json.dumps(plan(),ensure_ascii=False,sort_keys=True,indent=2));return
    if not all((a.root_owned_exclusive,a.commit,a.out,a.tuning_snapshot)):p.error('--run-gpu requires Root acknowledgement, exact commit, new out and tuning snapshot')
    run(a)

if __name__=='__main__':main()
