from pathlib import Path
import subprocess,sys,json,hashlib,socket,os,datetime,traceback
r=Path('/workspace/experiments/minicpm-s2')
source=r/'code/bf16-bulk-unsplit-repr-v1-20261025'
out=r/'bf16-bulk-unsplit-repr-v1-20261025'
state_path=r/'exports/bf16-bulk-unsplit-repr-v1-20261025.state.json'
commit='6a184b7a56e93a6f3865609281f0bebe097c0a1d'
state=dict(driver_pid=os.getpid(),source=str(source),commit=commit,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage='preflight')
def save():state_path.write_text(json.dumps(state,indent=2)+'\n')
save()
try:
 assert not out.exists()
 assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==commit
 assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
 with socket.socket() as s:assert s.connect_ex(('127.0.0.1',9031))!=0
 gpu=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
 assert 'No running processes found' in gpu
 cache=r/'cache/bf16-bulk-unsplit-repr-v1-20261025'
 env=os.environ.copy();env.update(PYTHONPATH=str(source),VLLM_PLUGINS='fl',TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'))
 command=['/usr/local/bin/python','-u','tools/bench_biv150_bf16_bulk_unsplit.py','--edge-gates','--out',str(out)]
 state.update(command=command,expected_records=136,preflight_gpu=gpu,stage='running')
 state['source_hashes']={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ['tools/bench_biv150_bf16_bulk_unsplit.py','tools/biv150_bf16_bulk_unsplit.py','vllm_fl/patches/biv150_native_w8a8.py']}
 with (r/'exports/bf16-bulk-unsplit-repr-v1-20261025.worker.log').open('x') as log:
  child=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
  state.update(pid=child.pid,start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21]);save()
  rc=child.wait()
 state.update(returncode=rc,stage='finished',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 result=out/'result.json'
 if result.exists():
  data=json.loads(result.read_text());state.update(summary=data.get('summary'),diagnostic_completed=data.get('diagnostic_completed',False))
 assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
 save()
except Exception as error:
 state.update(stage='failed',error=repr(error),traceback=traceback.format_exc());save();raise
