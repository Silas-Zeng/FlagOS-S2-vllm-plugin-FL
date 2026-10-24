"""Finite CPU-only observation of one existing owned acceptance, no service changes."""
import datetime,json,os,time,urllib.request
from pathlib import Path
engine, driver = 41838,41692
root=Path('/workspace/experiments/minicpm-s2/gate-retained-bf16-20261024')
out=root/'phase-metrics-20261024.jsonl'
def identity(pid):
 try:
  fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
  return fields[0],fields[19]
 except FileNotFoundError:return None
pin=identity(engine)
assert pin and pin[0]!='Z'
keys=('num_preemptions_total','num_requests_running','num_requests_waiting','kv_cache_usage_perc','prompt_tokens_total','generation_tokens_total')
with out.open('x') as handle:
 handle.write(json.dumps(dict(collector_pid=os.getpid(),engine_pid=engine,engine_start_ticks=pin[1],driver_pid=driver,interval_seconds=15,max_seconds=2400,kind='diagnostic_existing_metrics'))+'\n');handle.flush()
 deadline=time.monotonic()+2400
 while time.monotonic()<deadline:
  current=identity(engine)
  if current is None or current[0]=='Z' or current[1]!=pin[1]:break
  row=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),monotonic=time.monotonic(),phase=None)
  try:
   children=Path(f'/proc/{driver}/task/{driver}/children').read_text().split()
   for child in children:
    try:cmd=Path(f'/proc/{child}/cmdline').read_bytes().decode().split('\0')
    except FileNotFoundError:continue
    if '--label' in cmd:row['phase']=cmd[cmd.index('--label')+1]
   raw=urllib.request.urlopen('http://127.0.0.1:9031/metrics',timeout=5).read().decode()
   row['metrics']=[line for line in raw.splitlines() if not line.startswith('#') and any(key in line for key in keys)]
  except Exception as error:row['error']=repr(error)
  handle.write(json.dumps(row)+'\n');handle.flush()
  time.sleep(15)
 handle.write(json.dumps(dict(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n');handle.flush()
