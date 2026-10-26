from pathlib import Path
import subprocess,json,time,socket,os,datetime
R=Path('/workspace/experiments/minicpm-s2');E=R/'exports';STATE=E/'small-out-queue-v2-state.json'
def identity(pid):
    try:
        v=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,start_ticks=int(v[19]),sid=int(v[3]),state=v[0])
    except FileNotFoundError:return None

def live(member):
    v=identity(member['pid']);return bool(v and v['state']!='Z' and v['start_ticks']==member['start_ticks'])
owner=identity(os.getpid());state=dict(queue_pid=owner['pid'],queue_start_ticks=owner['start_ticks'],previous=dict(pid=451118,start_ticks=330838738),source_commit='1d00e0d78e7bb5388094a0d62c4a7ffcf24cb2af')
def save(phase,**kw):
    state.update(phase=phase,updated=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw)
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(STATE)
try:
    save('waiting_exact_pr57_gate')
    while live(state['previous']):time.sleep(10)
    prior=json.loads((R/'small-projection-gate-only-v2-20261026/state.json').read_text())
    assert prior['phase'] in ('completed','failed') and prior['cleanup_complete'],prior
    while True:
        with socket.socket() as sock:closed=sock.connect_ex(('127.0.0.1',9031))!=0
        idle='No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
        if closed and idle:break
        save('waiting_idle_after_pr57');time.sleep(15)
    src=R/'code/small-out-lowering-v2-20261026'
    assert subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip()==state['source_commit']
    assert not subprocess.check_output(['git','-C',str(src),'status','--porcelain','--untracked-files=no'],text=True).strip()
    out=R/'small-out-lowering-component-v2-20261026';assert not out.exists()
    command=['/usr/local/bin/python',str(src/'tools/check_biv250_small_out_lowering_gpu.py'),'--run-gpu','--source',str(src),'--commit',state['source_commit'],'--out',str(out),'--tuning-snapshot',str(E/'small-bf16-fullwarm-tuning-snapshot-v6-20261025')]
    with (E/'small-out-component-v2.log').open('x') as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save('component',child=identity(child.pid),command=command,out=str(out))
    code=child.wait();save('completed' if code==0 else 'failed',returncode=code)
except BaseException as error:
    import traceback
    save('failed',error=repr(error),traceback=traceback.format_exc());raise
