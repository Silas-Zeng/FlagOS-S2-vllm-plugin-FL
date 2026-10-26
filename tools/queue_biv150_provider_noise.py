"""CPU-only waiter for an owned completed control, then a single A/A service."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback

def identity(pid):
    try:
        x=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
        return dict(pid=pid,start_ticks=int(x[19]),sid=int(x[3]),state=x[0])
    except FileNotFoundError:return None

def live(x):
    y=identity(x['pid']);return bool(y and y['start_ticks']==x['start_ticks'] and y['state']!='Z')

def main(a):
    with a.state.open('x') as f:json.dump(dict(owner=identity(os.getpid()),phase='created'),f)
    state=dict(owner=identity(os.getpid()),tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def save(phase,**values):
        state.update(phase=phase,updated=datetime.datetime.now(datetime.timezone.utc).isoformat(),**values);p=a.state.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2)+'\n');p.replace(a.state)
    try:
        owners=[dict(pid=p,start_ticks=t) for p,t in a.upstream];assert owners and all(live(x) for x in owners)
        assert not a.out.exists() and not a.log.exists()
        save('waiting_control',upstream=owners)
        while any(live(x) for x in owners):time.sleep(5)
        q=json.loads(a.control_queue.read_text());s=json.loads((a.control_out/'state.json').read_text())
        assert q['phase']=='completed' and s['complete'] and s['cleanup_complete'],(q,s)
        assert s['commit']==a.commit
        with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',9031))!=0
        assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
        cmd=['/usr/local/bin/python',str(a.tool),'--seed',str(a.seed),'--commit',a.commit,'--out',str(a.out),'--fixtures',str(a.fixtures),'--repeats','3']
        save('starting_noise',command=cmd)
        with a.log.open('x') as log:child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save('noise',child=identity(child.pid));code=child.wait();save('noise_finished',returncode=code)
        assert code==0,code
        result=json.loads((a.out/'state.json').read_text());assert result['complete'] and result['cleanup_complete']
        save('completed',noise_summary=json.loads((a.out/'noise-summary.json').read_text()))
    except BaseException as e:save('failed',error=repr(e),traceback=traceback.format_exc());raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['state','log','out','control_queue','control_out','seed','tool','fixtures']:p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--upstream',type=int,nargs=2,action='append',required=True)
    main(p.parse_args())
