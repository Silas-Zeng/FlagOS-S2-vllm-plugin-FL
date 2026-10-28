"""Launch one frozen component on the explicitly authorized personal instance."""
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

spec = json.loads(Path(sys.argv[1]).read_text())
repo = Path(spec["repo"])
tag = spec["tag"]
assert re.fullmatch(r"cjj-[a-z0-9-]+", tag)
assert subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                               text=True).strip() == spec["commit"]
assert not subprocess.check_output(["git", "-C", str(repo), "status",
                                    "--porcelain", "--untracked-files=no"], text=True)
subprocess.run(["git", "-C", str(repo), "diff", "--exit-code",
                spec["base"], "--", "vllm_fl", "csrc"], check=True)
bundle = Path("/private/tmp") / (tag + ".bundle")
subprocess.run(["git", "-C", str(repo), "bundle", "create", str(bundle),
                spec.get("bundle_base", spec["base"]) + "..HEAD"], check=True)
key = "/Users/chenjunjie/.ssh/flagos_minicpm_personal_20261027_v2"
host = "lab_inst_01M5FK23N818VZHR7F0QSHBEMR@ssh.flagos.io"
options = ["-i", key, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
           "-o", "ConnectTimeout=15"]
subprocess.run(["scp", *options, "-P", "2224", str(bundle),
                host + ":/workspace/experiments/minicpm-s2/exports/" + bundle.name],
               check=True, timeout=60)
remote = r'''
from pathlib import Path
import subprocess,os,socket,json,datetime,hashlib
c=CONFIG
r=Path('/workspace/experiments/minicpm-s2')
base=r/'code/main-w16-qualified-20261027'
b=r/'exports'/(c['tag']+'.bundle'); source=r/'code'/c['tag']
assert not source.exists()
assert hashlib.sha256(b.read_bytes()).hexdigest()==c['bundle_sha256']
subprocess.run(['git','-C',str(base),'bundle','verify',str(b)],check=True)
subprocess.run(['git','-C',str(base),'bundle','unbundle',str(b)],check=True)
subprocess.run(['git','-C',str(base),'worktree','add','--detach',str(source),c['commit']],check=True)
subprocess.run(['git','-C',str(source),'diff','--exit-code',c['base'],'--','vllm_fl','csrc'],check=True)
prefix=c['tag'];out=r/prefix
log=r/'exports'/(prefix+'.log');owner=r/'exports'/(prefix+'-owner.json')
assert not out.exists() and not log.exists() and not owner.exists()
env=dict(os.environ,PYTHONPATH=str(source),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',
         TRITON_CACHE_DIR=str(r/(prefix+'-triton')),FLAGGEMS_CACHE_DIR=str(r/(prefix+'-flaggems')))
if c.get('cpu_script'):
 subprocess.run(['/usr/local/bin/python',str(source/c['cpu_script']),*c.get('cpu_args',[])],
                cwd=source,env=env,check=True)
report=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
if 'No running processes found' not in report:raise RuntimeError('GPU busy; nothing stopped')
with socket.socket() as s:
 if s.connect_ex(('127.0.0.1',9031))==0:raise RuntimeError('Existing service; nothing stopped')
args=[a.replace('{SOURCE}',str(source)).replace('{OUT}',str(out)) for a in c['args']]
cmd=['/usr/local/bin/python',str(source/c['script']),*args]
with log.open('xb') as f:p=subprocess.Popen(cmd,cwd=source,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
fields=Path(f'/proc/{p.pid}/stat').read_text().split(') ',1)[1].split()
v=dict(pid=p.pid,start_ticks=int(fields[19]),sid=os.getsid(p.pid),pgid=os.getpgid(p.pid),
       created=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=c['commit'],
       draft_pr=c['pr'],command=cmd,out=str(out),log=str(log),official_score=False,
       bundle_sha256=c['bundle_sha256'],launcher_sha256=c['launcher_sha256'])
owner.write_text(json.dumps(v,indent=2)+'\n');print('LAUNCHED',json.dumps(v),flush=True)
'''
spec["bundle_sha256"] = hashlib.sha256(bundle.read_bytes()).hexdigest()
spec["launcher_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
remote = remote.replace("CONFIG", repr(spec), 1)
subprocess.run(["ssh", *options, "-p", "2224", host,
                "python -c " + shlex.quote(remote)], check=True, timeout=60)
