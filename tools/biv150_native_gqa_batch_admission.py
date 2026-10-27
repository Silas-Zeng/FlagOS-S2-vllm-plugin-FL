"""Conservative diagnostic refusals, never an ixsmi exclusivity proof.

Root's explicit acknowledgement means Root checked the prior owned job terminal
and schedules this job serially. Namespace-hidden processes remain unknowable.
"""
import os
from pathlib import Path
import re
import subprocess

PRE_CUDA_MAX_MIB=1024
RUNNING_MAX_MIB=4096
MAX_PHASE_GROWTH_MIB=1024
NO_VISIBLE=('No running processes','No running processes found','No running processes found.')


def need(ok,message):
    if not ok:raise RuntimeError('diagnostic admission: '+message)


def process_identity():
    return dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
                scope='container process; not a driver host-PID mapping')


def parse_observation(full_text,pid_text):
    memory=re.findall(r'(\d+)MiB\s*/\s*(\d+)MiB',full_text)
    need(len(memory)==1,'unknown or multi-device GPU memory observation')
    used,total=map(int,memory[0]);need(0<=used<=total and total>0,'invalid GPU memory observation')
    text=pid_text.strip()
    if not text or text in NO_VISIBLE:pids=[]
    else:
        lines=text.splitlines()
        need(all(s.strip().isdigit() and int(s.strip())>0 for s in lines),'unknown native PID query')
        pids=[int(s.strip()) for s in lines]
    full_pids=[]
    if 'Processes:' in full_text:
        full_pids=[int(x) for x in re.findall(r'(?m)^\|\s*0\s+([1-9][0-9]*)\s+',full_text.split('Processes:',1)[1])]
    all_pids=sorted(set(pids+full_pids))
    return dict(memory_used_mib=used,memory_total_mib=total,visible_native_pids=all_pids,
                native_query_pids=pids,full_table_native_pids=full_pids,
                native_pid_visibility='visible_unowned' if all_pids else 'unavailable',
                no_visible_pid_is_idle_proof=False)


def observe():
    full=subprocess.run(['/usr/local/corex/bin/ixsmi','-i','0'],capture_output=True,text=True,timeout=15,check=True)
    pids=subprocess.run(['/usr/local/corex/bin/ixsmi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                        capture_output=True,text=True,timeout=15,check=True)
    return parse_observation(full.stdout,pids.stdout)


class Admission:
    def __init__(self,root_owned_exclusive,*,reader=observe,identity_reader=process_identity):
        need(root_owned_exclusive is True,'Root explicit serial ownership acknowledgement required')
        self.reader,self.identity_reader=reader,identity_reader;self.owner=identity_reader();self.previous=None
        need(type(self.owner['pid']) is int and self.owner['pid']>0
             and type(self.owner['start_ticks']) is int and self.owner['start_ticks']>0,'real process identity required')
        self.report=dict(schema='root-serial-memory-observation-v2',root_owned_exclusive_acknowledged=True,
            exclusivity_authority='Root checked prior owned job terminal and dispatches serially; external assertion',
            exclusive_by_ixsmi=False,container_process_identity=self.owner,driver_host_pid=None,
            driver_host_pid_mapping_verified=False,
            limits=dict(pre_cuda_max_used_mib=PRE_CUDA_MAX_MIB,running_max_used_mib=RUNNING_MAX_MIB,
                        max_running_phase_growth_mib=MAX_PHASE_GROWTH_MIB),
            tiny_initialization_outside_numerics=True,observations=[])

    def check(self,phase='numeric-phase'):
        try:
            identity=self.identity_reader();state=self.reader()
            record=dict(phase=phase,container_process_identity=identity,**state)
            self.report['observations'].append(record)
            need(identity==self.owner,'container PID/start ticks changed')
            # No verified namespace-to-host mapping is available. A visible PID
            # is never automatically declared ours, even if numerically equal.
            need(not state['visible_native_pids'],'visible native PID has no verified ownership')
            limit=PRE_CUDA_MAX_MIB if phase=='pre-CUDA' else RUNNING_MAX_MIB
            need(state['memory_used_mib']<=limit,'GPU memory above conservative refusal limit')
            if self.previous is not None and phase not in ('pre-CUDA','post-tiny-initialization'):
                need(state['memory_used_mib']-self.previous<=MAX_PHASE_GROWTH_MIB,'unexpected GPU memory growth')
            self.previous=state['memory_used_mib'];record['refusal_checks_passed']=True
            return record
        except BaseException as error:
            self.report.setdefault('errors',[]).append(dict(phase=phase,type=type(error).__name__,message=str(error)))
            raise
