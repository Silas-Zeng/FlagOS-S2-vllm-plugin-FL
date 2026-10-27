"""One eager numeric job: empty native PID→owned host PID; CPU imports only."""
import hashlib
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess

QUERY=['/usr/local/corex/bin/ixsmi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits']
PHASES=('compile','encode0','baseline0','unhoisted0','hoisted0','encode1','baseline1','unhoisted1','hoisted1','complete')


def query():
    result=subprocess.run(QUERY,capture_output=True,text=True,timeout=15,check=True)
    if result.stderr.strip(): raise ValueError('Native PID query stderr')
    rows=result.stdout.strip().splitlines() if result.stdout.strip() else []
    if any(not re.fullmatch(r'[1-9][0-9]*',s.strip(),re.ASCII) for s in rows): raise ValueError('Unknown native PID output')
    return [int(s.strip()) for s in rows]


def identity():
    raw=Path('/proc/self/stat').read_text();pid=os.getpid()
    if int(raw.split(' ',1)[0])!=pid: raise ValueError('Unknown container identity')
    return dict(pid=pid,start_ticks=int(raw[raw.rfind(')')+1:].split()[19]))


def closed():
    with socket.socket() as sock: return sock.connect_ex(('127.0.0.1',9031))!=0


def initialize():
    import torch
    tiny=torch.zeros(1,device='cuda');torch.cuda.synchronize();return tiny


class Admission:
    def __init__(self,preflight,binding,*,query=query,identity=identity,closed=closed,initialize=initialize):
        self.preflight,self.binding=preflight,binding
        self.query,self.identity,self.closed,self.initialize=query,identity,closed,initialize
        self.state='new';self.owner=None;self.roots=None;self.native=None;self.token=None;self.tiny=None
        self.phases=[];self.errors=[];self.initialization_attempted=False
    def begin(self):
        if self.state!='new': raise ValueError('Admission reuse')
        self.preflight()
        if not self.closed() or self.query()!=[]: raise ValueError('Initial GPU/service must be idle')
        self.owner=self.identity();self.roots=self.binding();self.token=secrets.token_hex(16)
        self.initialization_attempted=True;self.tiny=self.initialize();pids=self.query()
        if self.tiny is None or self.identity()!=self.owner or self.binding()!=self.roots:
            raise ValueError('Fresh initialization/identity binding failed')
        if type(pids) is not list or len(pids)!=1 or type(pids[0]) is not int or pids[0]<=0:
            raise ValueError('Fresh native host PID must be one known positive integer')
        self.native=pids[0];self.state='active';return self.token
    def check(self,phase,token):
        if self.state!='active' or token!=self.token or len(self.phases)>=len(PHASES) or PHASES[len(self.phases)]!=phase:
            raise ValueError('Missing/stale token or unknown phase order')
        if self.identity()!=self.owner or self.binding()!=self.roots or not self.closed(): raise ValueError('Owner/root/service changed')
        if self.query()!=[self.native]: raise ValueError('Native host PID foreign/missing/changed; never infer from container PID')
        self.phases.append(dict(phase=phase,native_host_pid=self.native))
        if phase=='complete': self.state='completed'
    def receipt(self):
        return dict(state=self.state,query=QUERY,container_identity=self.owner,native_host_pid=self.native,
                    root_identity=self.roots,token_sha256=hashlib.sha256(self.token.encode()).hexdigest() if self.token else None,
                    initialization_attempted=self.initialization_attempted,
                    initialization='one torch.zeros(1,cuda)+synchronize, outside numerical checks; GPU work is not zero',
                    phases=self.phases,errors=self.errors)
