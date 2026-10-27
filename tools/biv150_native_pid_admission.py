"""Whole-job empty→owned native-PID transition; importing is CPU-only."""
import copy
import hashlib
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess

QUERY=['/usr/local/corex/bin/ixsmi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits']


def parse_native_pids(stdout):
    assert type(stdout) is str, 'Unknown native PID output type'
    if not stdout.strip():return []
    lines=stdout.strip().splitlines()
    assert all(re.fullmatch(r'[1-9][0-9]*',line.strip(),re.ASCII) for line in lines), ('Unknown native PID output',stdout)
    return [int(line.strip()) for line in lines]


def query_native_pids():
    result=subprocess.run(QUERY,capture_output=True,text=True,timeout=10)
    assert result.returncode==0 and not result.stderr.strip(), ('Native PID query failed',result.returncode,result.stderr)
    return parse_native_pids(result.stdout)


def process_identity():
    pid=os.getpid();raw=Path('/proc/self/stat').read_text()
    assert int(raw.split(' ',1)[0])==pid
    # comm may contain spaces/parentheses; field22 is index19 after final ')'.
    start=int(raw[raw.rfind(')')+1:].split()[19])
    return dict(pid=pid,start_ticks=start)


def port_closed():
    with socket.socket() as sock:return sock.connect_ex(('127.0.0.1',9031))!=0


def initialize_tiny_cuda():
    import torch
    tensor=torch.zeros(1,device='cuda')
    torch.cuda.synchronize()
    return tensor  # Retain one live allocation until this whole job ends.


class NativeAdmission:
    def __init__(self,preflight,binding,*,query=query_native_pids,identity=process_identity,
                 closed=port_closed,initialize=initialize_tiny_cuda,nonce=lambda:secrets.token_hex(16)):
        self.preflight,self.binding=preflight,binding
        self.query,self.identity,self.closed,self.initialize,self.nonce=query,identity,closed,initialize,nonce
        self.state='new';self.owner=None;self.roots=None;self.host_pid=None;self.token=None
        self.tiny=None;self.attempted=False;self.initialized=False;self.phases=[];self.errors=[];self.queries=[]
    def identity_now(self):
        value=self.identity()
        assert set(value)=={'pid','start_ticks'} and all(type(x) is int and x>0 for x in value.values()), 'Unknown process identity'
        return value
    def pids_now(self):
        values=self.query()
        assert type(values) is list and all(type(x) is int and x>0 for x in values), 'Unknown native PID list'
        self.queries.append(dict(state=self.state,pids=list(values)))
        return values
    def begin(self):
        try:
            assert self.state=='new', 'Admission cannot be reused'
            self.preflight()  # Original source/clean-tree/9031/full-idle checks, before CUDA.
            assert self.closed(), 'Service active before CUDA admission'
            assert self.pids_now()==[], 'Native GPU PID list must initially be empty'
            self.owner=copy.deepcopy(self.identity_now());self.roots=copy.deepcopy(self.binding())
            self.token=self.nonce();assert type(self.token) is str and self.token
            self.attempted=True;self.state='initializing'
            self.tiny=self.initialize();assert self.tiny is not None;self.initialized=True
            assert self.identity_now()==self.owner and self.binding()==self.roots, 'Identity changed during initialization'
            pids=self.pids_now();assert len(pids)==1, 'Fresh CUDA initialization must yield one native PID'
            self.host_pid=pids[0]  # Never infer it from os.getpid()/NSpid.
            self.state='active';return self.token
        except Exception as error:
            self.state='failed';self.errors.append(repr(error));raise
    def check(self,phase,token):
        try:
            assert self.state=='active' and token==self.token, 'Missing/stale transition token'
            assert self.identity_now()==self.owner and self.binding()==self.roots, 'Container PID/start/root identity changed'
            assert phase.case==['steady4k'] and len(self.phases)<2
            assert phase.variant==('old','A')[len(self.phases)], 'Unexpected phase order/reuse'
            pids=self.pids_now();assert pids==[self.host_pid], 'Native PID missing, foreign or changed'
            self.phases.append(dict(case='steady4k',variant=phase.variant,native_pids=list(pids)))
            return True
        except Exception as error:
            self.state='failed';self.errors.append(repr(error));raise
    def complete(self):
        assert self.state=='active' and [x['variant'] for x in self.phases]==['old','A']
        self.state='completed'
    def receipt(self):
        return dict(state=self.state,query=QUERY,container_identity=self.owner,native_host_pid=self.host_pid,
            root_identity=self.roots,token_sha256=hashlib.sha256(self.token.encode()).hexdigest() if self.token else None,
            tiny_initialization_attempted=self.attempted,cuda_initialization_executed=self.initialized,
            initialization='one torch.zeros(1, device=cuda) plus synchronize',
            initialization_outside_numeric_and_timing=True,cuda_work_may_have_started=self.attempted,
            queries=copy.deepcopy(self.queries),phases=copy.deepcopy(self.phases),errors=list(self.errors),precision_gate=False,official_score=False)
