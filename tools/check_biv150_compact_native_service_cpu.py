#!/usr/bin/env python3
"""Combined91/98 CPU contract, standalone-controller C3, and pending-ON refusal."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as S

ROOT=Path(__file__).resolve().parents[1]
A='5ca6a5464f9e8d11f30428c46508ff0b76e204de'
B='c30cf4e2f56b58efaca100ffa54157dd53744944'
ENV='VLLM_FL_BIV150_KONLY_COMPACT_NATIVE_DECODE'
SO=Path('/private/tmp/cjj-native-k8-compile-v1-raw-20261028/out/build/cjj_native_k8_qk_probe.so')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def child(mode):
    assert not any(n=='vllm_fl' or n.startswith('vllm_fl.') for n in sys.modules)
    sem=load('_standalone_without_parent_compile_semantics',ROOT/'vllm_fl/compile_semantics.py')
    config=S(model_config=S(hf_config=S(**sem._GEOMETRY)))
    flags={'VLLM_FL_BIV150_NATIVE_W8A8':'0','VLLM_FL_BIV150_NATIVE_GQA':'0'}
    if mode=='encoded-native':
        flags.update(VLLM_FL_BIV150_KONLY_ENCODED_SERVICE='1',VLLM_FL_BIV150_KONLY_NATIVE_DECODE='1',
            VLLM_FL_BIV150_KONLY_NATIVE_LIBRARY=str(SO),
            VLLM_FL_BIV150_KONLY_NATIVE_LIBRARY_SHA256='22e4a5e2a0b9d9a36ea390a11126417e1bf58e3c27f6b96ba0913b27690f09dd')
    elif mode=='compact-TL':flags['VLLM_FL_BIV150_KONLY_COMPACT_SERVICE']='1'
    elif mode=='pending-native':flags.update({ENV:'1','VLLM_FL_BIV150_KONLY_COMPACT_SERVICE':'1'})
    else:assert mode=='stock'
    try:
        result=sem.build_fingerprint(config,'iluvatar',environ=flags,abi={'CPU_fixture':True})
    except RuntimeError as exc:
        assert mode=='pending-native' and 'PENDING qualification' in str(exc)
        result=dict(rejected=True,reason=str(exc))
    else:
        assert mode!='pending-native'
        if mode=='encoded-native':
            assert result['konly_encoded']['native_decode']['library_sha256']==flags['VLLM_FL_BIV150_KONLY_NATIVE_LIBRARY_SHA256']
            assert 'konly_compact' not in result
        if mode=='compact-TL':
            assert result['konly_compact']['page_bytes']==12544 and result['konly_compact']['q32_hoist_enabled'] is False
            assert 'konly_encoded' not in result
    assert not any(n=='vllm_fl' or n.startswith('vllm_fl.') for n in sys.modules)
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    return dict(mode=mode,without_parent_package=True,frameworks_imported=False,result=result)


def run():
    q=load('_CPU_compact_native_qualification',ROOT/'vllm_fl/compact_native_qualification.py')
    assert q.QUALIFICATION['state']=='pending'
    assert q.preflight({}) is None and q.preflight({ENV:'0',q.SHA_ENV:None}) is None
    rejected=[]
    def reject(name,fn):
        try:fn()
        except RuntimeError as exc:rejected.append(dict(name=name,reason=str(exc)))
        else:raise AssertionError('accepted '+name)
    for name,flags in [('missing_SO',{ENV:'1'}),('missing_SHA_None',{ENV:'1',q.SHA_ENV:None}),
        ('attempt_old22e',{ENV:'1',q.SHA_ENV:'22e4a5e2a0b9d9a36ea390a11126417e1bf58e3c27f6b96ba0913b27690f09dd'}),
        ('two_formats',{ENV:'1','VLLM_FL_BIV150_KONLY_ENCODED_SERVICE':'1','VLLM_FL_BIV150_KONLY_COMPACT_SERVICE':'1'})]:
        reject(name,lambda flags=flags:q.preflight(flags))
    original=q.QUALIFICATION
    for value in (None,{},dict(state='component-reviewed'),dict(state='component-reviewed',library_sha256=None)):
        q.QUALIFICATION=value;reject('incomplete_pin_'+str(value),q.pin)
    q.QUALIFICATION=original
    # Exact inverse of the one new package guard, preserving every legacy byte.
    init=(ROOT/'vllm_fl/__init__.py').read_bytes()
    start=init.index(b'# Pending compact-native must reject before Torch')
    end=init.index(b'# torch.float4_e2m1fn_x2 exists only',start)
    guard=init[start:end]
    parent_init=subprocess.check_output(['git','-C',str(ROOT),'show',A+':vllm_fl/__init__.py'])
    assert init[:start]+init[end:]==parent_init
    namespace={'os':os,'__builtins__':__builtins__}
    before_keys=set(namespace);before_modules=set(sys.modules);previous=os.environ.get(ENV)
    try:
        os.environ[ENV]='0';exec(compile(guard,'<actual-unique-package-guard-OFF>','exec'),namespace)
    finally:
        if previous is None:os.environ.pop(ENV,None)
        else:os.environ[ENV]=previous
    assert set(namespace)==before_keys and set(sys.modules)==before_modules
    # Actual full package prefix executes in a clean child, not a fake parent.
    code='''import importlib.util,sys
from pathlib import Path
p=Path(sys.argv[1]);s=importlib.util.spec_from_file_location('vllm_fl',p,submodule_search_locations=[str(p.parent)])
m=importlib.util.module_from_spec(s);sys.modules['vllm_fl']=m
try:s.loader.exec_module(m)
except RuntimeError as e:
 assert 'PENDING qualification' in str(e)
 assert not {'torch','triton','numpy','vllm','vllm_fl.worker'}.intersection(sys.modules)
 print('actual package rejected before Torch/worker')
else:raise AssertionError('pending package imported')
'''
    env=dict(os.environ,**{ENV:'1'})
    package=subprocess.run([sys.executable,'-B','-c',code,str(ROOT/'vllm_fl/__init__.py')],env=env,text=True,capture_output=True,check=True)
    modes={}
    for mode in ('stock','encoded-native','compact-TL','pending-native'):
        p=subprocess.run([sys.executable,'-B',__file__,'--child',mode],text=True,capture_output=True,check=True)
        modes[mode]=json.loads(p.stdout)
    originals={}
    for file,commit in [('check_biv150_konly_compact_install_cpu.py',A),('check_biv150_native_k8_service_cpu.py',B)]:
        path=ROOT/'tools'/file;raw=path.read_bytes()
        assert raw==subprocess.check_output(['git','-C',str(ROOT),'show',commit+':tools/'+file])
        p=subprocess.run([sys.executable,'-B',str(path)],env=dict(os.environ,**{ENV:'0'}),text=True,capture_output=True)
        assert p.returncode==0,p.stderr
        originals[file]=dict(checker_unchanged=True,source_sha256=hashlib.sha256(raw).hexdigest(),result=json.loads(p.stdout))
    old_device=ROOT/'tools/check_biv150_konly_compact_device_cpu.py'
    assert old_device.read_bytes()==subprocess.check_output(['git','-C',str(ROOT),'show',A+':tools/'+old_device.name])
    p=subprocess.run([sys.executable,'-B',str(old_device)],env=dict(os.environ,**{ENV:'0'}),text=True,capture_output=True)
    assert p.returncode!=0 and 'bootstrap source hash mismatch: vllm_fl' in p.stderr
    # GPU/generated math sources remain exact parents; no shared service proof is transferred.
    unchanged={}
    for prefix,commit in [('compact',A),('encoded',B)]:
        for role in ('writer','decode','q32','multi2','raw3d','normalized2d128'):
            name='vllm_fl/patches/biv150_konly_'+prefix+'_'+role+'.py';raw=(ROOT/name).read_bytes()
            assert raw==subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+name])
            unchanged[name]=hashlib.sha256(raw).hexdigest()
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    return dict(CPU_only=True,pending_qualification_explicit=True,rejected=rejected,
        package_guard_exact_inverse=True,parent91_init_sha256=hashlib.sha256(parent_init).hexdigest(),
        unique_guard_sha256=hashlib.sha256(guard).hexdigest(),guard_OFF_no_new_module_or_binding=True,
        old91_device_CPU_original_refusal=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr),
        package_prefix_subprocess=package.stdout.strip(),standalone_C3_subprocess=modes,
        original_checks=originals,GPU_sources_byte_exact=unchanged,
        actual_new_SO_known=False,library_loaded=False,device_verified=False,numeric_verified=False,
        model_verified=False,capacity_reclaimed=False,performance_verified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--child',choices=('stock','encoded-native','compact-TL','pending-native'))
    a=p.parse_args();print(json.dumps(child(a.child) if a.child else run(),sort_keys=True,indent=2))
