#!/usr/bin/env python3
"""CPU contract for the compile-only caller; never calls a real SDK or device."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import biv150_konly_multi2_service_compile_source as c

def run():
    checks=[]
    def check(value,label):assert value,label;checks.append(label)
    def reject(call,label):
        try:call()
        except ValueError:checks.append(label)
        else:raise AssertionError(label)
    plan=c.plan();new,proof=c.driver_source();old=c.frozen().driver_source()
    check(proof['exact_byte_inverse'] and proof['edits']==9,'nine bounded edits and full byte inverse')
    a,b=ast.parse(old),ast.parse(new)
    allocation=lambda t:next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='allocate')
    check(ast.dump(allocation(a))==ast.dump(allocation(b)),'original uninitialized allocate AST unchanged')
    calls=[n for n in ast.walk(b) if isinstance(n,ast.Call)]
    check(sum(isinstance(n.func,ast.Attribute) and n.func.attr=='prepare' for n in calls)==1,'one prepare invocation')
    check(not any(isinstance(n.func,ast.Subscript) for n in calls),'no reader/encoder/writer callable launch')
    driver=c.driver()
    reject(lambda:driver.run(NS(root_owned_exclusive=False)),'no Root ownership fails before output/SDK')
    with tempfile.TemporaryDirectory(prefix='multi2-compile-cpu-') as temp:
        out=Path(temp)/'absent'
        reject(lambda:driver.run(NS(root_owned_exclusive=True,commit='0'*40,out=out)),'wrong Git freeze fails before output/SDK')
        check(not out.exists(),'rejected compile created no output')
        reject(lambda:driver.run(NS(root_owned_exclusive=True,commit='0'*40,out=Path('relative'))),'relative output rejected')
        reject(lambda:driver.run(NS(root_owned_exclusive=True,commit='0'*40,out=c.ROOT/'forbidden')),'checkout output rejected')
    class Tensor:
        def __init__(self,*strides):self.strides=strides
        def stride(self,i):return self.strides[i]
    t=dict(q=Tensor(2048,128,1),out=Tensor(2048,128,1),k=Tensor(16384,256,128,1),v=Tensor(8192,256,128,1),table=Tensor(8192,1),
        kscale=object(),lengths=object(),cu_q=object(),k_descale=object(),v_descale=object())
    spec=dict(profile='multi2',sequences=2,segments=1,config=dict(BLOCK_M=256,BLOCK_Q=32,TILE_SIZE=64,IS_3D=False))
    kw=c.kwargs(spec,t,True);check(kw==c.frozen().kwargs(spec,t,True),'same complete original ABI mapping')
    check(kw['num_seqs']==2 and kw['query_stride_0']==2048 and kw['stride_k_cache_0']==16384,'fixed original geometry')
    # Exact cache format checks with clearly synthetic bytes, not an executable admission.
    with tempfile.TemporaryDirectory(prefix='multi2-cache-cpu-') as temp:
        root=Path(temp);record=dict(hash='12'*32,function_name='mock',artifacts={})
        directory=root/c.base64.b32encode(bytes.fromhex(record['hash'])).decode().rstrip('=');directory.mkdir();paths={}
        for ext in ('source','ttir','ttgir','llir','asm','cubin','json'):
            p=directory/('mock.'+ext);body=json.dumps({'hash':record['hash']}).encode() if ext=='json' else ('mock '+ext).encode();p.write_bytes(body);paths[p.name]=str(p)
            if ext!='json':record['artifacts'][ext]=c.sha(body)
        index=directory/'__grp__mock.json';index.write_text(json.dumps({'child_paths':paths}))
        check(len(c.cache_identity(root,record)['children'])==7,'all seven cache children match exact record')
        bad=copy.deepcopy(record);bad['artifacts']['cubin']='0'*64
        reject(lambda:c.cache_identity(root,bad),'changed final cubin rejected')
        p=directory/'mock.json';p.write_text(json.dumps({'hash':'0'*64}));reject(lambda:c.cache_identity(root,record),'changed cache metadata hash rejected')
        p.write_text(json.dumps({'hash':record['hash']}));del paths['mock.asm'];index.write_text(json.dumps({'child_paths':paths}));reject(lambda:c.cache_identity(root,record),'incomplete group rejected')
    check('binding is not None' in new and 'prepare retained unhoisted' in new,'fallback is not successful compile')
    check('JIT_warmups_attempted=None' in new,'failed prepare does not guess warmup count')
    check(not {'torch','triton','vllm','vllm_fl'}.intersection(sys.modules),'no framework/device imported')
    return dict(passed=True,checks=checks,check_count=len(checks),plan=plan,GPU_executed=False,
        numeric_verified=False,service_verified=False,source_sha256=c.sha(Path(c.__file__).read_bytes()))

if __name__=='__main__':print(json.dumps(run(),indent=2,sort_keys=True)+'\n',end='')
