"""CPU replay and faults for the exact-owner v2 contract, never target work."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import biv150_v8_bias_byte_stage as c
import bench_biv150_v8_bias_byte_stage as driver


def check():
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    files,_=c.archived_files();proof,report=c.archive_audit()
    tree=driver.adapted_run_tree(c.ROOT);compile(tree,'cpu_original_run_inverse','exec')
    rejected=[]
    def reject(label,fn):
        try:fn()
        except (AssertionError,KeyError,ValueError):rejected.append(label)
        else:raise AssertionError('Fault accepted: '+label)
    old=files['out/steady4k/gqa_old/actual.llir'].decode()
    new=files['out/steady4k/gqa_A/actual.llir'].decode()
    for label,bad in [('shared_offset',new.replace('i32 4096)','i32 4097)')),
                      ('tail_zero',new.replace('[ zeroinitializer,','[ poison,')),
                      ('bias',new.replace('0xC160001000000000','0xC160000000000000')),
                      ('numeric_CVT',new.replace('fptrunc float %775 to bfloat','sitofp i8 %499 to bfloat')),
                      ('PV_fragment',new.replace('bfloat %844, i64 1','bfloat %843, i64 1')),
                      ('barrier',new.replace('  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0), !dbg !58',''))]:
        assert bad!=new;reject('SSA_'+label,lambda text=bad:c.byte_stage_ssa(old,text))
    reader=lambda path:files['out/'+str(Path(path).relative_to(c.OLD_OUT))]
    reject('wrong_v1_hash',lambda:c.audit(files['out/compile-proof.json']+b' ',reader))
    reject('missing_role_file',lambda:c.audit(files['out/compile-proof.json'],lambda path:reader(path) if '/A_encoder/' not in path else b''))
    ttgir=files['out/steady4k/gqa_A/actual.ttgir'].decode()
    reject('TTGIR_tail_zero',lambda:c.ttgir_chain(ttgir.replace(', %V_141, %cst_8 ', ', %V_141, %cst_1 ')))
    binding=dict(original_source_root=str(c.OLD_ROOT),original_source_commit=c.SOURCE,
        validator_root=str(c.ROOT),validator_commit='CPU mock version',validator_files={'a':'CPU mock digest'})
    sidecar=dict(binding=binding,allowed_cases=['steady4k'],original_proof_sha256=c.PROOF_SHA,
        original_proof_path=str(c.OLD_OUT/'compile-proof.json'),precision_gate=False,official_score=False)
    c.sidecar_header(sidecar,binding)
    for key,value in [('original_source_root',str(c.ROOT)),('original_source_commit','other'),
                      ('validator_commit','other'),('validator_files',{}),('validator_root','other')]:
        bad=copy.deepcopy(sidecar);bad['binding'][key]=value
        reject('sidecar_'+key,lambda b=bad:c.sidecar_header(b,binding))
    reject('sidecar_other_case',lambda:c.sidecar_header(dict(sidecar,allowed_cases=['steady16k']),binding))
    reject('sidecar_other_proof',lambda:c.sidecar_header(dict(sidecar,original_proof_sha256='other'),binding))
    with tempfile.TemporaryDirectory(prefix='byte-stage-existing-out-') as folder:
        reject('existing_output_before_root_or_device_access',lambda:driver.run(SimpleNamespace(
            numeric_diagnostic=True,case=['steady4k'],out=Path(folder),sidecar=Path(folder)/'missing.json')))
    ns,_=c.frozen_namespace(c.ROOT);ns['json']=json
    source=ast.parse((c.ROOT/'tools/bench_biv150_v8_bias.py').read_bytes())
    cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='Proven')
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'pinned_original_Proven','exec'),ns)
    base=proof['cases'][0]['kernels']['gqa_A'];abi=ns['abi']((7,),dict(num_warps=1),(64,2,2))
    expected=dict(copy.deepcopy(base),abi=abi);fake_proof=dict(cases=[dict(kernels=dict(test=expected))])
    blobs={kind:reader(item['path']) for kind,item in base['files'].items()}
    class Kernel:
        def __init__(self):self.launches=0;self.bad_kind=None;self.bad_regs=False
        def warmup(self,*args,**kwargs):
            ir=dict(blobs)
            if self.bad_kind:ir[self.bad_kind]+=b'bad'
            return SimpleNamespace(hash=base['hash'],asm=ir,n_regs=None if self.bad_regs else base['n_regs'],n_spills=0,
                metadata=SimpleNamespace(**base['metadata'],**base['options']),_init_handles=lambda:None)
        def __getitem__(self,grid):
            def launch(*args,**kwargs):self.launches+=1
            return launch
    Wrapped=driver.all_files_proven(ns['Proven'],SimpleNamespace(abi=ns['abi']))
    good=Kernel();Wrapped(good,'test',fake_proof)[(64,2,2)](7,num_warps=1);assert good.launches==1
    for kind in sorted(c.KINDS):
        kernel=Kernel();kernel.bad_kind=kind
        reject('before_launch_'+kind,lambda k=kernel:Wrapped(k,'test',fake_proof)[(64,2,2)](7,num_warps=1))
        assert kernel.launches==0
    kernel=Kernel();guard=Wrapped(kernel,'test',fake_proof);kernel.bad_regs=True
    reject('unknown_resources_before_launch',lambda:guard[(64,2,2)](7,num_warps=1))
    assert not guard.six_checked and kernel.launches==0
    kernel.bad_regs=False;kernel.bad_kind='cubin'
    reject('failed_recheck_cannot_cache_six_file_acceptance',lambda:guard[(64,2,2)](7,num_warps=1))
    assert kernel.launches==0
    # Live-root failures are isolated CPU fixtures. No claim that the original
    # remote source root has been inspected from this machine is made.
    old_git,old_pins,old_oldroot=c.git,c.source_pins,c.OLD_ROOT
    try:
        c.source_pins=lambda root:{}
        for label,fn in [('old_commit',lambda root,*a:'wrong' if a==('rev-parse','HEAD') else ''),
                         ('old_dirty',lambda root,*a:c.SOURCE if a==('rev-parse','HEAD') else 'dirty'),
                         ('validator_dirty',lambda root,*a:c.SOURCE if a==('rev-parse','HEAD') else ('dirty' if root==c.ROOT else ''))]:
            c.git=fn;reject('dual_root_'+label,c.live_roots)
        c.OLD_ROOT=c.ROOT;reject('dual_root_alias',c.live_roots)
    finally:c.git,c.source_pins,c.OLD_ROOT=old_git,old_pins,old_oldroot
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    return dict(report=report,rejected_faults=rejected,original_v1_bytes_unchanged=c.sha(files['out/compile-proof.json'])==c.PROOF_SHA,
        unchanged_run_body_ast_inverse=True,all_six_runtime_artifacts_required=True,
        original_remote_root_live_verified=False,jit_executed=False,gpu_executed=False,precision_gate=False,official_score=False)


if __name__=='__main__':print(json.dumps(check(),indent=2))
