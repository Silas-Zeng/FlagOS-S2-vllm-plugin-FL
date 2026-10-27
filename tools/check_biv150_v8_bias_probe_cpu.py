"""CPU-only frozen-consumer/IR/bitwise/pre-launch falsifiers. No actual new IR."""
import ast
import copy
import json
from pathlib import Path
import re
import struct
import sys
import tarfile
import tempfile
from types import SimpleNamespace

import bench_biv150_v8_bias as runner
from biv150_v8_bias_probe import ROOT, OPTION_KEYS, adapted_probe, mechanism_check, descriptor_functions
from biv150_v8_bf16p_validation import numeric_timing_policy


class BytesTensor:
    def __init__(self,raw,dtype='uint8',shape=None):self.raw,self.dtype,self.shape=raw,dtype,shape or (len(raw),)
    def contiguous(self):return self
    def view(self,dtype):return BytesTensor(self.raw,dtype)


class ByteBackend:
    Tensor=BytesTensor;uint8='uint8'
    @staticmethod
    def equal(a,b):return a.raw==b.raw
    @staticmethod
    def save(value,path):Path(path).write_text('CPU fake-backend failure evidence only')


def check():
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    tree,adaptation=adapted_probe();compile(tree,'adapted_private_consumer','exec')
    plan,_=descriptor_functions()
    assert plan(64,4096,2)['tensors']['scratch']['stride']==[32768,2048,128,1]
    assert plan(32,16384,4)['tensors']['maxima']['stride']==[256,16,1]
    directory=ROOT/'docs/evidence/v8-bias-decode-20261027'
    manifest=json.loads((directory/'old-v8-ir-manifest.json').read_text())
    assert runner.sha((directory/'old-v8-ir.tar.gz').read_bytes())==manifest['archive_sha256']
    with tarfile.open(directory/'old-v8-ir.tar.gz') as archive:
        old={name:archive.extractfile('actual.'+name).read().decode() for name in ('ttgir','llir','asm')}
    for item in manifest['files']:
        assert runner.sha(old[item['path'].split('.')[-1]].encode())==item['sha256']
    rejected=[]
    def reject(label,action):
        try:action()
        except (AssertionError,KeyError,ValueError):rejected.append(label)
        else:raise AssertionError('Fault reached acceptance: '+label)
    old_verdict=mechanism_check(old,old);assert old_verdict['mechanism_passed'] is False
    # Parser grammar fixture only: physical SSA/register types are not repaired,
    # this text must never be compiled or treated as actual GPU evidence.
    new=dict(old)
    new['ttgir']=new['ttgir'].replace('arith.sitofp','arith.bitcast')
    def bias(match):
        dst,src=match[1],match[2]
        return (f'  %bias{dst} = bitcast i32 %{src} to float\n'
                f'  %wide{dst} = fsub float %bias{dst}, 8388736.0\n'
                f'  %{dst} = fptrunc float %wide{dst} to bfloat')
    new['llir']=re.sub(r'^\s*%(\d+) = sitofp i8 %(\d+) to bfloat[^\n]*',bias,new['llir'],flags=re.M)
    new['asm']=new['asm'].replace('ml_cvt_i16_i8','ml_or_b32').replace('ml_cvt_bf16_i16','ml_cvt_bf16_f32')
    positive=mechanism_check(old,new);assert positive['mechanism_passed'],positive['reasons']
    for label,mutant in [('old_CVT',old),('missing_LLVM_bias',dict(new,llir=new['llir'].replace('8388736.0','0.0'))),
                         ('ASM_integer_to_float',dict(new,asm=new['asm']+'\n ml_cvt_f32_i8 v1, v1\n')),
                         ('ASM_scalar_integer_to_float',dict(new,asm=new['asm']+'\n sl_cvt_f32_u8 s1, s1\n')),
                         ('wrong_MMA',dict(new,asm=new['asm'].replace('bf16_m16n16k16','i8_m16n16k32'))),
                         ('changed_store',dict(new,asm=new['asm'].replace('ml_slb_store_b16','ml_slb_store_b32',1))),
                         ('missing_IR',dict(new,llir=''))]:
        verdict=mechanism_check(old,mutant);assert not verdict['mechanism_passed'],label;rejected.append(label)
    hard=dict.fromkeys(('finite','baseline_numeric','codec','slot_mapping','unmapped_bytes','graph_eager'),True)
    policy=numeric_timing_policy(False,hard,diagnostic=True)
    assert policy['numeric_passed'] is False and policy['precision_gate'] is False
    reject('default_strict_old_ragged_failure',lambda:numeric_timing_policy(False,hard))
    for key in hard:
        reject('diagnostic_hard_'+key,lambda k=key:numeric_timing_policy(False,dict(hard,**{k:False}),diagnostic=True))
    metadata=dict(shared=4096,num_warps=1,num_stages=1,use_sme=79,warp_size=64,triton_version='3.6.0')
    opts=dict(enable_fp_fusion=True,enable_reflect_ftz=True,default_dot_input_precision='tf32',
              maxnreg=None,backend_name='corex',arch='sm71',sanitize_overflow=True,debug=False)
    ir=dict(ttgir='typed',llir='llvm',asm='asm')
    call_abi=runner.abi((7,),dict(num_warps=1),(8,2,16))
    expected=dict(hash='allowed',abi=call_abi,options=opts,n_regs=143,n_spills=0,metadata=metadata,
                  files={name:dict(sha256=runner.sha(text.encode())) for name,text in ir.items()})
    runner.known_resources(expected)
    for key,value in [('n_regs',None),('n_regs',True),('n_regs',0),('n_spills',None),('n_spills',False),('n_spills',1)]:
        reject('unknown_resource_'+key+'_'+str(value),lambda k=key,v=value:runner.known_resources(dict(expected,**{k:v})))
    for key,value in [('shared',None),('shared',False),('num_warps',None),('num_stages',0),
                      ('warp_size',None),('use_sme',None),('triton_version',None)]:
        reject('unknown_metadata_'+key,lambda k=key,v=value:runner.known_resources(dict(expected,metadata=dict(metadata,**{k:v}))))
    with tempfile.TemporaryDirectory(prefix='bias-resource-contract-') as folder:
        def frozen(texts,label):
            files={}
            for name,text in texts.items():
                path=Path(folder)/(label+'.'+name);path.write_text(text)
                files[name]=dict(path=str(path),bytes=len(text.encode()),sha256=runner.sha(text.encode()))
            return files
        kernels={role:dict(copy.deepcopy(expected),files=frozen(old,role)) for role in runner.ROLES}
        kernels['gqa_A'].update(hash='new',n_regs=137,metadata=dict(metadata,use_sme=78,shared=2048),files=frozen(new,'A'))
        row=dict(kernels=kernels)
        resources=runner.inspect(row)
        assert resources['mechanism_passed'],resources['reasons']
        assert set(resources['evidence']['resources'])==runner.ROLES
        assert resources['evidence']['resources']['gqa_A']['metadata']['use_sme']==78
        kernels['old_encoder']['n_spills']=None
        assert not runner.inspect(row)['mechanism_passed'], 'Unknown resource on a non-GQA role must stop'
    proof=dict(cases=[dict(kernels=dict(test=expected))])
    class Kernel:
        def __init__(self,**updates):self.launches=0;self.updates=updates
        def warmup(self,*args,**kwargs):
            result=dict(hash='allowed',asm=ir,n_regs=143,n_spills=0,
                metadata=SimpleNamespace(**opts,**metadata),_init_handles=lambda:None)
            result.update(self.updates);return SimpleNamespace(**result)
        def __getitem__(self,grid):
            def launch(*args,**kwargs):self.launches+=1
            return launch
    kernel=Kernel();runner.Proven(kernel,'test',proof)[(8,2,16)](7,num_warps=1);assert kernel.launches==1
    for label,updates in [('hash',dict(hash='other')),('IR',dict(asm=dict(ir,asm='wrong'))),
                           ('resource',dict(n_regs=144)),('options',dict(metadata=SimpleNamespace(**dict(opts,enable_fp_fusion=False),**metadata))),
                           ('metadata',dict(metadata=SimpleNamespace(**opts,**dict(metadata,use_sme=78)))),
                           ('unknown_registers',dict(n_regs=None)),('bool_registers',dict(n_regs=True)),
                           ('unknown_spills',dict(n_spills=None)),('bool_spills',dict(n_spills=False))]:
        kernel=Kernel(**updates)
        reject('before_launch_'+label,lambda k=kernel:runner.Proven(k,'test',proof)[(8,2,16)](7,num_warps=1))
        assert kernel.launches==0
    kernel=Kernel()
    reject('before_launch_ABI',lambda:runner.Proven(kernel,'test',proof)[(8,2,16)](9,num_warps=1))
    assert kernel.launches==0
    with tempfile.TemporaryDirectory(prefix='bias-private-byte-gates-') as folder:
        scope=runner.NumericScope({},Path(folder),tensor_backend=ByteBackend())
        fields={name:dict(shape=[1],dtype='float32',bytes=BytesTensor(struct.pack('<I',0x7FC00000))) for name in ('scratch','maxima','sums','out')}
        assert scope.equal(fields,copy.deepcopy(fields)), 'Equal NaN payload bits must match'
        payload=copy.deepcopy(fields);payload['out']['bytes']=BytesTensor(struct.pack('<I',0x7FC00001))
        assert not scope.equal(fields,payload), 'Different NaN payload bits must not match'
        zero=dict(fields,out=dict(shape=[1],dtype='float32',bytes=BytesTensor(struct.pack('<I',0))))
        minuszero=dict(fields,out=dict(shape=[1],dtype='float32',bytes=BytesTensor(struct.pack('<I',0x80000000))))
        assert not scope.equal(zero,minuszero), 'Signed zero must be compared by bytes'
        fixture=dict(table=[1],lengths=(1,),slot_mapping=(16,),patterns=[dict(query=BytesTensor(b'q')),dict(query=BytesTensor(b'r'))])
        key,value=BytesTensor(b'k'),BytesTensor(b'v')
        for pattern in (0,1):
            artifact=dict(case='ragged',pattern=pattern,bitwise_fields=fields)
            scope.capture(SimpleNamespace(variant='old'),fixture,artifact,key,value)
            scope.capture(SimpleNamespace(variant='A'),fixture,artifact,key,value)
        scope.require(SimpleNamespace(variant='A'),'ragged')
        for name in fields:
            bad=copy.deepcopy(fields);bad[name]['bytes']=BytesTensor(b'bad')
            reject('bitwise_'+name,lambda b=bad:scope.capture(SimpleNamespace(variant='A'),fixture,
                dict(case='ragged',pattern=0,bitwise_fields=b),key,value))
        reject('different_inputs',lambda:scope.capture(SimpleNamespace(variant='A'),fixture,
            dict(case='ragged',pattern=0,bitwise_fields=fields),BytesTensor(b'wrong'),value))
        reject('timing_after_failed_recheck',lambda:scope.require(SimpleNamespace(variant='A'),'ragged'))
        reject('graph_eager_raw_field_mismatch',lambda:scope.assert_snapshots(fields,zero))
        empty=runner.NumericScope({},Path(folder),tensor_backend=ByteBackend())
        reject('timing_without_two_patterns',lambda:empty.require(SimpleNamespace(variant='A'),'ragged'))
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    return dict(adaptation=adaptation,old_actual_IR_counterexample=old_verdict,
        synthetic_parser_positive_only=True,synthetic_IR_is_executable=False,
        rejected_faults=rejected,old_numeric_failure_policy=policy,
        nan_payload_and_signed_zero_compared_as_bytes=True,narrowed_descriptors_preserved=True,
        known_resources_required_for_all_ten_roles=True,different_known_use_sme_shared_registers_allowed=True,
        actual_new_IR_available=False,consumer_device_tested=False,jit_executed=False,gpu_executed=False,
        private_consumer_source_wired=True,precision_gate=False,official_score=False)


if __name__=='__main__':print(json.dumps(check(),indent=2))
