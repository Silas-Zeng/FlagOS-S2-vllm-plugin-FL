#!/usr/bin/env python3
"""CPU metadata fault fixtures only; fake IR/SO is never loaded or qualified."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import biv150_native_k8_compact_factory as f
import biv150_native_k8_compact_consumer_contract as c
import probe_biv150_native_k8_compact_consumer as frontend


def run():
    rejected=[]
    def reject(name,call):
        try:call()
        except (AssertionError,KeyError,TypeError):rejected.append(name)
        else:raise AssertionError('accepted mutant '+name)
    class T:
        def __init__(self,d,data=None):self.d=d;self.data=data
    size=128+2*f.PAGE+128;base=2**40;storage=123
    raw=T(dict(dtype='uint8',shape=(size,),strides=(1,),pointer=base,storage_id=storage,
        storage_base=base,storage_bytes=size,storage_offset=0,device='cuda:0'),bytes([0xa5])*size)
    role=f.roles(2,base+128,storage);views={}
    for name,d in role.items():
        width={'int8':1,'float32':4,'bfloat16':2}[d['dtype']]
        views[name]=T(dict(d,storage_base=base,storage_bytes=size,storage_offset=(d['pointer']-base)//width,device='cuda:0'))
    observe=lambda x:copy.deepcopy(x.d)
    gate=c.LiveEpoch(raw,views,observe,lambda x:x.data,pages=2,layer_offset=128,format_id=f.FORMAT)
    token=gate.activate(raw.data);assert gate.admit(token)['lease_is_process_local']
    reject('missing_live_lease',lambda:gate.admit(None))
    reject('copied_receipt_is_not_lease',lambda:gate.admit(dict(epoch=1)))
    new=gate.activate(raw.data);reject('stale_live_lease',lambda:gate.admit(token));token=new
    old=views['value'].d['pointer'];views['value'].d['pointer']+=4352
    reject('double_V_offset',lambda:gate.admit(token));views['value'].d['pointer']=old
    old=views['scale'].d['storage_offset'];views['scale'].d['storage_offset']+=1024
    reject('double_scale_offset',lambda:gate.admit(token));views['scale'].d['storage_offset']=old
    old=views['key'].d['storage_id'];views['key'].d['storage_id']+=1
    reject('foreign_cache_storage',lambda:gate.admit(token));views['key'].d['storage_id']=old
    old=views['key'].d['strides'];views['key'].d['strides']=(16384,256,128,1)
    reject('old_cache_stride',lambda:gate.admit(token));views['key'].d['strides']=old
    gate.format_id='cjj-konly-rn32-rne-v2';reject('old_format',lambda:gate.admit(token));gate.format_id=f.FORMAT
    reject('source_alias_cache',lambda:gate.admit(token,readonly=(raw,)))
    foreign=T(dict(raw.d,pointer=base+size,storage_base=base+size,storage_id=124))
    reject('output_alias_other_input',lambda:gate.admit(token,readonly=(foreign,),outputs=(foreign,)))
    raw.data=raw.data[:-1]+b'\x00';reject('GPU_readback_mismatch',lambda:gate.activate(bytes([0xa5])*size))
    reject('failed_readback_invalidates_lease',lambda:gate.admit(token));raw.data=bytes([0xa5])*size
    gate.close();reject('closed_epoch',lambda:gate.admit(token))
    # Compact serialization independently checks every address against the
    # canonical counter fixture, including its actual nonzero layer offset.
    import biv150_native_k8_consumer_v2_fixture as ref
    case=ref.fixture(32,4,2560,1);old=case['cache'];new=frontend.compact_bytes(old)
    assert new[:ref.LAYER_OFFSET]==old[:ref.LAYER_OFFSET] and new[-ref.CACHE_GUARD:]==old[-ref.CACHE_GUARD:]
    for page in range(ref.PAGES):
        for slot in range(16):
            for head in range(2):
                for dim in range(128):
                    k,s,v=f.addresses(page,slot,head,dim);a=ref.LAYER_OFFSET+page*16384
                    assert new[ref.LAYER_OFFSET+k]==old[a+slot*256+head*128+dim]
                    assert new[ref.LAYER_OFFSET+s:ref.LAYER_OFFSET+s+4]==old[a+4096+slot*8+head*4:a+4100+slot*8+head*4]
                    b=a+8192+slot*512+head*256+dim*2
                    assert new[ref.LAYER_OFFSET+v:ref.LAYER_OFFSET+v+2]==old[b:b+2]
        assert new[ref.LAYER_OFFSET+page*f.PAGE+4224:ref.LAYER_OFFSET+page*f.PAGE+4352]==bytes([0xa5])*128
    with tempfile.TemporaryDirectory(prefix='cjj-compact-consumer-CPU-') as directory:
        root=Path(directory);rows=[]
        def save(name,raw):
            path=root/name;path.parent.mkdir(exist_ok=True,parents=True);path.write_bytes(raw)
            row=dict(relative=name,bytes=len(raw),sha256=f.sha(raw));return row
        for role in f.FILES:
            path=f.path(role);rows.append(save(path.name,path.read_bytes()))
        header=f.ROOT/'tools/kernels/biv150_native_k8_codec.cuh';rows.append(save(header.name,header.read_bytes()))
        library=root/'build/cpu-fixture.so';rows.append(save('build/cpu-fixture.so',b'CPU fixture, never loaded'))
        import probe_biv150_native_k8_compact as compiler
        summary=dict(completed=True,compile_passed=True,errors=[],source_transform=f.verify(),artifact_manifest=rows,
            library=dict(path=str(library),sha256=f.sha(library.read_bytes())),explicit_cuda_flags=compiler.build_module().CUDA_FLAGS,
            **{k:0 for k in ('attention_launches','encoder_launches','codec_launches','reducer_launches','input_initializations')})
        path=root/'compile-summary.json';path.write_text(json.dumps(summary))
        exports=[save('actual-so-'+label+'.stdout.txt',b'CPU metadata fixture, not target IR') for label in ('sass','IR')]
        inspect=dict(completed=True,errors=[],compile_summary_sha256=f.sha(path.read_bytes()),library=summary['library'],objdump='/CPU-fixture-not-executable',
            commands=[dict(argv=['/CPU-fixture-not-executable','--'+label,str(library)],exit_code=0) for label in ('sass','IR')],artifact_manifest=exports)
        ip=root/'actual-so-inspection.json';ip.write_text(json.dumps(inspect))
        review=c.review_template(path);rp=root/'review.json'
        def admit():
            rp.write_text(json.dumps(review));return c.admission(path,rp,f.sha(rp.read_bytes()),library)
        reject('unreviewed_actual_SO',admit)
        for key in c.REVIEW_FLAGS:review[key]=True
        review['notes']='CPU admission control fixture only; not target evidence'
        for key in review['static_resources']:
            review['static_resources'][key]=dict(vgpr=100,sgpr=68,shared_bytes=8192,private_segment_bytes=0,vgpr_spills=0,sgpr_spills=0)
        assert admit()['qualification_inherited_from_original22e'] is False
        review['version']=True;reject('bool_review_version',admit);review['version']=1
        for key,value in (('vgpr',None),('sgpr',True),('vgpr_spills',None),('sgpr_spills',False),('shared_bytes',4096)):
            old=review['static_resources']['S2'][key];review['static_resources']['S2'][key]=value
            reject('resource_'+key+'_'+str(value),admit);review['static_resources']['S2'][key]=old
        old=review['actual_SO_SASS_sha256'];review['actual_SO_SASS_sha256']='0'*64
        reject('different_actual_SASS',admit);review['actual_SO_SASS_sha256']=old
        library.write_bytes(b'tampered');reject('tampered_compiled_SO',admit)
    default=subprocess.run([sys.executable,'-B',str(f.ROOT/'tools/probe_biv150_native_k8_compact_consumer.py')],text=True,capture_output=True,check=True)
    assert json.loads(default.stdout)==frontend.plan()
    missing=subprocess.run([sys.executable,'-B',str(f.ROOT/'tools/probe_biv150_native_k8_compact_consumer.py'),'--gpu'],text=True,capture_output=True)
    assert missing.returncode!=0 and 'AssertionError' in missing.stderr
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    return dict(CPU_only=True,rejected=rejected,actual_target_or_live_epoch_verified=False,
        canonical_fixture=[32,4,2560,1],counter_cache_all_bytes_remapped=True,
        nonzero_layer_offset=ref.LAYER_OFFSET,default_plan_equal=True,
        missing_device_admission_rejected_before_framework=True,
        fake_compiler_artifacts_only_test_admission_control=True,numeric_device_pending=True)


if __name__=='__main__':print(json.dumps(run(),sort_keys=True,indent=2))
