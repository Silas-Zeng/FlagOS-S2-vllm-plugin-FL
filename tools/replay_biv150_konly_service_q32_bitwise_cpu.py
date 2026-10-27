"""Independent byte/codec/INT64/FP64 audit of the first actual service Q32 hoist."""
import collections,gzip,hashlib,io,json,pickle,sys,zipfile,math
from pathlib import Path
import numpy as np
from biv150_konly_decode_quant_reference import rn32_div
import tempfile,tarfile,argparse
import biv150_konly_service_q32_bitwise_archive as archive
import biv150_konly_service_q32_fresh_binding as binding
import biv150_konly_service_q32_native as native
sha=lambda x:hashlib.sha256(x).hexdigest()
DT={'BFloat16Storage':'<u2','IntStorage':'<i4','FloatStorage':'<f4','ByteStorage':'u1','DoubleStorage':'<f8','CharStorage':'i1','LongStorage':'<i8'}
class Storage:
 def __init__(self,dtype,key,count,owner):self.dtype,self.key,self.count,self.owner=dtype,key,count,owner;self.value=None
 def raw(self):
  if self.value is None:self.value=self.owner.read('archive/data/'+self.key)
  assert len(self.value)==self.count*np.dtype(DT[self.dtype]).itemsize
  return self.value
class Tensor:
 def __init__(self,storage,offset,shape,stride,*args):self.storage,self.offset,self.shape,self.stride=storage,offset,tuple(shape),tuple(stride)
 def array(self):
  dt=np.dtype(DT[self.storage.dtype]);assert self.offset>=0 and all(x>=0 for x in self.shape+self.stride)
  maximum=self.offset+sum((n-1)*s for n,s in zip(self.shape,self.stride)) if all(self.shape) else self.offset
  assert maximum<self.storage.count
  return np.ndarray(self.shape,dtype=dt,buffer=self.storage.raw(),offset=self.offset*dt.itemsize,strides=tuple(s*dt.itemsize for s in self.stride))
 def raw(self):return self.array().tobytes()
class Safe(pickle.Unpickler):
 def __init__(self,data,owner):super().__init__(data);self.owner=owner;self.storages={}
 def find_class(self,module,name):
  if (module,name)==('torch._utils','_rebuild_tensor_v2'):return Tensor
  if (module,name)==('collections','OrderedDict'):return collections.OrderedDict
  if module=='torch' and name in DT:return name
  raise ValueError('unexpected pickle global '+module+'.'+name)
 def persistent_load(self,v):
  assert v[0]=='storage' and v[3]=='cpu' and v[1] in DT and str(v[2]).isdigit()
  if v[2] not in self.storages:self.storages[v[2]]=Storage(v[1],v[2],v[4],self.owner)
  return self.storages[v[2]]
def bf(bits):return (bits.astype(np.uint32)<<16).view(np.float32).astype(np.float64)
def quant(bits):
 x=bf(bits);assert np.isfinite(x).all();peak=np.max(np.abs(x),axis=-1)
 scale=np.array([1. if p==0 else max(rn32_div(float(p),127.),2.**-126) for p in peak.reshape(-1)],dtype=np.float32).reshape(peak.shape)
 ratio=(x/scale.astype(np.float64)[...,None]).astype(np.float32);encoded=np.clip(np.rint(ratio),-127,127).astype(np.int8)
 magnitude=np.abs(ratio).astype(np.float64);near=np.abs(magnitude-np.floor(magnitude)-.5)<=2*np.spacing(np.abs(ratio)).astype(np.float64)
 count=0
 for index in zip(*np.nonzero(near)):
  encoded[index]=max(-127,min(127,round(rn32_div(float(x[index]),float(scale[index[:-1]])))));count+=1
 return encoded,scale,count

def metrics(actual,ref):
 err=actual-ref;finite=bool(np.isfinite(actual).all() and np.isfinite(ref).all());rms=float(np.linalg.norm(err.reshape(-1))/math.sqrt(err.size));rrms=float(np.linalg.norm(ref.reshape(-1))/math.sqrt(ref.size));maximum=float(np.max(np.abs(err)))
 violations=int(np.count_nonzero(np.abs(err)>.005+.005*np.abs(ref)))
 return dict(finite=finite,pointwise_passed=violations==0,pointwise_violations=violations,relative_rmse=rms/rrms if rrms else (0. if maximum==0 else math.inf),rms=rms,max_abs=maximum)
def replay(files,compile_archive):
    resources={}
    with tempfile.TemporaryDirectory(prefix='cjj-q32-compile-cpu-') as temporary:
        with tarfile.open(compile_archive) as tar:
            members=tar.getmembers();assert len(members)==40 and all(m.isfile() and archive.safe(m.name) for m in members)
            for member in members:
                path=Path(temporary)/member.name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(tar.extractfile(member).read())
        reviewed=binding.load_compile(Path(temporary)/'out')
    for role in binding.ROLES:
        record=json.loads(files['out/'+role+'/record-loaded.json'])
        before=json.loads(files['out/'+role+'/record.json'])
        assert binding.same(record,{**before,'binding':record['binding'],'loaded_resources':record['loaded_resources']})
        observed={e:files['out/'+role+'/actual.'+e] for e in record['files']}
        for e,b in observed.items():assert record['files'][e]==dict(bytes=len(b),sha256=sha(b))
        binding.verify_recompile(record,reviewed['records'][role],observed,reviewed['actual'][role])
        if role=='proposed-hoist':native.audit(reviewed['override'],*(observed[e].decode() for e in ('ttgir','llir','asm')))
        resources[role]=record['loaded_resources']
        assert resources[role]['loaded_resources_verified'] is True and resources[role]['zero_spill_observed'] is True
        assert type(resources[role]['n_regs']) is int and resources[role]['n_regs']>0 and resources[role]['n_spills']==0
        assert (resources[role]['num_warps'],resources[role]['num_stages'],resources[role]['warp_size'])==(16,1,64)
        assert 0<=resources[role]['shared']<=65536
        assert binding.same(record,json.loads(files['out/bitwise-summary.json'])['records'][role])
    manifest=archive.originals(files)
    summary=json.loads(files['out/bitwise-summary.json']);assert summary['completed'] and summary['errors']==[] and summary['reader_launches']==4 and summary['encoder_launches']==0
    assert summary['original_BF16_quality_passed'] is False and summary['precision_gate'] is False
    records=[]
    for pattern in (0,1):
     with zipfile.ZipFile(io.BytesIO(gzip.decompress(files[f'out/pattern-{pattern}.pt.gz']))) as z:
      a=Safe(io.BytesIO(z.read('archive/data.pkl')),z).load();cpu=a['original_CPU'];packed=cpu['packed'].array();q=cpu['q'].array()
      code,scale,repairs=quant(packed[:,0]);raw=np.full((773,16384),0xa5,np.uint8)
      raw[:,:4096]=code.reshape(773,4096).view(np.uint8);raw[:,4096:4224]=scale.reshape(773,32).view(np.uint8).reshape(773,128);raw[:,8192:]=packed[:,1].copy().view(np.uint8).reshape(773,8192)
      assert raw.tobytes()==a['expected_encoded_bytes'].raw()==a['actual_encoded_bytes'].raw()
      assert code.tobytes()==a['expected_codes'].raw() and scale.tobytes()==a['expected_scales'].raw()
      fp={k:sha(cpu[source].raw()) for k,source in [('q','q'),('table','table'),('lengths','lengths'),('cu_q','cu_q'),('k_descale','k_scale'),('v_descale','v_scale')]};fp['encoded']=sha(raw.tobytes());assert fp==a['readonly_before']
      ids=cpu['table'].array()[0,:384].astype(np.int64);keys=bf(packed[ids,0].reshape(6144,2,128));values=bf(packed[ids,1].reshape(6144,2,128))
      rows=np.array(a['reference']['rows'],np.int64)
      if pattern==0:
       assert list(rows)==list(range(2048)) and not q.any()
       visible=4096+rows+1;original=np.repeat(values.cumsum(0)[visible-1]/visible[:,None,None],8,axis=1);quantized=original
      else:
       qc,qs,qrepairs=quant(q[rows]);qq=bf(q[rows]).reshape(len(rows),2,8,128).transpose(1,2,0,3)
       kc=code[ids].reshape(6144,2,128).astype(np.int64);ks=scale[ids].reshape(6144,2).astype(np.float64)
       qi=qc.astype(np.int64).reshape(len(rows),2,8,128).transpose(1,2,0,3)
       ints=np.einsum('hgrd,thd->hgrt',qi,kc,dtype=np.int64,optimize=True);assert np.max(np.abs(ints))<=128*127**2
       score=np.einsum('hgrd,thd->hgrt',qq,keys,optimize=True)*128**-.5
       quant_score=ints.astype(np.float64)*qs.astype(np.float64).reshape(len(rows),2,8).transpose(1,2,0)[...,None]*ks.T[:,None,None,:]*128**-.5
       mask=np.arange(6144)[None,:]<4096+rows[:,None]+1
       def attention(s):
        s=np.where(mask[None,None],s,-np.inf);e=np.exp(s-np.max(s,axis=-1,keepdims=True));p=e/e.sum(axis=-1,keepdims=True)
        return np.einsum('hgrt,thd->rhgd',p,values,optimize=True).reshape(len(rows),16,128)
       original=attention(score);quantized=attention(quant_score)
      reference_error={name:float(np.max(np.abs(value-a['reference'][name].array()))) for name,value in [('original',original),('quantized',quantized)]}
      assert max(reference_error.values())<5e-13
      before=a['guarded_before'];after=a['guarded_after'];outputs=a['outputs'];role_checks={};recomputed={}
      assert outputs['unhoisted'].raw()==outputs['proposed-hoist'].raw()
      for role in ('unhoisted','proposed-hoist'):
       pre=before[role].array();post=after[role].array();out=outputs[role].array()
       assert pre.shape==post.shape==(2050,16,128) and out.shape==(2048,16,128)
       assert np.all(pre[[0,2049]]==0x428e) and np.all(post[[0,2049]]==0x428e)
       interior=pre[1:2049];assert np.all(((interior&0x7f80)==0x7f80)&((interior&127)!=0))
       assert not np.any((out&0x7f80)==0x7f80) and post[1:2049].tobytes()==out.tobytes()
       assert a['readonly_after'][role]==fp
       role_checks[role]=dict(full_before_SHA256=sha(pre.tobytes()),full_after_SHA256=sha(post.tobytes()),full_output_SHA256=sha(out.tobytes()),both_canaries_71=True,pre_poison_all_nan=True,post_all_finite=True,full_output_equals_full_storage_interior=True,all_readonly_equal=True)
       actual=bf(out[rows]);qm=metrics(actual,quantized);om=metrics(actual,original);assert qm['pointwise_passed']
       recomputed[role]=dict(quantized_operand=qm,original_BF16=om)
       stored=summary['patterns'][pattern]['metrics'][role]
       for group in recomputed[role]:
        for k,v in recomputed[role][group].items():
         if isinstance(v,(bool,int)):assert v==stored[group][k]
         else:assert abs(v-stored[group][k])<=1e-12*max(1.,abs(v))
      records.append(dict(pattern=pattern,independent_full_encoded_SHA256=sha(raw.tobytes()),codec_exact=True,rational_boundary_repairs=repairs,INT64_QK_FP64_reference_recomputed=True,reference_max_abs_difference=reference_error,full_output_bitwise_equal=True,role_checks=role_checks,metrics=recomputed))
    
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    report=dict(CPU_only=True,recorded_loaded_resources=resources,actual_compilation_binding_verified=True,new_native_audit_verified=True,source_commit=summary['commit'],raw_files=manifest['raw_files'],raw_bytes=manifest['raw_bytes'],all_download_SHA256_verified=True,patterns=records,reader_launches=4,encoder_launches=0,full_output_bitwise_equal=True,original_BF16_quality_passed=False,precision_gate=False,model_verified=False,Graph_verified=False,side_stream_verified=False,timing_attempted=False,Torch_imported=False,Triton_imported=False)
    return report

def main():
    p=argparse.ArgumentParser(description='Pure CPU bit-array/codec/INT64/FP64 replay; NumPy only, no Torch/Triton')
    p.add_argument('--raw',type=Path);p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    assert bool(a.raw)!=bool(a.directory),'exactly one original snapshot or committed multipart directory required'
    if a.raw:files={p.relative_to(a.raw).as_posix():p.read_bytes() for p in a.raw.rglob('*') if p.is_file()}
    else:files=archive.load(a.directory)[0]
    compile_archive=Path(__file__).resolve().parents[1]/'docs/evidence/konly-service-q32-hoist-20261027/fresh-native-v1/original-compile.tar.gz'
    assert sha(compile_archive.read_bytes())=='aae141e17f58233df0ca18f92c998d18fb3052a2e0a53de5673b79bb68e1340e'
    result=replay(files,compile_archive);text=json.dumps(result,sort_keys=True,indent=2)+'\n'
    if a.out:
        with a.out.open('x') as out:out.write(text)
    print(text,end='')
if __name__=='__main__':main()
