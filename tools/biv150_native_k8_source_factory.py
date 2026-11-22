"""Exact reversible b3 source transform: private I8-QK, original BF16-PV."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'csrc/iluvatar/native_gqa_s2.cu'
PINS=json.loads((ROOT/'docs/evidence/native-k8-qk-feasibility-20261028/source-identities.json').read_text())
SOURCE_SHA=PINS['files']['csrc/iluvatar/native_gqa_s2.cu']['sha256']
DEST=ROOT/'tools/kernels/biv150_native_k8_qk.cu'

def digest(raw):return hashlib.sha256(raw).hexdigest()

def patches(source):
    def block(start,end):
        a=source.index(start);b=source.index(end,a)
        return source[a:b]
    added_types='''using QKI8Shape=nvcuda::wmma::MMBaseInfo<16,16,32,16,16,128>;
using AI8=nvcuda::wmma::FragmentA<QKI8Shape,signed char,nvcuda::wmma::row_major>;
using BI8=nvcuda::wmma::FragmentB<QKI8Shape,signed char,nvcuda::wmma::col_major>;
using CI8=nvcuda::wmma::FragmentC<QKI8Shape,signed int>;
static_assert(sizeof(AI8)==8 && sizeof(BI8)==8 && sizeof(CI8)==16,"I8 fragment ABI changed");
'''
    kload='''// Own independently checked I8 col image; no BF16 fragment/SME alias.
__device__ __forceinline__ unsigned k_i8_index(unsigned slot,unsigned dim){
 const unsigned panel=dim/64,within=dim%64,quartet=within/4,byte=within%4,bank=slot/4,subslot=slot%4;
 return panel*1024+bank*256+(quartet%4)*64+((quartet/4)^bank)*16+subslot*4+byte;
}
__device__ __forceinline__ void issue_k_i8(unsigned* stage,const signed char* key,int page,unsigned head,unsigned valid,unsigned lane){
 const auto* base=key+static_cast<std::int64_t>(page)*16384+head*128;
 // 512 unique aligned words. Invalid slots do not perform a source read.
 for(unsigned word=lane;word<512;word+=64){
  const unsigned slot=word/32,dim=(word%32)*4;
  unsigned bits=0;
  if(slot<valid)bits=*reinterpret_cast<const unsigned*>(base+slot*256+dim);
  stage[k_i8_index(slot,dim)/4]=bits;
 }
}
__device__ __forceinline__ int signed_i32_word(unsigned bits){
 union{unsigned bits;int signed_value;}value;value.bits=bits;return value.signed_value;
}
'''
    read_q=''' AI8 q[4];
 float qscale0,qscale1;
 float values0[8],values1[8],peak0=0.0f,peak1=0.0f;
 #pragma unroll
 for(unsigned k=0;k<8;++k){
  const auto* base=query+static_cast<std::int64_t>(seq)*q_token_stride;
  values0[k]=__bfloat162float(base[(h*8+2*g)*q_head_stride+16*k+col]);
  values1[k]=__bfloat162float(base[(h*8+2*g+1)*q_head_stride+16*k+col]);
  peak0=fmaxf(peak0,fabsf(values0[k]));peak1=fmaxf(peak1,fabsf(values1[k]));
 }
 #pragma unroll
 for(unsigned mask=8;mask;mask>>=1){
  peak0=fmaxf(peak0,shuffle(peak0,lane^mask));peak1=fmaxf(peak1,shuffle(peak1,lane^mask));
 }
 const float input_scale0=cjj_k8_scale_rn32(peak0),input_scale1=cjj_k8_scale_rn32(peak1);
 // C row g/g+4 differs from original input pair 2*g/2*g+1.
 // All lane collectives execute unconditionally before the local selection.
 const float c00=shuffle(input_scale0,(g/2)*16+col),c01=shuffle(input_scale1,(g/2)*16+col);
 const float c10=shuffle(input_scale0,(g/2+2)*16+col),c11=shuffle(input_scale1,(g/2+2)*16+col);
 qscale0=(g&1)?c01:c00;qscale1=(g&1)?c11:c10;
 #pragma unroll
 for(unsigned k=0;k<4;++k){
  #pragma unroll
  for(unsigned component=0;component<2;++component){
   const unsigned i=2*k+component;
   const unsigned pair=(static_cast<unsigned>(cjj_k8_code_rne(values0[i],input_scale0))&255u)
      |((static_cast<unsigned>(cjj_k8_code_rne(values1[i],input_scale1))&255u)<<8);
   const unsigned a=shuffle_bits(pair,((2*g)%4)*16+col);
   const unsigned b=shuffle_bits(pair,((2*g+1)%4)*16+col);
   q[k][component]=(a&65535u)|((b&65535u)<<16);
  }
 }
'''
    qk=''' // K ordinary global loads complete through dependent shared stores/barrier;
 // V retains original SME issue/wait/layout. No assumed SME8 instruction.
 issue_k_i8(stage,key,page,head,valid,lane);__syncthreads();
 static_assert(!OverlapV,"Only original overlap=false is admitted");
 issue_v(dst,descriptor(value,page,head));
 __ivcorex_sl_waitcnt(8ULL);__syncthreads();
 if(valid<16){clear_v_tail(stage,valid,lane);__syncthreads();}
 CI8 scores;
 #pragma unroll
 for(unsigned k=0;k<4;++k){
  BI8 b;nvcuda::wmma::load_matrix_sync_tcu(b,stage,0,k);
  nvcuda::wmma::mma_sync_tcu(scores,q[k],b,scores);
 }
 // Full layer-page0 F32 view; internal +1024 elements is added exactly once.
 const float kscale=col<valid?key_scale_full[static_cast<std::int64_t>(page)*4096+1024+col*2+head]:1.0f;
 const float s0=col<valid?static_cast<float>(signed_i32_word(scores[0]))*qscale0*kscale*scale:-INFINITY;
 const float s1=col<valid?static_cast<float>(signed_i32_word(scores[1]))*qscale1*kscale*scale:-INFINITY;
'''
    ops=[
      ('#include <mma.h>\n','#include <mma.h>\n#include "biv150_native_k8_codec.cuh"\n',1),
      ('using Desc=int __attribute__((ext_vector_type(4)));\n',added_types+'using Desc=int __attribute__((ext_vector_type(4)));\n',1),
      (block('__device__ __forceinline__ void issue_k(', '__device__ __forceinline__ void issue_v('),kload,1),
      (' const A<QKShape> (&q)[8],const __nv_bfloat16* key,const __nv_bfloat16* value,',
       ' const AI8 (&q)[4],float qscale0,float qscale1,const signed char* key,const float* key_scale_full,const __nv_bfloat16* value,',1),
      (block(' issue_k(dst,descriptor(key,page,head));',' float page_m0=s0,page_m1=s1;'),qk,1),
      ('static_assert(COREX_NATIVE_GQA_V_OVERLAP==0 || COREX_NATIVE_GQA_V_OVERLAP==1,"variant must be explicit 0/1");',
       'static_assert(COREX_NATIVE_GQA_V_OVERLAP==0,"Private K8 prototype requires original overlap=false");',1),
      (' const __nv_bfloat16* query,const __nv_bfloat16* key,const __nv_bfloat16* value,',
       ' const __nv_bfloat16* query,const signed char* key,const float* key_scale_full,const __nv_bfloat16* value,',1),
      (block(' A<QKShape> q[8];',' __shared__ __align__(16) unsigned stage[2048];'),read_q,1),
      ('(q,key,value,row[','(q,qscale0,qscale1,key,key_scale_full,value,row[',2),
      (' const void* query,const void* key,const void* value,const int* seq_lens,',
       ' const void* query,const void* key,const void* key_scale_full,const void* value,const int* seq_lens,',2),
      ('static_cast<const __nv_bfloat16*>(query),static_cast<const __nv_bfloat16*>(key),',
       'static_cast<const __nv_bfloat16*>(query),static_cast<const signed char*>(key),static_cast<const float*>(key_scale_full),',2),
      ('corex_native_gqa','corex_native_k8_gqa',source.count('corex_native_gqa')),
    ]
    return ops

def original():
    raw=SOURCE.read_bytes()
    if digest(raw)!=SOURCE_SHA:raise ValueError('Pinned b3 CUDA source differs')
    return raw.decode('utf-8')

def build():
    raw=original();out=raw
    for old,new,count in patches(raw):
        if out.count(old)!=count:raise ValueError('Exact transform anchor count changed: '+old[:80])
        out=out.replace(old,new)
    if inverse(out)!=raw:raise AssertionError('Exact source inverse failed')
    return out

def inverse(out):
    raw=original()
    for old,new,count in reversed(patches(raw)):
        if out.count(new)!=count:raise ValueError('Private source anchor count changed: '+new[:80])
        out=out.replace(new,old)
    if digest(out.encode())!=SOURCE_SHA:raise ValueError('Non-QK source or original formatting changed')
    return out

def verify():
    generated=build();actual=DEST.read_bytes()
    if actual!=generated.encode():raise ValueError('Committed generated private kernel differs')
    return dict(base=PINS['base'],original_sha256=SOURCE_SHA,private_sha256=digest(actual),
        inverse_byte_exact=True,PV_online_and_partial_output_unchanged=True,reducer_source_not_modified=True,
        changes='typed K/code/scale ABI, Q RN32/RNE+packing, K load+QK+score scales, symbol names only',
        device_compiled=False,codec_actual_IR_verified=False)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args()
    if args.write:DEST.write_bytes(build().encode())
    print(json.dumps(verify(),sort_keys=True,indent=2))
