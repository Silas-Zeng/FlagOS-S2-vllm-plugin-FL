"""Exact 8b86 source inverse; one conservative, unqualified K lookahead variant."""
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = 'a455a3de364051fbea80226870c239de3a48e716'
DEVICE_SOURCE = 'a1eb7deee5e8ef99136128984256762096a3657f'
REFERENCE_SO = '8b86cb688ad72dcf9859994ed2092620312cdd5ae96d194904b2bde7926aacc5'
NAMESPACE = 'cjj_native_k8_klookahead_probe'
FILES = {'qk.cu': 'biv150_native_k8_v_overlap_qk.cu',
         'qk.cpp': 'biv150_native_k8_v_overlap_qk.cpp',
         'codec.cu': 'biv150_native_k8_v_overlap_codec.cu',
         'codec.cuh': 'biv150_native_k8_codec.cuh'}
PINS = {'qk.cu': '8f22c9464b4595caf0059e9571e92d1ede83abe1430fda8f95a7e8b649efe3bf',
        'qk.cpp': '5b71b43807c8e5f57a4f495a2733f1738ee1248ebec91f4a2f8e26b84b18e0ef',
        'codec.cu': '8c4d67925f9d731f9fda36c82b75d35da8372ffbb9579b1c2891898a19c371b3',
        'codec.cuh': '37d5c7548c0301ee522f50fb0d4d2da53300676e71b03bb806c2d88779801d79'}
RENAMES = (('cjj_native_k8_v_overlap_gqa_producer', 'cjj_native_k8_klookahead_gqa_producer'),
           ('launch_cjj_native_k8_v_overlap_', 'launch_cjj_native_k8_klookahead_'),
           ('cjj_launch_native_k8_v_overlap_codec', 'cjj_launch_native_k8_klookahead_codec'),
           ('cjj_native_k8_v_overlap_codec_probe', 'cjj_native_k8_klookahead_codec_probe'),
           ('cjj_native_k8_v_overlap_probe', NAMESPACE))

HELPERS = '''// Only a later full page in this segment can enter this packed stage.
// Ordinary global reads start AFTER the current V wait8 + ready barrier.
struct PackedNextK { unsigned word[8]; };
__device__ __forceinline__ void read_next_k(PackedNextK& next,const signed char* key,int page,unsigned head,unsigned lane){
 const auto* base=key+static_cast<std::int64_t>(page)*16384+head*128;
 #pragma unroll
 for(unsigned i=0;i<8;++i){
  const unsigned word=lane+i*64,slot=word/32,dim=(word%32)*4;
  next.word[i]=*reinterpret_cast<const unsigned*>(base+slot*256+dim);
 }
}
__device__ __forceinline__ void publish_next_k(unsigned* stage,const PackedNextK& next,unsigned lane){
 #pragma unroll
 for(unsigned i=0;i<8;++i){
  const unsigned word=lane+i*64,slot=word/32,dim=(word%32)*4;
  stage[k_i8_index(slot,dim)/4]=next.word[i];
 }
}
'''

OLD_LOOP = ''' int position=lower;
 while(position<full_upper){
  consume_page<COREX_NATIVE_GQA_V_OVERLAP!=0>(q,qscale0,qscale1,key,key_scale_full,value,row[position/16],h,16,lane,col,scale,stage,m0,m1,l0,l1,acc);
  position+=16;
 }
'''
NEW_LOOP = ''' int position=lower;
 if(position<full_upper){
  int page=row[position/16];
  issue_k_i8(stage,key,page,h,16,lane);__syncthreads(); // Prologue K0 ready.
  while(position<full_upper){
   const int next_position=position+16;
   const bool has_next=next_position<full_upper;
   int next_page=0;
   if(has_next)next_page=row[next_position/16]; // No tail/next-segment speculation.
   consume_page<COREX_NATIVE_GQA_V_OVERLAP!=0,true>(q,qscale0,qscale1,key,key_scale_full,value,page,h,16,lane,col,scale,stage,m0,m1,l0,l1,acc,has_next,next_page);
   page=next_page;
   position=next_position;
  }
 }
'''

EDITS = (
    ('__device__ __forceinline__ int signed_i32_word(unsigned bits){',
     HELPERS+'__device__ __forceinline__ int signed_i32_word(unsigned bits){'),
    ('template<bool OverlapV>\n__device__ __forceinline__ void consume_page(',
     'template<bool OverlapV,bool KReady=false>\n__device__ __forceinline__ void consume_page('),
    (' C<PVShape> (&acc)[8]){', ' C<PVShape> (&acc)[8],bool has_next=false,int next_page=0){'),
    (' issue_k_i8(stage,key,page,head,valid,lane);__syncthreads();',
     ' if constexpr(!KReady){issue_k_i8(stage,key,page,head,valid,lane);__syncthreads();}'),
    (' }\n #pragma unroll\n for(unsigned n=0;n<8;++n){\n  BRow<PVShape> b;',
     ' }\n PackedNextK next;\n if(has_next)read_next_k(next,key,next_page,head,lane);\n #pragma unroll\n for(unsigned n=0;n<8;++n){\n  BRow<PVShape> b;'),
    (' running_m0=new_m0;running_m1=new_m1;\n __syncthreads(); // All shared fragment reads precede the next page overwrite.',
     ' running_m0=new_m0;running_m1=new_m1;\n if(has_next)publish_next_k(stage,next,lane); // Current QK and PV readers have ended.\n __syncthreads(); // Publish dependent K stores; finish current PV before the next V issue.'),
    (OLD_LOOP, NEW_LOOP),
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def original(role):
    raw = subprocess.check_output(['git', '-C', str(ROOT), 'show',
                                   BASE+':tools/kernels/'+FILES[role]])
    if sha(raw) != PINS[role]:
        raise ValueError('Pinned 8b86 source mismatch: '+role)
    return raw.decode()


def path(role):
    name = FILES[role] if role == 'codec.cuh' else 'biv150_native_k8_klookahead_'+role
    return ROOT/'tools/kernels'/name


def generate(role):
    value = original(role)
    if role == 'qk.cu':
        for old, new in EDITS:
            if value.count(old) != 1:
                raise ValueError('Unique parent anchor missing: '+old[:70])
            value = value.replace(old, new)
    for old, new in RENAMES:
        value = value.replace(old, new)
    return value


def inverse(value, role):
    for old, new in reversed(RENAMES):
        value = value.replace(new, old)
    if role == 'qk.cu':
        for old, new in reversed(EDITS):
            if value.count(new) != 1:
                raise ValueError('Exact candidate anchor changed: '+new[:70])
            value = value.replace(new, old)
    if sha(value.encode()) != PINS[role]:
        raise ValueError('Non-lookahead edit: '+role)
    return value


def verify():
    files = {}
    for role in FILES:
        value = path(role).read_text()
        if value != generate(role):
            raise ValueError('Generated bytes changed: '+role)
        inverse(value, role)
        files[role] = dict(parent_sha256=PINS[role], variant_sha256=sha(value.encode()),
                           inverse_byte_exact=True)
    return dict(base=BASE, device_source=DEVICE_SOURCE, reference_SO=REFERENCE_SO,
        namespace=NAMESPACE, files=files, scopes=[[64,2,2048],[64,2,2560],[32,4,2048],[32,4,2560]],
        K_words_per_thread=8, shared_bytes=8192, page_bytes=16384,
        read_scope='only later full page of this segment; after original V wait8 + barrier',
        publish_scope='after current PV; before original page-end barrier; next QK after barrier',
        tail_original_default_KReady_false=True, Kpair_or_shared6K_combined=False,
        codec_QK_softmax_PV_reducer_math_unchanged=True,
        actual_CoreX_counter_scope_verified=False, actual_SO_schedule_verified=False,
        device_compile=False, numeric_verified=False, performance_verified=False,
        service_enabled=False, precision_gate=False)


if __name__ == '__main__':
    import argparse
    import json
    p = argparse.ArgumentParser()
    p.add_argument('--write', action='store_true')
    a = p.parse_args()
    if a.write:
        for role in FILES:
            if role != 'codec.cuh':
                path(role).write_text(generate(role))
    print(json.dumps(verify(), sort_keys=True, indent=2))
