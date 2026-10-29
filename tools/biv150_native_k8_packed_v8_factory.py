"""Single N64/S2 compile scout: exact 5898 inverse except private V operand path."""
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = '899ce77470b0b539aebc108fd122f668c2d5894d'
REFERENCE_SO = '5898a00a4a4e3314ee8ed1fc40dffde02a766ca6d0e5c116dbd161dccaa3286b'
NAMESPACE = 'cjj_native_k8_packed_v8_probe'
PARENT = {'qk.cu': 'tools/kernels/biv150_native_k8_klookahead_qk.cu',
          'qk.cpp': 'tools/kernels/biv150_native_k8_klookahead_qk.cpp',
          'codec.cuh': 'tools/kernels/biv150_native_k8_codec.cuh',
          'frontend': 'tools/compile_biv150_native_k8_klookahead.py'}
PINS = {'qk.cu': 'a491757fff161436a323305f75c3223d12d2d91bd8420b2332e633012f72dd4c',
        'qk.cpp': '8b672b6f1d02ec3420eaa11186735f10c7cb235d083b1379275fe227c26a1ed9',
        'codec.cuh': '37d5c7548c0301ee522f50fb0d4d2da53300676e71b03bb806c2d88779801d79',
        'frontend': '382451160caf44076fee2890f699a76f19708d5d4089fbbbd6806c56b7f289a0'}

V_HELPERS = '''// Private V8 role: base already +8192B; page pitch remains 16384B.
// Only packed words are requested to stay live across QK/P packing.
struct PackedV { unsigned word[8]; };
__device__ __forceinline__ void read_packed_v(PackedV& packed,const signed char* value,int page,unsigned head,unsigned valid,unsigned lane){
 const auto* base=value+static_cast<std::int64_t>(page)*16384+head*128;
 #pragma unroll
 for(unsigned i=0;i<4;++i){
  const unsigned w=lane+i*64,slot=2*(w/32),dim=4*(w%32);
  packed.word[2*i]=slot<valid?*reinterpret_cast<const unsigned*>(base+slot*256+dim):0u;
  packed.word[2*i+1]=slot+1<valid?*reinterpret_cast<const unsigned*>(base+(slot+1)*256+dim):0u;
 }
}
__device__ __forceinline__ __nv_bfloat16 decode_v_byte(unsigned bits,unsigned byte,float scale){
 const int code=(static_cast<int>((bits>>(8*byte))&255u)^128)-128;
 return __float2bfloat16_rn(static_cast<float>(code)*scale);
}
__device__ __forceinline__ void publish_packed_v(unsigned* stage,const PackedV& packed,const float* scale_full,int page,unsigned head,unsigned valid,unsigned lane){
 // Exactly one F32 scale load per valid slot/head; every lane shuffles.
 const float own_scale=lane<valid?scale_full[static_cast<std::int64_t>(page)*4096+3072+lane*2+head]:1.0f;
 #pragma unroll
 for(unsigned i=0;i<4;++i){
  const unsigned w=lane+i*64,slot=2*(w/32),dim=4*(w%32);
  const float low_scale=shuffle(own_scale,slot),high_scale=shuffle(own_scale,slot+1);
  #pragma unroll
  for(unsigned j=0;j<4;++j){
   const auto low=decode_v_byte(packed.word[2*i],j,low_scale);
   const auto high=decode_v_byte(packed.word[2*i+1],j,high_scale);
   // Adjacent slots, same D: one owner of both halves. No b16 RMW/byte stage.
   stage[1024+v_half_index(slot,dim+j)/2]=pair_bits(low,high);
  }
 }
}
'''


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def original(role):
    raw = subprocess.check_output(['git', '-C', str(ROOT), 'show', BASE+':'+PARENT[role]])
    if sha(raw) != PINS[role]:
        raise ValueError('Pinned main899 source changed: '+role)
    return raw.decode()


def path(role):
    return ROOT/(PARENT[role] if role == 'codec.cuh'
                 else 'tools/kernels/biv150_native_k8_packed_v8_'+role)


def edits(role):
    parent = original(role)
    if role == 'qk.cu':
        issue = parent[parent.index('__device__ __forceinline__ void issue_v('):
                       parent.index('\ntemplate<bool OverlapV,bool KReady=false>')]
        s4 = parent[parent.index('\nextern "C" void launch_cjj_native_k8_klookahead_s4('):]
        return [
            (issue, V_HELPERS),
            ('const float* key_scale_full,const __nv_bfloat16* value,',
             'const float* key_scale_full,const signed char* value,'),
            (' const unsigned dst=static_cast<unsigned>(reinterpret_cast<std::uintptr_t>(stage));\n', ''),
            (' // V keeps SME issue/layout; its completion wait moves to the PV consumer.',
             ' // V8 replaces BF16 SME issue; PV shared layout remains unchanged.'),
            (' issue_v(dst,descriptor(value,page,head));',
             ' PackedV packed_v;read_packed_v(packed_v,value,page,head,valid,lane);'),
            ('  // Deferred V wait: the unchanged PV consumer must see completed V.\n'
             '  // Actual compiler may force this wait earlier; LLVM/ASM must decide.\n'
             '  __ivcorex_sl_waitcnt(8ULL);',
             '  // Kept only as a scout instruction, NOT proof of ordinary-load completion.\n'
             '  // No V SME task exists now: final-SO counter safety is an open gate.\n'
             '  __ivcorex_sl_waitcnt(8ULL);\n'
             '  publish_packed_v(stage,packed_v,key_scale_full,page,head,valid,lane);'),
            ('  if(valid<16)clear_v_tail(stage,valid,lane);',
             '  // Publisher owns both halves and already zeroes every invalid slot.\n'
             '  // Do not run the old differently-owned shared tail RMW before this barrier.'),
            ('static_assert(Segments==2 || Segments==4,"Only separately qualified S2/S4 planned");',
             'static_assert(Segments==2,"Private V8 scout is N64/S2 only");'),
            ('// First draft exposes only S2; S4 remains the same source template, not an\n'
             '// enabled/qualified path. A future R32 component must instantiate/validate S4.',
             '// Only N64/S2 is instantiated. No S4 or production entry is provided.'),
            (s4, ''),
            ('static_cast<const __nv_bfloat16*>(value),seq_lens,block_table,scale,',
             'static_cast<const signed char*>(value),seq_lens,block_table,scale,'),
        ]
    if role == 'qk.cpp':
        s4 = parent[parent.index('extern "C" void launch_cjj_native_k8_klookahead_s4('):
                    parent.index('extern "C" void cjj_launch_native_k8_klookahead_codec(')]
        codec_decl = 'extern "C" void cjj_launch_native_k8_klookahead_codec(const void*,void*,float*,int,cudaStream_t);\n'
        codec = parent[parent.index('void codec('):parent.index('\n}\nTORCH_LIBRARY')]
        return [(s4, ''), (codec_decl, ''),
            ('TORCH_CHECK(segments==2 || segments==4,"S2/S4 only");',
             'TORCH_CHECK(segments==2,"Private V8 scout N64/S2 only");'),
            ('const std::int64_t rows=segments==2?64:32;', 'const std::int64_t rows=64;'),
            ('v.scalar_type()==at::kBFloat16 && v.sizes()==k.sizes() && v.strides()==at::IntArrayRef({8192,256,128,1}),"already-offset original BF16 V role"',
             'v.scalar_type()==at::kChar && v.sizes()==k.sizes() && v.strides()==at::IntArrayRef({16384,256,128,1}),"private V8 +8192B role; BF16/compact forbidden"'),
            (' // per-token/head scales must be finite. This is NOT a live service entry.',
             ' // K scales and V scales at byte12288/token/head must be finite positive.\n'
             ' // V codes/scale are privately prepared; no V writer or live service entry is certified.'),
            ('Launch launch=segments==2?launch_cjj_native_k8_klookahead_s2:launch_cjj_native_k8_klookahead_s4;',
             'Launch launch=launch_cjj_native_k8_klookahead_s2;'),
            (codec, ''),
            (' m.def("codec(Tensor input, Tensor(a!) codes, Tensor(b!) scales) -> ()");\n', ''),
            ('m.impl("produce",&produce);m.impl("codec",&codec);', 'm.impl("produce",&produce);')]
    return []


RENAMES = (('cjj_native_k8_klookahead_gqa_producer', 'cjj_native_k8_packed_v8_gqa_producer'),
           ('launch_cjj_native_k8_klookahead_s2', 'launch_cjj_native_k8_packed_v8_s2'),
           ('cjj_native_k8_klookahead_probe', NAMESPACE))


def transform(value, changes, reverse=False):
    for old, new in (reversed(changes) if reverse else changes):
        a, b = (new, old) if reverse else (old, new)
        if not a or value.count(a) != 1:
            raise ValueError('Unique transform anchor missing: '+repr(a[:70]))
        value = value.replace(a, b)
    return value


def generate(role):
    value = original(role)
    # Several ABI type tokens occur twice; these two replacements are explicit.
    for old, new in edits(role):
        count = 2 if role == 'qk.cu' and old.startswith('const float* key_scale_full,') else 1
        if value.count(old) != count:
            raise ValueError('Parent anchor/count changed: '+repr(old[:70]))
        value = value.replace(old, new)
    for old, new in RENAMES:
        value = value.replace(old, new)
    return value


def verify():
    files = {}
    for role in ('qk.cu', 'qk.cpp', 'codec.cuh'):
        value = path(role).read_text()
        if value != generate(role):
            raise ValueError('Candidate source differs from finite parent transform: '+role)
        # Reverse deletions at their original offsets, not unconstrained replace('',...).
        stages = [original(role)]
        for old, new in edits(role):
            stages.append(stages[-1].replace(old, new))
        inverse = value
        for old, new in reversed(RENAMES):
            inverse = inverse.replace(new, old)
        for index in reversed(range(len(edits(role)))):
            old, new = edits(role)[index]
            before = stages[index]
            if new:
                inverse = inverse.replace(new, old)
            else:
                position = before.index(old)
                inverse = inverse[:position]+old+inverse[position:]
            if inverse != before:
                raise ValueError('Exact inverse failed: '+role)
        if sha(inverse.encode()) != PINS[role]:
            raise ValueError('Parent identity changed: '+role)
        files[role] = dict(parent_sha256=PINS[role], candidate_sha256=sha(value.encode()), inverse_byte_exact=True)
    return dict(base=BASE, reference_SO=REFERENCE_SO, namespace=NAMESPACE, files=files,
        scope=dict(N=64,S=2,Q_pitches=[2048,2560],page_bytes=16384,shared_bytes=8192),
        contract='Vop=BF16_RNE(F32(signed_I8_code * F32_scale)); P remains BF16_RNE(p)',
        K_Q_codec_QK_softmax_PV_scratch_source_unchanged=True,
        V_SME_replaced_by_ordinary_packed_reads=True,
        V_writer_implemented=False, V_codec_or_model_precision_verified=False,
        actual_wait8_counter_safety_verified=False, actual_late_unpack_verified=False,
        actual_resources=None, device_compile=False, numeric_verified=False,
        performance_verified=False, service_enabled=False, capacity_recovered=False)


def frontend_source():
    value = original('frontend')
    review_start = value.index('        both_S2_S4_full_next_last_and_original_tail_audited=False,')
    review_end = value.index('resources={\'S2\':None,\'S4\':None},numeric_verified=False,performance_verified=False))')
    old_review = value[review_start:review_end]
    changes = [
        ('import biv150_native_k8_klookahead_factory as factory', 'import biv150_native_k8_packed_v8_factory as factory'),
        ("('qk.cpp', 'qk.cu', 'codec.cu')", "('qk.cpp', 'qk.cu')"),
        ('    return dict(source_transform=factory.verify(), explicit_cuda_flags=CUDA_FLAGS,',
         '    return dict(source_transform=factory.verify(), explicit_cuda_flags=CUDA_FLAGS,\n'
         '        frontend_sha256=sha(__file__), factory_sha256=sha(factory.__file__),\n'
         '        generated_frontend_sha256=factory.sha(factory.frontend_source().encode()),'),
        ("requested='only packed K-next read after original V wait+barrier; publish after current PV'",
         "requested='N64/S2 private packed-V8 late decode and paired BF16 shared stores; no launches'"),
        ("    subprocess.run(['git','-C',str(factory.ROOT),'merge-base','--is-ancestor',factory.BASE,commit],check=True)",
         "    if commit != args.expected_source_commit:\n        raise ValueError('Expected exact frozen source commit')\n    subprocess.run(['git','-C',str(factory.ROOT),'merge-base','--is-ancestor',factory.BASE,commit],check=True)"),
        ("library = load(name='cjj_native_k8_klookahead_probe'", "library = load(name='cjj_native_k8_packed_v8_probe'"),
        (old_review, '        S2_full_tail_empty_and_Klookahead_audited=False,\n'
         '        no_V_SME_tasks_and_retained_wait8_counter_safety_verified=False,\n'
         '        packed_V_live_until_P_pack_then_dependent_pair_stores_verified=False,\n'
         '        pair_store_barrier_dominates_PV_and_Knext_publish_verified=False,\n'
         '        Q_K_codec_softmax_PV_unchanged_new_V_operand_verified=False,\n'
         '        no_b16_byte_stage_extra_barrier_or_spill_verified=False,\n        '),
        ("resources={'S2':None,'S4':None}", "resources={'S2':None}"),
        ("    p.add_argument('--out',type=Path)", "    p.add_argument('--out',type=Path)\n    p.add_argument('--expected-source-commit')"),
        ('        if a.out or a.root_owned_exclusive:', '        if a.out or a.root_owned_exclusive or a.expected_source_commit:'),
        ('    if a.out is None or not a.out.is_absolute() or a.out.exists():',
         "    if not a.expected_source_commit or len(a.expected_source_commit)!=40 or any(c not in '0123456789abcdef' for c in a.expected_source_commit):\n        p.error('Explicit full expected source commit required')\n    if a.out is None or not a.out.is_absolute() or a.out.exists():"),
    ]
    generated = transform(value, changes)
    if transform(generated, changes, reverse=True) != value:
        raise ValueError('Compile frontend inverse changed')
    return generated
