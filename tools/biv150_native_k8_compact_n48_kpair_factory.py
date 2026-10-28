"""Private K pair-read source fork; device alignment/ISA/numerics are pending."""
import hashlib
from pathlib import Path
import subprocess

import biv150_native_k8_compact_n48_factory as parent

ROOT = Path(__file__).resolve().parents[1]
PARENT = 'e1c5a8fc3c267f1faf9aa790f3c8f80cc65c12a8'
NAMESPACE = 'cjj_native_k8_compact_n48_kpair_probe'
FILES = {role: parent.path(role).name for role in parent.FILES}
RENAMES = (
    ('cjj_native_k8_compact_n48_s4_probe', NAMESPACE),
    ('cjj_native_k8_compact_n48_producer', 'cjj_native_k8_compact_n48_kpair_producer'),
    ('launch_cjj_native_k8_compact_n48_s2_unused', 'launch_cjj_native_k8_compact_n48_kpair_s2_unused'),
    ('launch_cjj_native_k8_compact_n48_s4', 'launch_cjj_native_k8_compact_n48_kpair_s4'),
    ('cjj_launch_native_k8_compact_n48_codec', 'cjj_launch_native_k8_compact_n48_kpair_codec'),
    ('cjj_native_k8_compact_n48_codec_probe', 'cjj_native_k8_compact_n48_kpair_codec_probe'),
)
SCALAR_LOOP = ''' for(unsigned word=lane;word<512;word+=64){
  const unsigned slot=word/32,dim=(word%32)*4;
  unsigned bits=0;
  if(slot<valid)bits=*reinterpret_cast<const unsigned*>(base+slot*256+dim);
  stage[k_i8_index(slot,dim)/4]=bits;
 }'''
PAIR_LOOP = ''' // Private K-only transport change. Global page/slot/head strides stay original.
 static_assert(sizeof(uint2)==8 && alignof(uint2)==8 &&
   __builtin_offsetof(uint2,x)==0 && __builtin_offsetof(uint2,y)==4,"uint2 ABI changed");
 // Uniform within the original one-warp block. The host only guarantees 4B.
 if((reinterpret_cast<std::uintptr_t>(base)&7u)==0){
  #pragma unroll
  for(unsigned pair=lane;pair<256;pair+=64){
   const unsigned word=pair*2,slot=word/32,dim=(word%32)*4;
   uint2 bits{0u,0u};
   if(slot<valid)bits=*reinterpret_cast<const uint2*>(base+slot*256+dim);
   stage[k_i8_index(slot,dim)/4]=bits.x;
   stage[k_i8_index(slot,dim+4)/4]=bits.y;
  }
 }else{
 // Preserve the exact original scalar loop for 4B-but-not-8B base alignment.
''' + SCALAR_LOOP + '''
 }'''
COMPILE_EDITS = (
    ('import biv150_native_k8_compact_n48_factory as factory',
     'import biv150_native_k8_compact_n48_kpair_factory as factory'),
    ("name='cjj_native_k8_compact_n48_s4_probe'", "name='" + NAMESPACE + "'"),
    ("requested='deferred-V-wait compile-only; no device qualification'",
     "requested='K pair-read compile-only; actual ISA/alignment/numerics pending'"),
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(role):
    return subprocess.check_output(['git', '-C', str(ROOT), 'show',
                                   PARENT + ':tools/kernels/' + FILES[role]])


def path(role):
    name = FILES[role] if role == 'codec.cuh' else 'biv150_native_k8_compact_n48_kpair_' + role
    return ROOT / 'tools/kernels' / name


def generate(role):
    text = blob(role).decode()
    if role == 'qk.cu':
        assert text.count(SCALAR_LOOP) == 1 and PAIR_LOOP not in text
        text = text.replace(SCALAR_LOOP, PAIR_LOOP)
    for old, new in RENAMES:
        assert new not in text
        text = text.replace(old, new)
    return text


def inverse(text, role):
    for old, new in reversed(RENAMES):
        text = text.replace(new, old)
    if role == 'qk.cu':
        assert text.count(PAIR_LOOP) == 1
        text = text.replace(PAIR_LOOP, SCALAR_LOOP)
    assert text.encode() == blob(role), 'undeclared source change: ' + role
    return text


def compile_source():
    text = subprocess.check_output(['git', '-C', str(ROOT), 'show',
                                   PARENT + ':tools/compile_biv150_native_k8_compact_n48.py']).decode()
    for old, new in COMPILE_EDITS:
        assert text.count(old) == 1 and new not in text
        text = text.replace(old, new)
    return text


def verify():
    parent_proof = parent.verify()
    files = {}
    for role in FILES:
        raw = path(role).read_bytes()
        assert raw == generate(role).encode()
        inverse(raw.decode(), role)
        files[role] = dict(parent_sha256=sha(blob(role)), candidate_sha256=sha(raw),
                           inverse_byte_exact=True)
    cuda = path('qk.cu').read_text()
    assert cuda.count(SCALAR_LOOP) == 1 and cuda.count(PAIR_LOOP) == 1
    assert '__shared__ __align__(16) unsigned stage[2048];' in cuda
    assert 'base=key+static_cast<std::int64_t>(page)*12544+head*128;' in cuda
    assert (ROOT / 'tools/compile_biv150_native_k8_compact_n48_kpair.py').read_text() == compile_source()
    return dict(parent=PARENT, namespace=NAMESPACE, files=files,
        parent_source_proof=parent_proof, CUDA_edit_scope='issue_k_i8 loop only, plus private names',
        scalar_loop_byte_exact=True, compile_frontend_inverse_byte_exact=True,
        scope=dict(rows=48, segments=4, Q_pitches=[2048, 2560], physical_page_bytes=12544),
        selected_by_actual_base_low3_bits=True, scalar_path_base_mod8=4, pair_path_base_mod8=0,
        shared_source_bytes=8192, host_ABI_4B_alignment_unchanged=True,
        source_global_K_bytes_per_head_page=2048, unique_shared_K_words=512,
        V_wait_codec_math_writer_reducer_and_layout_unchanged=True,
        original_entire_cache_4B_non8B_device_support_known=False,
        no_PR113_shared_relocation=True, no_PR103_Q_hoist=True,
        actual_library_sha256=None, actual_compiled_resources=None,
        compile_verified=False, actual_vector_load_lowering_verified=False,
        numeric_verified=False, Graph_verified=False, side_stream_verified=False,
        performance_verified=False, occupancy_verified=False, service_enabled=False,
        precision_gate=False)


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    if args.write:
        for role in FILES:
            if role != 'codec.cuh':
                path(role).write_text(generate(role))
        (ROOT / 'tools/compile_biv150_native_k8_compact_n48_kpair.py').write_text(compile_source())
    print(json.dumps(verify(), indent=2, sort_keys=True))
