"""Private N48/S4 host-scope fork of PR110; device math/codec/wait are exact."""
import hashlib
from pathlib import Path
import subprocess

import biv150_native_k8_compact_v_overlap_factory as parent

ROOT=Path(__file__).resolve().parents[1]
PARENT='4d1166792a5247b774d6b1977e60466574e6a34c'
NAMESPACE='cjj_native_k8_compact_n48_s4_probe'
FILES={
 'qk.cpp':'biv150_native_k8_compact_v_overlap_qk.cpp',
 'qk.cu':'biv150_native_k8_compact_v_overlap_qk.cu',
 'codec.cu':'biv150_native_k8_compact_v_overlap_codec.cu',
 'codec.cuh':'biv150_native_k8_codec.cuh',
}
PINS={
 'qk.cpp':'3c396691f89f7173ee86302df916dff07f84593fdedc49de10b18066fe1d7750',
 'qk.cu':'99a3f0a9868ed08d260956bef39315b99face6b1069ee23d8ebf6e58f6f1a3ce',
 'codec.cu':'871861f0a272260dd12f3eb1ca7a2f2740a43eee6702918956fa9dd14dbbe6f1',
 'codec.cuh':'37d5c7548c0301ee522f50fb0d4d2da53300676e71b03bb806c2d88779801d79',
}
RENAMES=(
 ('cjj_native_k8_compact_v_overlap_producer','cjj_native_k8_compact_n48_producer'),
 ('launch_cjj_native_k8_compact_v_overlap_s2','launch_cjj_native_k8_compact_n48_s2_unused'),
 ('launch_cjj_native_k8_compact_v_overlap_s4','launch_cjj_native_k8_compact_n48_s4'),
 ('cjj_launch_native_k8_compact_v_overlap_codec','cjj_launch_native_k8_compact_n48_codec'),
 ('cjj_native_k8_compact_v_overlap_codec_probe','cjj_native_k8_compact_n48_codec_probe'),
 ('cjj_native_k8_compact_v_overlap_probe',NAMESPACE),
)
CPP_EDITS=(
 (' TORCH_CHECK(segments==2 || segments==4,"S2/S4 only");',
  ' TORCH_CHECK(segments==4,"Private N48/S4 only");'),
 (' const std::int64_t rows=segments==2?64:32;',
  ' const std::int64_t rows=48;'),
)


def sha(raw):return hashlib.sha256(raw).hexdigest()
def blob(relative):return subprocess.check_output(['git','-C',str(ROOT),'show',PARENT+':'+relative])
def original(role):
    raw=blob('tools/kernels/'+FILES[role]);assert sha(raw)==PINS[role]
    return raw.decode()
def path(role):
    return ROOT/'tools/kernels'/(FILES[role] if role=='codec.cuh' else 'biv150_native_k8_compact_n48_'+role)


def generate(role):
    text=original(role)
    if role=='qk.cpp':
        for old,new in CPP_EDITS:
            assert text.count(old)==1 and new not in text; text=text.replace(old,new)
    for old,new in RENAMES:
        assert new not in text
        text=text.replace(old,new)
    return text


def inverse(text,role):
    for old,new in reversed(RENAMES):text=text.replace(new,old)
    if role=='qk.cpp':
        for old,new in reversed(CPP_EDITS):
            assert text.count(new)==1;text=text.replace(new,old)
    assert text.encode()==blob('tools/kernels/'+FILES[role]),'out-of-scope source change: '+role
    return text


def verify():
    parent_proof=parent.verify();proof={}
    for role in FILES:
        actual=path(role).read_text();assert actual==generate(role)
        restored=inverse(actual,role)
        assert (ROOT/'tools/kernels'/FILES[role]).read_text()==restored
        proof[role]=dict(parent_sha256=PINS[role],candidate_sha256=sha(actual.encode()),
                         inverse_byte_exact=True,changes='private symbols only' if role!='qk.cpp' else 'private symbols and two exact host-scope lines')
    cuda=path('qk.cu').read_text()
    assert 'seq=blockIdx.x' in cuda and 'dim3(batch,2,4)' in cuda
    assert 'cjj_native_k8_compact_n48_producer<4>' in cuda
    return dict(parent=PARENT,namespace=NAMESPACE,files=proof,
      parent_wait_proof=parent_proof,deferred_V_wait_preserved=True,device_math_codec_strides_byte_exact_after_symbol_inverse=True,
      source_scope=dict(rows=48,segments=4,Q_pitches=[2048,2560],physical_page_bytes=12544,scale_full_page0=True,V_offset_bytes=4352),
      unused_S2_device_symbol_retained_for_exact_inverse=True,S2_host_entry_rejected=True,
      compile_flags_unchanged=True,shared_stage_source_bytes=8192,
      actual_library_sha256=None,actual_compile_verified=False,actual_resource_or_schedule_verified=False,
      numeric_verified=False,Graph_verified=False,side_stream_verified=False,performance_verified=False,
      service_enabled=False,precision_gate=False)


if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write',action='store_true');a=p.parse_args()
    if a.write:
        for role in FILES:
            if role!='codec.cuh':path(role).write_text(generate(role))
    print(json.dumps(verify(),indent=2,sort_keys=True))
