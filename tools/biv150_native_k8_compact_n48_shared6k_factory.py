"""Private source-only 6KiB stage fork; no inherited device qualification."""
import hashlib
from pathlib import Path
import re
import subprocess

import biv150_native_k8_compact_n48_factory as parent

ROOT=Path(__file__).resolve().parents[1]
PARENT='e1c5a8fc3c267f1faf9aa790f3c8f80cc65c12a8'
NAMESPACE='cjj_native_k8_compact_n48_shared6k_probe'
FILES={role:parent.path(role).name for role in parent.FILES}
RENAMES=(
 ('cjj_native_k8_compact_n48_s4_probe',NAMESPACE),
 ('cjj_native_k8_compact_n48_producer','cjj_native_k8_compact_n48_shared6k_producer'),
 ('launch_cjj_native_k8_compact_n48_s2_unused','launch_cjj_native_k8_compact_n48_shared6k_s2_unused'),
 ('launch_cjj_native_k8_compact_n48_s4','launch_cjj_native_k8_compact_n48_shared6k_s4'),
 ('cjj_launch_native_k8_compact_n48_codec','cjj_launch_native_k8_compact_n48_shared6k_codec'),
 ('cjj_native_k8_compact_n48_codec_probe','cjj_native_k8_compact_n48_shared6k_codec_probe'))
CUDA_EDITS=(
 ('index=1024+v_half_index(slot,dim)/2','index=512+v_half_index(slot,dim)/2'),
 ('__ivcorex_sme_load_16x1b64_rowxfb16(dst+4096,desc,0,0);\n __ivcorex_sme_load_16x1b64_rowxfb16(dst+5120,desc,64,0);\n __ivcorex_sme_load_16x1b64_rowxfb16(dst+6144,desc,128,0);\n __ivcorex_sme_load_16x1b64_rowxfb16(dst+7168,desc,192,0);',
  '__ivcorex_sme_load_16x1b64_rowxfb16(dst+2048,desc,0,0);\n __ivcorex_sme_load_16x1b64_rowxfb16(dst+3072,desc,64,0);\n __ivcorex_sme_load_16x1b64_rowxfb16(dst+4096,desc,128,0);\n __ivcorex_sme_load_16x1b64_rowxfb16(dst+5120,desc,192,0);'),
 ('load_matrix_sync_tcu(b,stage+1024,n,0)','load_matrix_sync_tcu(b,stage+512,n,0)'),
 ('__shared__ __align__(16) unsigned stage[2048];','__shared__ __align__(16) unsigned stage[1536];'))


def sha(raw):return hashlib.sha256(raw).hexdigest()
def blob(role):return subprocess.check_output(['git','-C',str(ROOT),'show',PARENT+':tools/kernels/'+FILES[role]])
def path(role):return ROOT/'tools/kernels'/(FILES[role] if role=='codec.cuh' else 'biv150_native_k8_compact_n48_shared6k_'+role)


def generate(role):
    text=blob(role).decode()
    if role=='qk.cu':
        for a,b in CUDA_EDITS:
            assert text.count(a)==1 and b not in text;text=text.replace(a,b)
    for a,b in RENAMES:
        assert b not in text;text=text.replace(a,b)
    return text


def inverse(text,role):
    for a,b in reversed(RENAMES):text=text.replace(b,a)
    if role=='qk.cu':
        for a,b in reversed(CUDA_EDITS):assert text.count(b)==1;text=text.replace(b,a)
    assert text.encode()==blob(role),'undeclared source change: '+role
    return text


def verify():
    parent_proof=parent.verify();files={}
    for role in FILES:
        raw=path(role).read_bytes();assert raw==generate(role).encode();inverse(raw.decode(),role)
        files[role]=dict(parent_sha256=sha(blob(role)),candidate_sha256=sha(raw),inverse_byte_exact=True)
    text=path('qk.cu').read_text()
    # The retained BF16 K helpers are NOT compatible with the new V range.
    # They must remain unreachable, not silently interpreted as I8 helpers.
    assert len(re.findall(r'\bclear_k_tail\s*\(',text))==1
    assert len(re.findall(r'\bk_half_index\s*\(',text))==2
    assert len(re.findall(r'\bissue_k_i8\s*\(',text))==2
    assert '__shared__ __align__(16) unsigned stage[1536];' in text
    return dict(parent=PARENT,namespace=NAMESPACE,files=files,parent_source_proof=parent_proof,
      scope=dict(rows=48,segments=4,physical_page_bytes=12544,Q_pitches=[2048,2560]),
      declared_CUDA_edits=[dict(old=a,new=b) for a,b in CUDA_EDITS],
      shared_source_bytes=6144,old_shared_source_bytes=8192,V_base_bytes=2048,K_bytes=2048,V_bytes=4096,
      global_cache_layout_codec_host_scope_math_wait_and_compile_flags_unchanged=True,
      dead_BF16_K_helpers_retained_unreachable=True,dead_BF16_K_helpers_safe_if_called=False,
      actual_library_sha256=None,actual_compiled_shared_bytes=None,hardware_SME_alignment_verified=False,
      compile_verified=False,resources_verified=False,numeric_verified=False,Graph_verified=False,
      performance_verified=False,occupancy_verified=False,service_enabled=False,precision_gate=False)


if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write',action='store_true');a=p.parse_args()
    if a.write:
        for role in FILES:
            if role!='codec.cuh':path(role).write_text(generate(role))
    print(json.dumps(verify(),indent=2,sort_keys=True))
