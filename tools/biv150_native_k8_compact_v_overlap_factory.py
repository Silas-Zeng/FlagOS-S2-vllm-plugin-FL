"""One reversible PR106 wait move applied to the unchanged compact12544 reader."""
import ast
import hashlib
from pathlib import Path
import subprocess

import biv150_native_k8_compact_factory as compact
import biv150_native_k8_v_overlap_factory as overlap

ROOT = Path(__file__).resolve().parents[1]
BASE = 'a455a3de364051fbea80226870c239de3a48e716'
EVIDENCE_BASE = '46b44a1326973ccb6b874452841726298507414c'
PARENT = '51da6fd73e7fff1513081fb330323ce72c00856c'
WAIT_PARENT = 'a1eb7deee5e8ef99136128984256762096a3657f'
WRITER_PARENT = 'dc5fd2e5b86a3d57aceaad26e8556d169ec7462a'
NAMESPACE = 'cjj_native_k8_compact_v_overlap_probe'
FILES = {'qk.cu':'biv150_native_k8_compact_qk.cu',
         'qk.cpp':'biv150_native_k8_compact_qk.cpp',
         'codec.cu':'biv150_native_k8_compact_codec.cu',
         'codec.cuh':'biv150_native_k8_codec.cuh'}
PARENT_SHA = {
    'qk.cu':'3bd67edd58f1ea822a908dcc9217aceeef93124a7e9e00e6d7815dcdffe7a81d',
    'qk.cpp':'ca74d8d1234c9fcfa4178ec111d139dc045f13a2a70f078f4ba86a2f307c8761',
    'codec.cu':'d713a00547cfb337ff32413ca6e92e3d8b010d6d82f277ee78005c2f3f82e14b',
    'codec.cuh':'37d5c7548c0301ee522f50fb0d4d2da53300676e71b03bb806c2d88779801d79',
}
RENAMES = (
    ('cjj_native_k8_compact_producer','cjj_native_k8_compact_v_overlap_producer'),
    ('launch_cjj_native_k8_compact_s2','launch_cjj_native_k8_compact_v_overlap_s2'),
    ('launch_cjj_native_k8_compact_s4','launch_cjj_native_k8_compact_v_overlap_s4'),
    ('cjj_launch_native_k8_compact_codec','cjj_launch_native_k8_compact_v_overlap_codec'),
    ('cjj_native_k8_compact_codec_probe','cjj_native_k8_compact_v_overlap_codec_probe'),
    ('cjj_native_k8_compact_probe',NAMESPACE),
)
# Use the exact six edits that produced the reviewed8b source, including its
# existing late consumer branch. No independent wait/synchronization protocol.
EDITS = overlap.EDITS


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(commit,name):
    return subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+name])


def original(role):
    raw=blob(PARENT,'tools/kernels/'+FILES[role])
    assert sha(raw)==PARENT_SHA[role]
    return raw.decode()


def path(role):
    return ROOT/'tools/kernels'/(FILES[role] if role=='codec.cuh' else 'biv150_native_k8_compact_v_overlap_'+role)


def generate(role):
    text=original(role)
    if role=='qk.cu':
        for old,new in EDITS:
            assert text.count(old)==1,old
            text=text.replace(old,new)
    for old,new in RENAMES:text=text.replace(old,new)
    return text


def inverse(text,role):
    for old,new in reversed(RENAMES):text=text.replace(new,old)
    if role=='qk.cu':
        for old,new in reversed(EDITS):
            assert text.count(new)==1,new
            text=text.replace(new,old)
    assert sha(text.encode())==PARENT_SHA[role],'out-of-scope change: '+role
    return text


def to_reviewed8b(text,role):
    """The two independent source edits commute: compact addresses vs V wait."""
    for old,new in reversed(RENAMES):text=text.replace(new,old)
    for old,new in reversed(compact.RENAMES):text=text.replace(new,old)
    for old,new in reversed(compact.EDITS[role]):
        assert text.count(new)==1,new
        text=text.replace(new,old)
    for old,new in overlap.RENAMES:text=text.replace(old,new)
    reviewed=blob(WAIT_PARENT,'tools/kernels/biv150_native_k8_v_overlap_'+role)
    assert text.encode()==reviewed,'does not commute with reviewed wait move: '+role
    return sha(reviewed)


def verify():
    frozen=ast.parse(blob(WAIT_PARENT,'tools/biv150_native_k8_v_overlap_factory.py').decode())
    recipe=next(n.value for n in frozen.body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='EDITS' for t in n.targets))
    assert EDITS==ast.literal_eval(recipe)
    rows={}
    for role in FILES:
        actual=path(role).read_text()
        assert actual==generate(role)
        restored=inverse(actual,role)
        # Original compact sources and the common codec header also remain intact.
        assert (ROOT/'tools/kernels'/FILES[role]).read_text()==restored
        rows[role]=dict(parent_sha256=sha(restored.encode()),variant_sha256=sha(actual.encode()),
                        inverse_byte_exact=True)
        if role!='codec.cuh':rows[role]['commutes_to_reviewed8b_sha256']=to_reviewed8b(actual,role)
    writer_pins={}
    for role,expected in (
        ('writer','a0cacf3f13f450f1f633d0bf562b3b67239cdc4b4e2a9b1159bcbde4b3fa290f'),
        ('views','1ab738e4b8178ee0d8ce12536ac7e6428ed09f7252ea7349c8d71f73bae64b1e'),
    ):
        name='vllm_fl/patches/biv150_konly_compact_'+role+'.py'
        actual=blob(WRITER_PARENT,name);assert sha(actual)==expected
        writer_pins[name]=sha(actual)
    return dict(base=BASE,evidence_base=EVIDENCE_BASE,parent=PARENT,wait_parent=WAIT_PARENT,
        namespace=NAMESPACE,files=rows,dc5_writer_reference=dict(commit=WRITER_PARENT,source_pins=writer_pins,writer_in_this_component=False),
        issue_order='K ordinary load/store + barrier, then V SME (unchanged)',
        deferred_wait='after QK and P softmax/packing; before V tail clear and PV',
        physical_page_bytes=12544,address_proof=compact.address_proof(),
        shared_bytes=8192,shared_K_bytes=[0,2048],shared_V_bytes=[4096,8192],
        scope=['N64/S2','N32/S4'],Q_pitches=[2048,2560],N40_N48_added=False,
        PR103_Q32_hoist_combined=False,V_first_PR96_combined=False,
        Q_quantization_QK_softmax_PV_and_outputs_unchanged=True,
        actual_library_sha256=None,actual_device_schedule_verified=False,numeric_verified=False,
        performance_verified=False,service_enabled=False,precision_gate=False)


if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write',action='store_true');a=p.parse_args()
    if a.write:
        for role in FILES:
            if role!='codec.cuh':path(role).write_text(generate(role))
    print(json.dumps(verify(),indent=2,sort_keys=True))
