"""CPU address-only contract. No cache registration, service hook or device import."""
import ast
import hashlib
from pathlib import Path
import subprocess

import biv150_native_k8_source_factory as parent

ROOT = Path(__file__).resolve().parents[1]
MAIN = '4229b48b64c7bd21f9f8fc3c5ff8775500f8d02b'
PARENT = '3ebdbebf61e0e90735cd8ad83eb1605627b6ef46'
COMPACT = 'dc5fd2e5b86a3d57aceaad26e8556d169ec7462a'
FORMAT = 'cjj-konly-compact12544-rn32-rne-v1'
NAMESPACE = 'cjj_native_k8_compact_probe'
PAGE = 12544
FILES = {'qk.cu': 'biv150_native_k8_qk.cu', 'qk.cpp': 'biv150_native_k8_qk.cpp',
         'codec.cu': 'biv150_native_k8_codec.cu'}
RENAMES = (
    ('corex_native_k8_gqa_producer_draft', 'cjj_native_k8_compact_producer'),
    ('launch_corex_native_k8_gqa_s2_draft', 'launch_cjj_native_k8_compact_s2'),
    ('launch_corex_native_k8_gqa_s4_draft', 'launch_cjj_native_k8_compact_s4'),
    ('cjj_launch_native_k8_codec', 'cjj_launch_native_k8_compact_codec'),
    ('cjj_native_k8_codec_probe', 'cjj_native_k8_compact_codec_probe'),
    ('cjj_native_k8_qk_probe', NAMESPACE),
)
EDITS = {
    'qk.cu': (
        ('static_cast<std::uint64_t>(page)*8192+head*128', 'static_cast<std::uint64_t>(page)*6272+head*128'),
        ('static_cast<std::int64_t>(page)*16384+head*128', 'static_cast<std::int64_t>(page)*12544+head*128'),
        ('static_cast<std::int64_t>(page)*4096+1024+col*2+head', 'static_cast<std::int64_t>(page)*3136+1024+col*2+head'),
    ),
    'qk.cpp': (
        ('{16384,256,128,1}', '{12544,256,128,1}'),
        ('"16KiB K code strides only"', '"compact12544 K code strides only"'),
        ('{8192,256,128,1}', '{6272,256,128,1}'),
        ('pages*4096', 'pages*3136'),
        ('kb+8192', 'kb+4352'),
        ('"V must already point to +8192B once"', '"V must already point to +4352B once"'),
        ('pages*16384', 'pages*12544'),
        ('"complete canonical cache allocation"', '"complete compact cache allocation"'),
    ),
    'codec.cu': (),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def blob(commit, path):
    return subprocess.check_output(['git', '-C', str(ROOT), 'show', commit + ':' + path])


def path(role):
    return ROOT / 'tools/kernels' / ('biv150_native_k8_compact_' + role)


def generate(role):
    text = blob(PARENT, 'tools/kernels/' + FILES[role]).decode()
    for old, new in EDITS[role]:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    for old, new in RENAMES:
        text = text.replace(old, new)
    return text


def inverse(text, role):
    for old, new in reversed(RENAMES):
        text = text.replace(new, old)
    for old, new in reversed(EDITS[role]):
        assert text.count(new) == 1, new
        text = text.replace(new, old)
    assert text.encode() == blob(PARENT, 'tools/kernels/' + FILES[role]), 'non-address edit: ' + role
    return text


def addresses(page, slot, head, dim):
    assert type(page) is int and 0 <= page < 2**31-1
    assert type(slot) is int and 0 <= slot < 16
    assert type(head) is int and 0 <= head < 2
    assert type(dim) is int and 0 <= dim < 128
    return (page*PAGE + slot*256 + head*128 + dim,
            (page*3136 + 1024 + slot*2 + head)*4,
            (page*6272 + 2176 + slot*256 + head*128 + dim)*2)


def roles(pages, base, storage_id):
    assert type(pages) is int and 0 < pages < 2**31
    assert type(base) is int and base >= 0 and base % 4 == 0 and base+pages*PAGE < 2**63
    # Offsets are relative to the actual layer data_ptr, not backing storage0.
    return dict(key=dict(dtype='int8', shape=(pages,16,2,128), strides=(PAGE,256,128,1), pointer=base, storage_id=storage_id),
                scale=dict(dtype='float32', shape=(pages*3136,), strides=(1,), pointer=base, storage_id=storage_id),
                value=dict(dtype='bfloat16', shape=(pages,16,2,128), strides=(6272,256,128,1), pointer=base+4352, storage_id=storage_id))


def admit_fixture(receipt, actual, *, epoch):
    """CPU-only future frontend contract; metadata cannot certify cache contents.

    There is deliberately no numerical/service launcher in this commit. A later
    device frontend must construct this receipt from its own live allocation and
    encoder/writer checks, hold the raw storage strongly, and never accept a user
    JSON as a substitute. It must bind this check immediately before dispatch.
    """
    assert type(receipt) is dict and set(receipt) == {
        'format', 'epoch', 'purpose', 'storage_id', 'layer_base', 'pages', 'layer_bytes', 'layout_checked'}
    assert receipt['format'] == FORMAT and receipt['purpose'] == 'Root-owned synthetic component fixture'
    assert type(epoch) is int and epoch > 0
    assert type(receipt['epoch']) is int and receipt['epoch'] == epoch
    assert type(receipt['storage_id']) is int and receipt['storage_id'] > 0
    assert receipt['layout_checked'] is True
    assert receipt['layer_bytes'] == receipt['pages']*PAGE
    def exact(a,b):
        assert type(a) is type(b)
        if type(b) is dict:
            assert set(a) == set(b)
            for key in b:
                exact(a[key],b[key])
        elif type(b) is tuple:
            assert len(a) == len(b)
            for x,y in zip(a,b):
                exact(x,y)
        else:
            assert a == b
    exact(actual, roles(receipt['pages'], receipt['layer_base'], receipt['storage_id']))
    return True


def address_proof():
    coverage = {key: set() for key in ('key', 'scale', 'value')}
    for slot in range(16):
        for head in range(2):
            for dim in range(128):
                k,s,v = addresses(0,slot,head,dim)
                coverage['key'].add(k)
                coverage['scale'].update(range(s,s+4))
                coverage['value'].update((v,v+1))
    assert coverage['key'] == set(range(4096))
    assert coverage['scale'] == set(range(4096,4224))
    assert coverage['value'] == set(range(4352,PAGE))
    assert set(range(PAGE))-set.union(*coverage.values()) == set(range(4224,4352))
    for page in (0,1,32767,2**31-2):
        assert page*PAGE == page*3136*4 == page*6272*2
        assert max(addresses(page,15,1,127)) < (page+1)*PAGE < 2**63
        for head in range(2):
            read = {4352+page*PAGE+head*256+r*512+c+i
                    for r in range(16) for c in (0,64,128,192) for i in range(64)}
            expected = {addresses(page,r,head,d)[2]+i for r in range(16) for d in range(128) for i in (0,1)}
            assert read == expected and len(read) == 4096
    return dict(complete_page_bytes=PAGE, key_bytes=4096, scale_bytes=128, reserved_bytes=128, value_bytes=8192,
                last_value_byte=12543, SME_row_pitch_bytes=512, SME_full_footprint_each_head_bytes=4096,
                page_edge_ids=[0,1,32767,2**31-2], page_arithmetic_signed_I64=True)


def compile_adapter():
    original = blob(PARENT, 'tools/compile_biv150_native_k8_qk.py').decode()
    fn = next(n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef) and n.name == 'compile_only')
    source = ast.get_source_segment(original, fn)
    old = "name='cjj_native_k8_qk_probe'"
    new = 'name=' + repr(NAMESPACE)
    assert source.count(old) == 1
    adapted = source.replace(old, new)
    assert adapted.replace(new,old) == source
    return adapted, dict(parent_compile_body_sha256=sha(source.encode()), adapter_body_sha256=sha(adapted.encode()),
                         namespace_only_inverse_byte_exact=True,
                         standalone_flags_do_not_prove_extension_schedule=True)


def verify():
    inherited = parent.verify()
    files = {}
    for role in FILES:
        text = path(role).read_text()
        assert text == generate(role)
        restored = inverse(text,role)
        if role == 'qk.cu':
            assert parent.inverse(restored).encode() == parent.SOURCE.read_bytes()
        files[role] = dict(parent_sha256=sha(restored.encode()), variant_sha256=sha(text.encode()), inverse_byte_exact=True)
    pins = {}
    for name in ('views','writer','spec','runtime'):
        file = 'vllm_fl/patches/biv150_konly_compact_' + name + '.py'
        pins[file] = sha(blob(COMPACT,file))
    return dict(main=MAIN, original_parent=PARENT, compact_reference=COMPACT, format=FORMAT, namespace=NAMESPACE,
                parent_source_proof=inherited, files=files, compact_source_pins=pins, address_proof=address_proof(),
                device_address_edits=3, host_typed_layout_edits=5,
                shared_bytes=8192, shared_K_bytes=[0,2048], shared_V_bytes=[4096,8192],
                wait_tail_Qcodec_QK_BF16P_PV_reducer_math_unchanged=True, V_preoffset_bytes=4352,
                scope=['N64/S2','N32/S4'], N40_S4_supported=False, N40_policy_S16_unchanged=True,
                actual_compile_verified=False, actual_SO_IR_SASS_verified=False, numeric_bitexact_verified=False,
                actual_epoch_binding=False, service_installed=False, capacity_reclaimed=False, performance_verified=False)
