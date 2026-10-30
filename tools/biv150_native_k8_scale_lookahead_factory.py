"""Finite 5898 inverse: carry only the next complete page's K scale with K."""
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = '5f0086be50f1d6d115c1ebb25b9d3c7250470372'
REFERENCE_SO = '5898a00a4a4e3314ee8ed1fc40dffde02a766ca6d0e5c116dbd161dccaa3286b'
NAMESPACE = 'cjj_native_k8_scale_lookahead_probe'
FILES = ('qk.cu', 'qk.cpp', 'codec.cu', 'codec.cuh')
PARENT = {r: 'tools/kernels/biv150_native_k8_klookahead_'+r for r in FILES}
PARENT['codec.cuh'] = 'tools/kernels/biv150_native_k8_codec.cuh'
PARENT['frontend'] = 'tools/compile_biv150_native_k8_klookahead.py'
PINS = {
    'qk.cu': 'a491757fff161436a323305f75c3223d12d2d91bd8420b2332e633012f72dd4c',
    'qk.cpp': '8b672b6f1d02ec3420eaa11186735f10c7cb235d083b1379275fe227c26a1ed9',
    'codec.cu': '783d7b5e78c8b0c5c19cc3ea0b588b27a7e869997f98e920c9470a74cd4f809e',
    'codec.cuh': '37d5c7548c0301ee522f50fb0d4d2da53300676e71b03bb806c2d88779801d79',
    'frontend': '382451160caf44076fee2890f699a76f19708d5d4089fbbbd6806c56b7f289a0',
}
RENAMES = (('native_k8_klookahead', 'native_k8_scale_lookahead'),)
EDITS = (
    ('int next_position=0){', 'int next_position=0,float* scale_carry=nullptr){'),
    (' const float kscale=col<valid?key_scale_full[static_cast<std::int64_t>(page)*4096+1024+col*2+head]:1.0f;',
     ' // Only KReady full pages receive a local carry. The original tail has no carry.\n'
     ' const float kscale=KReady?*scale_carry:(col<valid?key_scale_full[static_cast<std::int64_t>(page)*4096+1024+col*2+head]:1.0f);'),
    (' int next_page=0;\n PackedNextK next;',
     ' int next_page=0;\n float next_kscale=1.0f;\n PackedNextK next;'),
    ('  read_next_k(next,key,next_page,head,lane);\n }',
     '  read_next_k(next,key,next_page,head,lane);\n'
     '  next_kscale=key_scale_full[static_cast<std::int64_t>(next_page)*4096+1024+col*2+head];\n }'),
    (' if(has_next)publish_next_k(stage,next,lane); // Current QK and PV readers have ended.',
     ' if(has_next){\n'
     '  publish_next_k(stage,next,lane); // Current QK and PV readers have ended.\n'
     '  *scale_carry=next_kscale; // Private per-thread carry; no extra shared publication.\n'
     ' }'),
    ('  int page=row[position/16];\n  issue_k_i8(',
     '  int page=row[position/16];\n'
     '  float kscale=key_scale_full[static_cast<std::int64_t>(page)*4096+1024+col*2+h];\n'
     '  issue_k_i8('),
    ('acc,has_next,row,next_position);', 'acc,has_next,row,next_position,&kscale);'),
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def original(role):
    raw = subprocess.check_output(['git', '-C', str(ROOT), 'show', BASE+':'+PARENT[role]])
    if sha(raw) != PINS[role]:
        raise ValueError('Pinned main5f/5898 source changed: '+role)
    return raw.decode()


def path(role):
    return ROOT/(PARENT[role] if role == 'codec.cuh'
                 else 'tools/kernels/biv150_native_k8_scale_lookahead_'+role)


def transform(value, changes, reverse=False):
    for old, new in (reversed(changes) if reverse else changes):
        before, after = (new, old) if reverse else (old, new)
        if not before or value.count(before) != 1:
            raise ValueError('Unique transform anchor missing: '+repr(before[:80]))
        value = value.replace(before, after)
    return value


def generate(role):
    value = original(role)
    if role == 'qk.cu':
        value = transform(value, EDITS)
    for old, new in RENAMES:
        value = value.replace(old, new)
    return value


def inverse(value, role):
    for old, new in reversed(RENAMES):
        value = value.replace(new, old)
    if role == 'qk.cu':
        value = transform(value, EDITS, reverse=True)
    if sha(value.encode()) != PINS[role]:
        raise ValueError('Non-scale-carry change: '+role)
    return value


def verify():
    files = {}
    for role in FILES:
        value = path(role).read_text()
        if value != generate(role) or inverse(value, role) != original(role):
            raise ValueError('Exact generated source/inverse mismatch: '+role)
        files[role] = dict(parent_sha256=PINS[role], candidate_sha256=sha(value.encode()),
                           inverse_byte_exact=True)
    return dict(base=BASE, reference_SO=REFERENCE_SO, namespace=NAMESPACE, files=files,
        scopes=[[64,2,2048],[64,2,2560],[32,4,2048],[32,4,2560]],
        page_bytes=16384, source_shared_bytes=8192,
        next_scale_scope='has_next full page only; after original V ready barrier beside next K',
        prologue_scope='inside first-full-page guard; one F32 scale per lane',
        carry_scope='private per-thread local; updated after current PV; no shared layout change',
        original_tail_call_and_mask=True, source_math_host_ABI_codec_unchanged=True,
        actual_SO_schedule_verified=False, actual_resources=None, device_compile=False,
        numeric_verified=False, performance_verified=False, precision_gate=False,
        service_enabled=False, capacity_recovered=False)


def frontend_source():
    parent = original('frontend')
    changes = (
        ('import biv150_native_k8_klookahead_factory as factory',
         'import biv150_native_k8_scale_lookahead_factory as factory'),
        ('    return dict(source_transform=factory.verify(), explicit_cuda_flags=CUDA_FLAGS,',
         '    return dict(source_transform=factory.verify(), explicit_cuda_flags=CUDA_FLAGS,\n'
         '        frontend_sha256=sha(__file__), factory_sha256=sha(factory.__file__),\n'
         '        generated_frontend_sha256=factory.sha(factory.frontend_source().encode()),'),
        ("requested='only packed K-next read after original V wait+barrier; publish after current PV'",
         "requested='next full-page K scale carried beside accepted K lookahead; zero launches'"),
        ("    subprocess.run(['git','-C',str(factory.ROOT),'merge-base','--is-ancestor',factory.BASE,commit],check=True)",
         "    if commit != args.expected_source_commit:\n        raise ValueError('Expected exact frozen source commit')\n"
         "    subprocess.run(['git','-C',str(factory.ROOT),'merge-base','--is-ancestor',factory.BASE,commit],check=True)"),
        ("library = load(name='cjj_native_k8_klookahead_probe'", "library = load(name='cjj_native_k8_scale_lookahead_probe'"),
        ('        both_S2_S4_full_next_last_and_original_tail_audited=False,',
         '        both_S2_S4_full_next_last_and_original_tail_audited=False,\n'
         '        first_full_scale_and_no_tail_carry_verified=False,\n'
         '        next_scale_after_V_ready_and_before_current_PV_verified=False,\n'
         '        scale_carry_current_page_identity_and_load_dependency_verified=False,\n'
         '        no_full_loop_duplicate_current_scale_load_verified=False,'),
        ("    p.add_argument('--out',type=Path)",
         "    p.add_argument('--out',type=Path)\n    p.add_argument('--expected-source-commit')"),
        ('        if a.out or a.root_owned_exclusive:',
         '        if a.out or a.root_owned_exclusive or a.expected_source_commit:'),
        ('    if a.out is None or not a.out.is_absolute() or a.out.exists():',
         "    if not a.expected_source_commit or len(a.expected_source_commit)!=40 or any(c not in '0123456789abcdef' for c in a.expected_source_commit):\n"
         "        p.error('Explicit full expected source commit required')\n"
         '    if a.out is None or not a.out.is_absolute() or a.out.exists():'),
    )
    value = transform(parent, changes)
    if transform(value, changes, reverse=True) != parent:
        raise ValueError('Compile frontend inverse changed')
    return value


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    if args.write:
        for role in FILES:
            if role != 'codec.cuh':
                path(role).write_text(generate(role))
    print(json.dumps(verify(), sort_keys=True, indent=2))
