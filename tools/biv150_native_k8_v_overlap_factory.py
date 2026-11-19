"""One reversible scheduling edit of the device-tested NativeK8 source."""
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PARENT = '37713909f1f25b69d5ef6c155fad3b87351046fb'
BASE = '1cbac531ca9afaa0e9a68624d5d10cb1d0412055'
NAMESPACE = 'cjj_native_k8_v_overlap_probe'
FILES = {'qk.cu': 'biv150_native_k8_qk.cu',
         'qk.cpp': 'biv150_native_k8_qk.cpp',
         'codec.cu': 'biv150_native_k8_codec.cu',
         'codec.cuh': 'biv150_native_k8_codec.cuh'}
PARENT_SHA = {
    'qk.cu': 'bf84c82ff7e73684c3923ca8944c6564f37c21fb636134e106d0542f5ba47cd9',
    'qk.cpp': 'bc6d9d3c289d49435f7598e6b5c2325cfc3f867ab915ab424ffb289bffb20b29',
    'codec.cu': '7b744ba7112083add31ccb49c35055d4b573d718df4343ccf9d10ab998abaad2',
    'codec.cuh': '37d5c7548c0301ee522f50fb0d4d2da53300676e71b03bb806c2d88779801d79',
}
RENAMES = (
    ('corex_native_k8_gqa_producer_draft', 'cjj_native_k8_v_overlap_gqa_producer'),
    ('launch_corex_native_k8_gqa_s2_draft', 'launch_cjj_native_k8_v_overlap_s2'),
    ('launch_corex_native_k8_gqa_s4_draft', 'launch_cjj_native_k8_v_overlap_s4'),
    ('cjj_launch_native_k8_codec', 'cjj_launch_native_k8_v_overlap_codec'),
    ('cjj_native_k8_codec_probe', 'cjj_native_k8_v_overlap_codec_probe'),
    ('cjj_native_k8_qk_probe', NAMESPACE),
)
EDITS = (
    (' // V retains original SME issue/wait/layout. No assumed SME8 instruction.',
     ' // V keeps SME issue/layout; its completion wait moves to the PV consumer.'),
    (' static_assert(!OverlapV,"Only original overlap=false is admitted");',
     ' static_assert(OverlapV,"This isolated variant requires deferred V wait");'),
    (' __ivcorex_sl_waitcnt(8ULL);__syncthreads();\n'
     ' if(valid<16){clear_v_tail(stage,valid,lane);__syncthreads();}\n CI8 scores;',
     ' CI8 scores;'),
    ('  // Wait-moving is a future variant only, not current numeric/perf baseline.',
     '  // Deferred V wait: the unchanged PV consumer must see completed V.'),
    ('#define COREX_NATIVE_GQA_V_OVERLAP 0', '#define COREX_NATIVE_GQA_V_OVERLAP 1'),
    ('static_assert(COREX_NATIVE_GQA_V_OVERLAP==0,"Private K8 prototype requires original overlap=false");',
     'static_assert(COREX_NATIVE_GQA_V_OVERLAP==1,"Private K8 V-overlap requires macro 1");'),
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def original(role):
    raw = subprocess.check_output(['git', '-C', str(ROOT), 'show',
                                   PARENT + ':tools/kernels/' + FILES[role]])
    assert sha(raw) == PARENT_SHA[role]
    return raw.decode()


def path(role):
    if role == 'codec.cuh':
        return ROOT / 'tools/kernels' / FILES[role]
    return ROOT / 'tools/kernels' / ('biv150_native_k8_v_overlap_' + role)


def generate(role):
    text = original(role)
    if role == 'qk.cu':
        for old, new in EDITS:
            assert text.count(old) == 1, old
            text = text.replace(old, new)
    for old, new in RENAMES:
        text = text.replace(old, new)
    return text


def inverse(text, role):
    for old, new in reversed(RENAMES):
        text = text.replace(new, old)
    if role == 'qk.cu':
        for old, new in reversed(EDITS):
            assert text.count(new) == 1, new
            text = text.replace(new, old)
    assert sha(text.encode()) == PARENT_SHA[role], 'out-of-scope change: ' + role
    return text


def verify():
    rows = {}
    for role in FILES:
        actual = path(role).read_text()
        old = inverse(actual, role)
        rows[role] = dict(parent_sha256=sha(old.encode()),
                          variant_sha256=sha(actual.encode()), inverse_byte_exact=True)
    return dict(base=BASE, parent=PARENT, namespace=NAMESPACE, files=rows,
                issue_order='K ordinary load/store + barrier, then V SME (unchanged)',
                deferred_wait='after QK/softmax/P packing; before V tail clear and PV',
                V_first_PR96_combined=False, physical_page_bytes=16384,
                Q_quantization_QK_softmax_PV_and_outputs_unchanged=True,
                actual_device_schedule_verified=False, numeric_verified=False,
                performance_verified=False, service_enabled=False, precision_gate=False)


if __name__ == '__main__':
    import argparse
    import json
    p = argparse.ArgumentParser()
    p.add_argument('--write', action='store_true')
    a = p.parse_args()
    if a.write:
        for role in FILES:
            path(role).parent.mkdir(parents=True, exist_ok=True)
            path(role).write_text(generate(role))
    print(json.dumps(verify(), indent=2, sort_keys=True))
