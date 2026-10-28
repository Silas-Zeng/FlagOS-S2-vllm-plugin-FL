#!/usr/bin/env python3
"""Execute unchanged actual94db expected_fingerprint, standalone, without GPU imports."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
HELPER=Path('/private/tmp/cjj-actual-owned-service-helper-94db-20261028.py')
HELPER_SHA='94dbbb0fbc8f3c8837ea239ec18fc0ec4872a2b0049a1843d96a3e6a9701e976'
LIBRARY=Path('/private/tmp/cjj-native-k8-v-overlap-compile-v1-raw-20261028/out/build/cjj_native_k8_v_overlap_probe.so')
LIBRARY_SHA='8b86cb688ad72dcf9859994ed2092620312cdd5ae96d194904b2bde7926aacc5'

def run(helper,library):
    assert hashlib.sha256(helper.read_bytes()).hexdigest()==HELPER_SHA
    assert library.is_absolute() and hashlib.sha256(library.read_bytes()).hexdigest()==LIBRARY_SHA
    spec=importlib.util.spec_from_file_location('actual94db_owned_helper_cpu',helper)
    owned=importlib.util.module_from_spec(spec);spec.loader.exec_module(owned)
    # This is the actual helper implementation, not a reimplementation or a
    # package-import substitute. It creates stock105_expected_semantics with
    # no parent package and supplies the exact two-field ABI explicitly.
    standalone=owned.module(ROOT/'vllm_fl/compile_semantics.py','check_native_standalone_semantics')
    assert standalone.__package__ in (None,'')
    base={'VLLM_FL_BIV150_NATIVE_W8A8':'0','VLLM_FL_BIV150_NATIVE_GQA':'0',
          'VLLM_FL_BIV150_KONLY_ENCODED_SERVICE':'1','VLLM_FL_BIV150_KONLY_NATIVE_DECODE':'0',
          'VLLM_FL_BIV150_KONLY_NATIVE_LIBRARY':str(library),'VLLM_FL_BIV150_KONLY_NATIVE_LIBRARY_SHA256':LIBRARY_SHA}
    abi={'torch':'2.10.0','cxx11_abi':False};calls=[]
    for enabled,current in [(False,abi),(True,abi),(True,{'torch':'2.10.0','cxx11_abi':True}),(True,{'torch':'2.9.0','cxx11_abi':False})]:
        env={**base,'VLLM_FL_BIV150_KONLY_NATIVE_DECODE':'1' if enabled else '0'}
        fp=owned.expected_fingerprint(ROOT,env,current)
        record=fp['konly_encoded'].get('native_decode')
        if enabled:assert record==dict(enabled=True,library_sha256=LIBRARY_SHA,operator='cjj_native_k8_v_overlap_probe::produce')
        else:assert record is None
        calls.append(dict(enabled=enabled,abi=current,result=fp))
    assert len({r['result']['sha256'] for r in calls})==4
    no_private=owned.expected_fingerprint(ROOT,{**base,'VLLM_FL_BIV150_KONLY_ENCODED_SERVICE':'0'},abi)
    assert 'konly_encoded' not in no_private
    rejected=0
    for extra in [{'VLLM_FL_BIV150_KONLY_ENCODED_SERVICE':'0'},
                  {'VLLM_FL_BIV150_KONLY_COMPACT_SERVICE':'1'},
                  {'VLLM_FL_BIV150_KONLY_NATIVE_LIBRARY_SHA256':'0'*64},
                  {'VLLM_FL_BIV150_KONLY_NATIVE_LIBRARY_SHA256':'22e4a5e2a0b9d9a36ea390a11126417e1bf58e3c27f6b96ba0913b27690f09dd'},
                  {'VLLM_FL_BIV150_KONLY_NATIVE_LIBRARY':'relative.so'}]:
        try:owned.expected_fingerprint(ROOT,{**base,'VLLM_FL_BIV150_KONLY_NATIVE_DECODE':'1',**extra},abi)
        except (ValueError,RuntimeError):rejected+=1
        else:raise AssertionError('bad NativeK8 identity accepted')
    assert rejected==5 and not {'torch','triton','vllm','vllm_fl'}.intersection(sys.modules)
    assert not any(k.startswith('vllm_fl.patches.biv150_konly') for k in sys.modules)
    assert hashlib.sha256(helper.read_bytes()).hexdigest()==HELPER_SHA
    return dict(passed=True,CPU_only=True,actual_helper_sha256=HELPER_SHA,actual_expected_fingerprint_called=True,
                standalone_no_parent_package=True,private_or_GPU_package_imported=False,
                on_off_and_ABI_cases=calls,all_private_off=no_private,rejections=rejected,
                actual_library_content_SHA_verified=LIBRARY_SHA,helper_modified=False,library_loaded=False,GPU_or_SSH_executed=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--helper',type=Path,default=HELPER);p.add_argument('--library',type=Path,default=LIBRARY)
    a=p.parse_args();print(json.dumps(run(a.helper,a.library),indent=2))
