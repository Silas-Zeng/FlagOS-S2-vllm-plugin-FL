#!/usr/bin/env python3
"""Source-only subprocess through actual helper94db standalone semantics loader."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HELPER_SHA='94dbbb0fbc8f3c8837ea239ec18fc0ec4872a2b0049a1843d96a3e6a9701e976'
CHILD=r'''
import hashlib,importlib.abc,importlib.util,json,sys
from pathlib import Path
class NoFramework(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','triton','vllm','vllm_fl','numpy'):
   raise AssertionError('Unexpected framework import: '+fullname)
sys.meta_path.insert(0,NoFramework())
helper,source=map(Path,sys.argv[1:3])
assert hashlib.sha256(helper.read_bytes()).hexdigest()==sys.argv[3]
spec=importlib.util.spec_from_file_location('CPU_actual_helper94db',helper)
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
assert h.__package__==''
records=[]
native_path=sys.argv[4] if len(sys.argv)>4 else None
for native,enabled in [(n,h) for n in (('0','1') if native_path else ('0',)) for h in ('0','1')]:
 env={'VLLM_FL_BIV150_NATIVE_W8A8':'0','VLLM_FL_BIV150_NATIVE_GQA':'0',
      'VLLM_FL_BIV150_KONLY_COMPACT_SERVICE':'1',
      'VLLM_FL_BIV150_KONLY_ENCODED_SERVICE':'0',
      'VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST':enabled}
 if native=='1':
  env.update(VLLM_FL_BIV150_KONLY_COMPACT_NATIVE_DECODE='1',
             VLLM_FL_BIV150_KONLY_COMPACT_NATIVE_LIBRARY=native_path,
             VLLM_FL_BIV150_KONLY_COMPACT_NATIVE_LIBRARY_SHA256=hashlib.sha256(Path(native_path).read_bytes()).hexdigest())
 # Actual unmodified expected_fingerprint calls helper.module, which uses
 # spec_from_file_location with no package or sys.modules registration.
 fp=h.expected_fingerprint(source,env,{'source_only_CPU':True})
 assert fp['konly_compact']['q32_hoist_enabled']==(enabled=='1')
 assert fp['konly_compact']['format']=='cjj-konly-compact12544-rn32-rne-v1'
 if native=='1':assert fp['konly_compact']['native_decode']['qualification']['library_sha256']==env['VLLM_FL_BIV150_KONLY_COMPACT_NATIVE_LIBRARY_SHA256']
 records.append({'native':native,'hoist':enabled,'fingerprint':fp['sha256'],
                 'implementation_sha256':fp['konly_compact']['implementation_sha256']})
assert len({r['fingerprint'] for r in records})==len(records)
assert len({r['implementation_sha256'] for r in records})==1
assert not any(k.split('.')[0] in ('torch','triton','vllm','vllm_fl','numpy') for k in sys.modules)
print(json.dumps({'passed':True,'actual_helper_sha256':sys.argv[3],
 'actual_helper_expected_fingerprint_function_used':True,'no_parent_package':True,
 'framework_imports_blocked_and_absent':True,'explicit_CPU_ABI_no_SO_or_Torch_required':native_path is None,
 'actual_SO_read_not_loaded':native_path is not None,
 'records':records,'GPU_or_compiler_executed':False,'service_or_numeric_verified':False},sort_keys=True))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--helper',type=Path,required=True)
    p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--compact-native-library',type=Path);a=p.parse_args()
    assert hashlib.sha256(a.helper.read_bytes()).hexdigest()==HELPER_SHA
    result=subprocess.check_output([sys.executable,'-I','-S','-B','-W','error','-c',CHILD,
                                    str(a.helper.resolve()),str(a.source.resolve()),HELPER_SHA]
                                    + ([str(a.compact_native_library.resolve())] if a.compact_native_library else []),
                                   text=True,timeout=30)
    print(json.dumps(json.loads(result),sort_keys=True,indent=2))


if __name__=='__main__':main()
