"""Finite original-evidence/SDK/C-address review, CPU only; no numeric claim."""
import argparse,ast,base64,hashlib,json,re,sys
from pathlib import Path
import biv150_hidden_gather_launcher_address as h
import biv150_hidden_gather_service_object_cache as cache

def main():
 p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--service-out',type=Path,required=True);p.add_argument('--driver',type=Path,required=True);p.add_argument('--build',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();checks=[]
 def ok(v,label):
  if not v:raise ValueError(label)
  checks.append(label)
 def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 m=json.loads((a.raw/'download-manifest.json').read_bytes());total=0
 ok(sha(a.raw/'download-manifest.json')=='d6649429e717829f48b4bbb8ccf35fea658e8262092de584b4f60b76b0de2203','original21 manifest')
 for r in m['files']:
  f=a.raw/r['relative'];ok(not f.is_symlink() and f.resolve().is_relative_to(a.raw.resolve()) and f.stat().st_size==r['bytes'] and sha(f)==r['sha256'],'original '+r['relative']);total+=r['bytes']
 ok(len(m['files'])==21 and total==255586,'all21 originals255586B')
 s=json.loads((a.raw/'out/helper-summary.json').read_bytes())
 ok(s['source_commit']=='d74a51b7b8792f5bd883f4c4d071a2f7eb5b3f1a' and s['completed'] is False and s['errors']==["ValueError('actual private original launcher path')"],'original d74 failure unchanged')
 ok(s['records']==[] and s['numeric_candidate_launches_profile_observed']==0 and s['production_warmups_completed']==1 and s['natural_seed']['natural_SME']==1,'one natural warmup,0helper numeric')
 ack=json.loads((a.service_out/'gather-smoke-snapshots/smoke-after.json').read_bytes());actual=ack['compiled'][0]
 for ext,item in actual['files'].items():ok(sha(a.raw/'out/actual-seed-18-1'/('actual.'+ext))==item['sha256'],'actual GPU child unchanged '+ext)
 host=a.service_out/'triton-cache'/cache.HANDLER;new=a.raw/'out/triton-cache'/cache.HANDLER
 ok(sha(host)==sha(new)==h.HOST_SHA,'same5124 present in original service and d74 numeric cache')
 old=a.service_out/'triton-cache/BWNUEQYDUA2UCX5OVOXXXGXFGKQMTEK6QO3ONMXK6GVLAZSEISPQ/__triton_launcher.cpython-312-x86_64-linux-gnu.so'
 ok(sha(old)==cache.REPORTED_MODULE_SHA,'historical module file4105 preserved')
 ok(sha(a.driver)==h.DRIVER_SHA and sha(a.build)==h.BUILD_SHA,'actual saved SDK driver/build')
 tree=ast.parse(a.driver.read_text());names={'ty_to_cpp','make_launcher','FLOAT_STORAGE_TYPE','FLOAT_PACK_FUNCTION','_BASE_ARGS_FORMAT','_BASE_ARGS_FORMAT_LEN'}
 nodes=[n for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name in names) or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets))]
 ns={'re':re};exec(compile(ast.Module(nodes,type_ignores=[]),str(a.driver),'exec'),ns)
 signature={int(k):v for k,v in actual['ABI']['signature'].items()};constants={(int(k),):v for k,v in actual['ABI']['constants'].items()}
 c_source=ns['make_launcher'](constants,signature,[]);platform_inferred='x86_64,Linux,64bit,'
 key=hashlib.sha256((c_source+platform_inferred).encode()).digest();directory=base64.b32encode(key).decode().rstrip('=')
 ok(hashlib.sha256(c_source.encode()).hexdigest()==h.C_SOURCE_SHA and directory==h.HOST_DIRECTORY,'exact SDK ABI-derived C and original H7 cache key')
 expected='iiiKKppOOOOOOOOOiiOO';foreign='iiiKKppOOOOOOOdOiiiiiOO'
 ok(expected in c_source and expected.encode() in host.read_bytes() and foreign.encode() in old.read_bytes() and expected.encode() not in old.read_bytes(),'5124 parser matches gather ABI;4105 parser differs')
 # A CPU builtin has a real C-address owner. Mutating its module label must not
 # change that owner. This demonstrates robust observation, not historic cause.
 import _json
 fn=_json.scanstring;before=h.builtin_binding(fn);original=_json.__file__
 try:
  _json.__file__='/private/tmp/UNTRUSTED-module-label-not-a-loaded-function'
  after=h.builtin_binding(fn)
 finally:_json.__file__=original
 ok(before['C_address']==after['C_address'] and before['DSO_path']==after['DSO_path'] and before['DSO_sha256']==after['DSO_sha256'] and before['legacy_module_path']!=after['legacy_module_path'],'dladdr ownership survives mutable module label; C builtin never invoked')
 try:h.builtin_binding(lambda:None)
 except ValueError:ok(True,'nonbuiltin refused before C API')
 else:raise ValueError('nonbuiltin accepted')
 ok(not {'torch','triton','numpy'}.intersection(sys.modules),'no frameworks or GPU')
 result=dict(passed=True,CPU_only=True,checks=checks,check_count=len(checks),source_commit=cache.c.git('rev-parse','HEAD'),original_failure_preserved=True,original_5f3_not_modified=True,actual_historical_service_C_pointer_available=False,automatic_module_alias_cause_proven=False,platform_key_inferred_from_exact_cache_key=platform_inferred,platform_key_live_must_be_recorded=True,correct_ABI_generated_host_sha256=h.HOST_SHA,reported_module_file_sha256=cache.REPORTED_MODULE_SHA,reported_module_file_is_not_C_pointer_binding=True,device_numeric_verified=False,service_verified=False)
 a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps(dict(checks=len(checks),sha256=sha(a.out))))
if __name__=='__main__':main()
