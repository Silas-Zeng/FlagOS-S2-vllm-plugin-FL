#!/usr/bin/env python3
"""Verify this finite evidence bundle; optionally replay the preserved CPU audit."""
import argparse,hashlib,json,shutil,subprocess,sys,tarfile,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
COMMIT='6c08ec2561b1c3cee96838bffd4e296b757ad877'
sha=lambda b:hashlib.sha256(b).hexdigest()
def verify():
 manifest=json.loads((HERE/'files-manifest.json').read_text())
 assert {str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()}=={'files-manifest.json',*(x['path'] for x in manifest['files'])}
 for row in manifest['files']:
  data=(HERE/row['path']).read_bytes();assert len(data)==row['bytes']<=2*1024*1024 and sha(data)==row['sha256']
 index=json.loads((HERE/'archive-index.json').read_text());seen={'v1':{},'v2':{}}
 for package in index['archives']:
  data=(HERE/package['path']).read_bytes();assert len(data)==package['bytes'] and sha(data)==package['sha256']
  expected={x['path']:x for x in package['members']}
  with tarfile.open(HERE/package['path'],'r:gz') as archive:
   rows=archive.getmembers();assert len(rows)==len(expected) and {x.name for x in rows}==set(expected)
   for member in rows:
    assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
    assert member.name not in seen[package['version']]
    content=archive.extractfile(member).read();record=dict(bytes=len(content),sha256=sha(content))
    assert record=={k:v for k,v in expected[member.name].items() if k!='path'}
    seen[package['version']][member.name]=record
 for version,count,size in [('v1',2,1931),('v2',50,4937816)]:
  downloaded=json.loads((HERE/('compile-'+version+'-download-manifest.json')).read_text())
  assert len(seen[version])==downloaded['raw_files']==count and sum(x['bytes'] for x in seen[version].values())==downloaded['raw_bytes']==size
  assert seen[version]=={x['relative']:dict(bytes=x['bytes'],sha256=x['sha256']) for x in downloaded['files']}
 # Inspect v1's original traceback only; do not rerun compilation.
 with tarfile.open(HERE/'raw-compile-v1.tar.gz','r:gz') as archive:
  log=archive.extractfile('exports/cjj-compact-q32-hoist-compile-v1-20261028.log').read().decode()
  assert 'line 89, in run' in log and 'args.out.mkdir()' in log and 'FileNotFoundError' in log
 print(json.dumps(dict(package_verified=True,raw_v1_files=2,raw_v2_files=50,all_files_at_most_2MiB=True,originals_changed=False,full_audit_replayed=False)))
 return index

def replay(index,output):
 output=output.resolve();erratum=output.with_name(output.stem+'.erratum.json')
 assert not output.exists() and not erratum.exists() and output.parent.is_dir()
 script=HERE/'cjj-pr103-actual-compile-review-20261028.py';original=json.loads((HERE/'cjj-pr103-actual-compile-review-20261028.json').read_text())
 assert sha(script.read_bytes())==original['review_script_sha256']
 with tempfile.TemporaryDirectory(prefix='compact103-review-') as temporary:
  root=Path(temporary);raw=root/'raw-v2';raw.mkdir()
  for package in index['archives']:
   if package['version']!='v2':continue
   with tarfile.open(HERE/package['path'],'r:gz') as archive:
    for member in archive.getmembers():
     dest=raw/member.name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(archive.extractfile(member).read())
  shutil.copyfile(HERE/'compile-v2-download-manifest.json',raw/'download-manifest.json')
  source=root/'source'
  # Local read-only object sharing; no fetch, network or changes to the parent checkout.
  subprocess.run(['git','clone','--quiet','--shared','--no-checkout',str(HERE.parents[2]),str(source)],check=True)
  subprocess.run(['git','-C',str(source),'checkout','--quiet','--detach',COMMIT],check=True)
  replacements={
   '/private/tmp/cjj-compact-q32-hoist-compile-v2-raw-20261028':str(raw),
   '/private/tmp/flagos-cjj-compact-q32-probe-frozen-6c0-20261028':str(source),
   '/private/tmp/cjj-pr103-actual-compile-review-20261028.json':str(output),
   '/private/tmp/flagos-cjj-native-k8-vfirst-20261028/tools':str(HERE/'replay-deps/elf-reader/tools'),
   '/private/tmp/cjj-pr103-next-compile-review-20261028.json':str(HERE/'cjj-pr103-next-compile-review-20261028.json'),
   '/private/tmp/flagos_launch_personal_component_v3_bundle_base_20261027.py':str(HERE/'replay-deps/flagos_launch_personal_component_v3_bundle_base_20261027.py'),
   '/private/tmp/cjj-pr103-next-compile-review-erratum-20261028.json':str(erratum)}
  for name in ('cjj-compact-q32-hoist-compile-v1-20261028-launch.json','cjj-compact-q32-hoist-compile-v1-20261028-root-reviewed-launch.json','cjj-compact-q32-hoist-compile-v2-20261028-root-reviewed-launch.json'):
   replacements['/private/tmp/'+name]=str(HERE/'launch'/name)
  text=script.read_text()
  for old,new in replacements.items():
   assert text.count(repr(old))==1,(old,text.count(repr(old)))
   text=text.replace(repr(old),repr(new))
  # Only filesystem literals change in memory. Preserved source and original reports stay unchanged.
  exec(compile(text,str(script),'exec'),{'__name__':'__main__','__file__':str(script)})

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');parser.add_argument('--out',type=Path)
 args=parser.parse_args();assert args.replay==bool(args.out),'--replay requires a new --out JSON path'
 data=verify()
 if args.replay:replay(data,args.out)
