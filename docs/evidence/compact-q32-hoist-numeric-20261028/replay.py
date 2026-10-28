#!/usr/bin/env python3
"""Lightweight package verification; explicit optional replay needs the local full archive."""
import argparse,hashlib,json,shutil,subprocess,sys,tarfile,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
COMPILE=HERE.parent/'compact-q32-hoist-compile-20261028'
sha=lambda b:hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_text())
def file_sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def verify():
 files=read(HERE/'files-manifest.json')['files']
 assert {str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()}=={'files-manifest.json',*(x['path'] for x in files)}
 for x in files:
  p=HERE/x['path'];assert p.stat().st_size==x['bytes']<=2*1024*1024 and file_sha(p)==x['sha256']
 index=read(HERE/'archive-index.json');all_rows=read(HERE/'download-manifest-complete71.json')['files']
 actual={};all_expected={x['relative']:dict(bytes=x['bytes'],sha256=x['sha256']) for x in all_rows}
 for package in index['archives']:
  p=HERE/package['path'];assert p.stat().st_size==package['bytes'] and file_sha(p)==package['sha256']
  expected={x['path']:dict(bytes=x['bytes'],sha256=x['sha256']) for x in package['members']}
  with tarfile.open(p,'r:gz') as archive:
   members=archive.getmembers();assert len(members)==len(expected) and {x.name for x in members}==set(expected)
   for member in members:
    assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
    assert not member.name.endswith('.npz') and member.name not in actual
    value=archive.extractfile(member).read();record=dict(bytes=len(value),sha256=sha(value));assert record==expected[member.name]==all_expected[member.name]
    actual[member.name]=record
 omitted={x['relative']:dict(bytes=x['bytes'],sha256=x['sha256']) for x in index['omitted_NPZ_originals']}
 assert len(actual)==65 and sum(x['bytes'] for x in actual.values())==6793606
 assert len(omitted)==6 and all(x.endswith('.npz') for x in omitted)
 assert not set(actual)&set(omitted) and {**actual,**omitted}==all_expected and len(all_expected)==71
 assert sum(x['bytes'] for x in all_expected.values())==374807676
 deps=read(HERE/'replay-dependencies.json')
 for path,digest in deps['existing_evidence_files_SHA256'].items():assert file_sha(REPO/path)==digest
 for path,digest in deps['source_files_SHA256'].items():
  assert sha(subprocess.check_output(['git','show',deps['source_commit']+':'+path],cwd=REPO))==digest
 print(json.dumps(dict(package_verified=True,Git_shared_originals=65,omitted_NPZ_originals=6,complete_manifest_originals=71,full_arrays_Git_shared=False,full_numeric_audit_replayed=False)))
 return deps

def unpack(archive_path,destination,expected):
 seen=set()
 with tarfile.open(archive_path,'r:gz') as archive:
  for member in archive:
   assert member.isfile() and member.name in expected and member.name not in seen
   assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
   dest=destination/member.name;dest.parent.mkdir(parents=True,exist_ok=True)
   with dest.open('wb') as f:shutil.copyfileobj(archive.extractfile(member),f)
   row=expected[member.name];assert dest.stat().st_size==row['bytes'] and file_sha(dest)==row['sha256'];seen.add(member.name)
 assert seen==set(expected)

def replay(deps,full,output):
 output=output.resolve();assert not output.exists() and output.parent.is_dir()
 script=HERE/'cjj-pr103-actual-numeric-review-20261028.py';assert file_sha(script)==deps['original_review_script_SHA256']
 with tempfile.TemporaryDirectory(prefix='compact103-numeric-review-') as temp:
  root=Path(temp);numeric=root/'numeric';numeric.mkdir();compile_root=root/'compile';compile_root.mkdir()
  manifest=HERE/'download-manifest-complete71.json'
  expected={x['relative']:dict(bytes=x['bytes'],sha256=x['sha256']) for x in read(manifest)['files']}
  expected['download-manifest.json']=dict(bytes=manifest.stat().st_size,sha256=file_sha(manifest))
  unpack(full,numeric,expected)
  cm=read(COMPILE/'compile-v2-download-manifest.json');expected_compile={x['relative']:dict(bytes=x['bytes'],sha256=x['sha256']) for x in cm['files']}
  # The immutable parent evidence contains exactly one complete compile-v2 package.
  packages=[x for x in read(COMPILE/'archive-index.json')['archives'] if x['version']=='v2'];assert len(packages)==1
  unpack(COMPILE/packages[0]['path'],compile_root,expected_compile)
  shutil.copyfile(COMPILE/'compile-v2-download-manifest.json',compile_root/'download-manifest.json')
  source=root/'source'
  subprocess.run(['git','clone','--quiet','--shared','--no-checkout',str(REPO),str(source)],check=True)
  subprocess.run(['git','-C',str(source),'checkout','--quiet','--detach',deps['source_commit']],check=True)
  replacements={
   '/private/tmp/cjj-compact-q32-hoist-numeric-v1-raw-20261028':str(numeric),
   '/private/tmp/cjj-compact-q32-hoist-compile-v2-raw-20261028':str(compile_root),
   '/private/tmp/flagos-cjj-compact-q32-probe-frozen-6c0-20261028':str(source),
   '/private/tmp/cjj-pr103-actual-numeric-review-20261028.json':str(output),
   '/private/tmp/cjj-compact-q32-hoist-numeric-v1-raw-20261028.tar-stream-v1.tar.gz':str(full.resolve()),
   '/private/tmp/flagos-cjj-native-k8-vfirst-20261028/tools':str(COMPILE/'replay-deps/elf-reader/tools')}
  text=script.read_text()
  for old,new in replacements.items():
   assert text.count(repr(old))==1,(old,text.count(repr(old)));text=text.replace(repr(old),repr(new))
  # Only local filesystem literals are remapped; original script and all numerical gates stay unchanged.
  exec(compile(text,str(script),'exec'),{'__name__':'__main__','__file__':str(script)})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path);p.add_argument('--replay',action='store_true');p.add_argument('--out',type=Path);a=p.parse_args()
 assert a.replay==bool(a.out),'--replay requires a fresh --out path'
 assert not a.replay or a.archive is not None,'full CPU replay requires the non-Git full archive'
 deps=verify()
 if a.archive:
  expected=read(HERE/'external-storage.json')
  assert a.archive.stat().st_size==expected['bytes'] and file_sha(a.archive)==expected['sha256']
  print(json.dumps(dict(full_archive_hash_verified=True,full_numeric_audit_replayed=False)))
 if a.replay:replay(deps,a.archive,a.out)
