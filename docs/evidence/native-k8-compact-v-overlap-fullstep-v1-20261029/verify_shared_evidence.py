"""Light stdlib checksum/metadata verification, never a numerical/GPU rerun."""
import argparse,gzip,hashlib,io,json,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--full-archive',type=Path);a=p.parse_args()
 index=json.loads((HERE/'evidence-files.json').read_text())
 for name,row in index['files'].items():
  q=HERE/name;data=q.read_bytes();assert len(data)==row['bytes'] and sha(data)==row['sha256'],name
  assert len(data)<=2097152,name
 package=json.loads((HERE/'metadata-package.json').read_text());m=json.loads((HERE/'original-download-manifest.json').read_text())
 assert m['raw_files']==len(m['files'])==181 and m['raw_bytes']==sum(x['bytes'] for x in m['files'])==145951897
 assert m['original_owner']==dict(pid=61630,start_ticks=355834758,same_start=True,state='Z')
 original={x['relative']:x for x in m['files']};inc=package['included_raw_files'];exc=package['excluded_raw_files']
 assert len(inc)==37 and len(exc)==144 and {x['relative'] for x in inc+exc}==set(original)
 archive=(HERE/package['archive']['path']).read_bytes();assert sha(archive)==package['archive']['sha256'] and len(archive)==package['archive']['bytes']<=2097152
 members={}
 with tarfile.open(fileobj=io.BytesIO(gzip.decompress(archive)),mode='r:') as tf:
  for member in tf.getmembers():
   assert member.isfile() and member.name not in members and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
   assert member.uid==member.gid==member.mtime==0
   members[member.name]=tf.extractfile(member).read()
 assert set(members)=={'download-manifest.json'}|{x['relative'] for x in inc}
 assert members.pop('download-manifest.json')==(HERE/'original-download-manifest.json').read_bytes()
 for name,data in members.items():assert sha(data)==original[name]['sha256'] and len(data)==original[name]['bytes']
 summary=json.loads(members['out/fullstep-summary.json']);review=json.loads((HERE/'independent-review.json').read_text())
 assert summary['completed'] and not summary['errors'] and len(summary['records'])==2
 assert summary['K8_library_sha256']==review['K8_SO']=='7892a6aad09d247a322a74906312f250a9d253160202fee23259adac7b7260d4'
 assert summary['BF16_library_sha256']==review['BF16_SO']=='aace693bdb302d2685ec1b5e11e859c8a4a77d7f45c0f4dacddc6b5e49b015b8'
 assert review['passed'] and review['source_commit']=='4d1166792a5247b774d6b1977e60466574e6a34c'
 assert review['ideal_partial_diagnostic_failures_preserved']=={'4k':12,'16k':12}
 full=json.loads((HERE/'full-archive-location.json').read_text());full_checked=False
 if a.full_archive is not None:
  h=hashlib.sha256();size=0
  with a.full_archive.open('rb') as stream:
   for chunk in iter(lambda:stream.read(4194304),b''):h.update(chunk);size+=len(chunk)
  assert h.hexdigest()==full['sha256'] and size==full['bytes'];full_checked=True
 print(json.dumps(dict(shared_files_verified=len(index['files']),metadata_raw_members_verified=37,full_manifest_entries=181,metadata_archive_sha256=package['archive']['sha256'],review_sha256=sha((HERE/'independent-review.json').read_bytes()),full_external_archive_rechecked=full_checked,full_arrays_in_Git=False,numerical_review_rerun=False,GPU_or_frameworks_used=False),sort_keys=True,indent=2))
if __name__=='__main__':main()
