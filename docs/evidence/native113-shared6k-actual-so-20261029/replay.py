"""Verify the bounded original package; optional local CPU-only SO review replay."""
import argparse,hashlib,io,json,subprocess,sys,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def sha(raw):return hashlib.sha256(raw).hexdigest()
def unpack(raw,target=None):
    rows=[]
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as t:
        for m in t.getmembers():
            p=Path(m.name);assert m.isfile() and not p.is_absolute() and '..' not in p.parts
            b=t.extractfile(m).read();rows.append(dict(path=m.name,bytes=len(b),sha256=sha(b)))
            if target:
                out=target/p;assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(b)
    assert len(rows)==len({r['path'] for r in rows})
    return rows
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--out',type=Path);a=p.parse_args()
    manifest=json.loads((HERE/'manifest.json').read_text())
    for row in manifest['files']:
        b=(HERE/row['path']).read_bytes();assert (len(b),sha(b))==(row['bytes'],row['sha256'])
    raw=(HERE/'compile-original-records.tar.gz').read_bytes();index=json.loads((HERE/'archive-index.json').read_text());assert unpack(raw)==index['members']
    original=json.loads((HERE/'actual-so-review.json').read_text());approved=json.loads((HERE/'root-numeric-admission.json').read_text())
    assert original['manual_Root_approval'] is False and approved['manual_Root_approval'] is True
    assert original['script_sha256']==sha((HERE/'review_actual_so.py').read_bytes())
    assert sha((HERE/'root-numeric-admission.json').read_bytes())=='b481ccf7ec5da72b3036b91d212ebdf02c97d906d93f0914d083df7b35d2d87f'
    if a.run:
        assert a.out and a.out.is_absolute() and not a.out.exists();a.out.mkdir()
        unpack(raw,a.out/'candidate')
        dep=json.loads((HERE/'existing-dependencies.json').read_text())['reference_archive']
        old=subprocess.check_output(['git','-C',str(ROOT),'show',dep['commit']+':'+dep['path']]);assert sha(old)==dep['sha256'];unpack(old,a.out/'reference')
        report=a.out/'actual-so-review.json'
        subprocess.run([sys.executable,'-B',str(HERE/'review_actual_so.py'),'--repo',str(ROOT),'--raw',str(a.out/'candidate'),'--reference-raw',str(a.out/'reference'),'--report',str(report)],check=True)
        assert report.read_bytes()==(HERE/'actual-so-review.json').read_bytes()
    else:assert a.out is None
    print(json.dumps(dict(passed=True,archive_members=len(index['members']),CPU_full_review_replayed=a.run,GPU_executed=False)))
if __name__=='__main__':main()
