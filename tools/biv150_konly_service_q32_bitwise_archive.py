"""CPU byte-preserving multipart archive; each outer gzip is <=2MiB and valid tar."""
import argparse,copy,gzip,hashlib,io,json
from pathlib import Path
import tarfile

SCHEMA='cjj-service-q32-bitwise-originals-v1'
LIMIT=2*1024*1024
CHUNK=1900000
sha=lambda b:hashlib.sha256(b).hexdigest()

def safe(name):return isinstance(name,str) and name and not name.startswith('/') and '..' not in Path(name).parts

def tar_bytes(members):
    out=io.BytesIO()
    with gzip.GzipFile(fileobj=out,mode='wb',filename='',mtime=0,compresslevel=6) as stream:
        with tarfile.open(fileobj=stream,mode='w') as tar:
            for name,b in sorted(members.items()):
                assert safe(name);m=tarfile.TarInfo(name);m.size=len(b);m.mtime=0;m.mode=0o644;tar.addfile(m,io.BytesIO(b))
    assert len(out.getvalue())<=LIMIT
    return out.getvalue()

def originals(files):
    manifest=json.loads(files['download-manifest.json'])
    assert (manifest['raw_files'],manifest['raw_bytes'])==(42,60998965)
    assert len(manifest['files'])==42 and len({r['relative'] for r in manifest['files']})==42
    assert sum(r['bytes'] for r in manifest['files'])==60998965 and all(safe(r['relative']) for r in manifest['files'])
    assert manifest['original_owner']==dict(pid=652604,start_ticks=346185455,same_start=True,state='Z')
    assert set(files)=={r['relative'] for r in manifest['files']}|{'download-manifest.json'}
    for r in manifest['files']:assert (len(files[r['relative']]),sha(files[r['relative']]))==(r['bytes'],r['sha256'])
    return manifest

def pack(raw,dest):
    files={p.relative_to(raw).as_posix():p.read_bytes() for p in raw.rglob('*') if p.is_file()};manifest=originals(files)
    dest.mkdir(parents=True,exist_ok=False);index=dict(schema=SCHEMA,raw_files=42,raw_bytes=60998965,
        original_files={n:dict(bytes=len(b),sha256=sha(b)) for n,b in sorted(files.items())},parts=[])
    small={n:b for n,b in files.items() if not n.endswith('.pt.gz')}
    groups=[[(n,n,0,b) for n,b in sorted(small.items())]]
    for number,(name,b) in enumerate(sorted(files.items())):
        if not name.endswith('.pt.gz'):continue
        for offset in range(0,len(b),CHUNK):groups.append([(f'fragments/file-{number:03d}-offset-{offset:09d}.bin',name,offset,b[offset:offset+CHUNK])])
    for i,group in enumerate(groups):
        name=f'original-bitwise-part-{i:03d}.tar.gz';b=tar_bytes({member:value for member,_,_,value in group});(dest/name).write_bytes(b)
        index['parts'].append(dict(name=name,bytes=len(b),sha256=sha(b),members=[dict(member=member,original=original,offset=offset,bytes=len(value),sha256=sha(value)) for member,original,offset,value in group]))
    (dest/'index.json').write_text(json.dumps(index,sort_keys=True,indent=2)+'\n');return index

def load(directory,index=None):
    index=json.loads((directory/'index.json').read_text()) if index is None else index
    assert index['schema']==SCHEMA and (index['raw_files'],index['raw_bytes'])==(42,60998965)
    names=[p['name'] for p in index['parts']];assert len(names)==len(set(names)) and all(safe(n) for n in names)
    chunks={n:[] for n in index['original_files']}
    for p in index['parts']:
        b=(directory/p['name']).read_bytes();assert len(b)<=LIMIT and (len(b),sha(b))==(p['bytes'],p['sha256'])
        with tarfile.open(fileobj=io.BytesIO(b)) as tar:
            members=tar.getmembers();expected={m['member']:m for m in p['members']}
            assert len(expected)==len(members) and all(m.isfile() and safe(m.name) for m in members) and {m.name for m in members}==set(expected)
            for m in members:
                data=tar.extractfile(m).read();r=expected[m.name];assert (len(data),sha(data))==(r['bytes'],r['sha256'])
                assert r['original'] in chunks and type(r['offset']) is int and r['offset']>=0
                chunks[r['original']].append((r['offset'],data))
    files={}
    for name,parts in chunks.items():
        assert safe(name);parts.sort();end=0;out=[]
        for offset,b in parts:assert offset==end;end+=len(b);out.append(b)
        b=b''.join(out);assert (len(b),sha(b))==(index['original_files'][name]['bytes'],index['original_files'][name]['sha256']);files[name]=b
    manifest=originals(files)
    return files,index,manifest

def faults(directory):
    _,original,_=load(directory);passed=[]
    for name,change in [('part SHA',lambda x:x['parts'][0].update(sha256='0'*64)),
        ('missing part',lambda x:x['parts'].pop()),('fragment gap',lambda x:x['parts'][-1]['members'][0].update(offset=1)),
        ('file SHA',lambda x:x['original_files']['download-manifest.json'].update(sha256='0'*64)),
        ('raw count',lambda x:x.update(raw_files=41)),('duplicate part',lambda x:x['parts'].append(x['parts'][0]))]:
        index=copy.deepcopy(original);change(index)
        try:load(directory,index)
        except (AssertionError,ValueError,KeyError):passed.append(name)
        else:raise AssertionError('accepted '+name)
    return passed

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pack',type=Path);p.add_argument('--directory',type=Path,required=True);p.add_argument('--restore',type=Path);a=p.parse_args()
    if a.pack:pack(a.pack,a.directory)
    files,index,m=load(a.directory)
    if a.restore:
        a.restore.mkdir(parents=True,exist_ok=False)
        for name,b in files.items():path=a.restore/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b)
    print(json.dumps(dict(CPU_only=True,all_original_bytes_verified=True,raw_files=m['raw_files'],raw_bytes=m['raw_bytes'],archive_parts=len(index['parts']),archive_bytes=sum(p['bytes'] for p in index['parts']),index_SHA256=sha((a.directory/'index.json').read_bytes()),faults_rejected=faults(a.directory),GPU_executed=False),sort_keys=True,indent=2))
if __name__=='__main__':main()
