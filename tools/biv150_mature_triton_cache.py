"""Copy a completed matching run's Triton artifacts and rebase cache-group links.

No imports of Torch/Triton or GPU use. Does not merge tuning databases, alter
kernel artifacts, or accept a cache from a different runtime/binding/library.
"""
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore(seed_run,target,*,source,commit,binding,library_sha256):
    seed_run=Path(seed_run).resolve();target=Path(target).resolve();source=Path(source).resolve()
    manifest=json.loads((seed_run/'manifest.json').read_text());state=json.loads((seed_run/'state.json').read_text())
    assert state['complete'] and state['cleanup_complete'],'Seed run is not completed/cleaned'
    assert manifest['commit']==commit and Path(manifest['source']).resolve()==source
    assert manifest['small_binding']==binding and manifest['stable_small_o']=='1' and manifest['native_mode']=='on'
    assert manifest['gate_library_sha256']==library_sha256
    root=seed_run/'triton-cache';assert root.is_dir() and not root.is_symlink() and not target.exists()
    paths=sorted(root.rglob('*'));assert not any(p.is_symlink() for p in paths),'Cache must be self-contained'
    originals={str(p.relative_to(root)):sha(p) for p in paths if p.is_file()}
    shutil.copytree(root,target)
    assert all(sha(target/name)==digest for name,digest in originals.items())
    rebased=[]
    for path in sorted(target.rglob('__grp__*.json')):
        group=json.loads(path.read_text());assert isinstance(group.get('child_paths'),dict)
        children={}
        for name,value in group['child_paths'].items():
            old=Path(value).resolve();relative=old.relative_to(root)
            new=target/relative;assert new.is_file() and sha(new)==originals[str(relative)]
            children[name]=str(new)
        group['child_paths']=children;path.write_text(json.dumps(group,separators=(',',':')))
        rebased.append({'path':str(path.relative_to(target)),'old_sha256':originals[str(path.relative_to(target))],'new_sha256':sha(path)})
    for name,digest in originals.items():
        if not Path(name).name.startswith('__grp__'):assert sha(target/name)==digest,'Kernel artifact changed'
    return {'seed_run':str(seed_run),'source_commit':commit,'binding':binding,
            'seed_manifest_sha256':sha(seed_run/'manifest.json'),'seed_state_sha256':sha(seed_run/'state.json'),
            'copied_files':len(originals),'source_file_sha256':originals,'rebased_groups':rebased,
            'kernel_artifact_bytes_unchanged':True,'tuning_database_merged':False}
