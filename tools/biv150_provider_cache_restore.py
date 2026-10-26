"""Restore only a completed, identical provider runtime's private Triton cache."""
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_seed(seed, qualified):
    seed = Path(seed).resolve()
    manifest = json.loads((seed / 'manifest.json').read_text())
    state = json.loads((seed / 'state.json').read_text())
    assert all(state.get(k) for k in ('complete', 'cleanup_complete', 'semantic_passed', 'post_score_semantic_passed'))
    for key in ('commit', 'source', 'command', 'native_mode', 'stable_small_o', 'out_lowering', 'large_linear', 'gate_library_sha256'):
        assert manifest[key] == qualified[key], key
    assert state['commit'] == qualified['commit']
    assert manifest['native_mode'] == 'on'
    assert manifest['stable_small_o'] == manifest['out_lowering'] == manifest['large_linear'] == '1'
    return manifest


def restore(seed, target, qualified):
    seed, target = Path(seed).resolve(), Path(target).resolve()
    validate_seed(seed, qualified)
    root = seed / 'triton-cache'
    assert root.is_dir() and not root.is_symlink() and not target.exists()
    paths = sorted(root.rglob('*'))
    assert not any(p.is_symlink() for p in paths)
    originals = {str(p.relative_to(root)): sha(p) for p in paths if p.is_file()}
    assert originals
    shutil.copytree(root, target)
    assert all(sha(target / name) == digest for name, digest in originals.items())
    rebased = []
    for path in sorted(target.rglob('__grp__*.json')):
        group = json.loads(path.read_text())
        assert isinstance(group.get('child_paths'), dict)
        children = {}
        for name, value in group['child_paths'].items():
            relative = Path(value).resolve().relative_to(root)
            new = target / relative
            assert new.is_file() and sha(new) == originals[str(relative)]
            children[name] = str(new)
        group['child_paths'] = children
        path.write_text(json.dumps(group, separators=(',', ':')))
        rebased.append(dict(path=str(path.relative_to(target)), old_sha256=originals[str(path.relative_to(target))], new_sha256=sha(path)))
    for name, digest in originals.items():
        if not Path(name).name.startswith('__grp__'):
            assert sha(target / name) == digest, name
    return dict(seed_run=str(seed), seed_manifest_sha256=sha(seed / 'manifest.json'),
                seed_state_sha256=sha(seed / 'state.json'), copied_files=len(originals),
                source_file_sha256=originals, rebased_groups=rebased,
                kernel_artifact_bytes_unchanged=True, tuning_database_merged=False)
