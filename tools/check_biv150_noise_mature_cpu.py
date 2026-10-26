"""CPU-only contracts for exact cache restore, owned cleanup and WAL snapshot."""
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

from biv150_provider_cache_restore import restore, validate_seed, sha
from run_biv150_provider_noise import derive, snapshot_database, tuning_config_fingerprint


def rejected(call):
    try:call()
    except (AssertionError, ValueError):return
    raise AssertionError('Unsafe restore accepted')


def main(source):
    base=source/'tools/run_biv150_small_out_acceptance.py'
    for cached in (False, True):
        text=derive(base.read_text(),sha(base),restore_cache=cached)
        compile(text,'owned-fixture','exec')
        assert text.count('_noise_cache_restore(')==int(cached)
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary).resolve();seed=root/'seed';cache=seed/'triton-cache';cache.mkdir(parents=True)
        qualified=dict(commit='frozen',source='/frozen',command=['fixed','serve'],native_mode='on',stable_small_o='1',out_lowering='1',large_linear='1',gate_library_sha256='library')
        state=dict(commit='frozen',complete=True,cleanup_complete=True,semantic_passed=True,post_score_semantic_passed=True)
        (seed/'manifest.json').write_text(json.dumps(qualified));(seed/'state.json').write_text(json.dumps(state))
        binary=cache/'key/kernel.bin';binary.parent.mkdir();binary.write_bytes(b'\x00\xff\x01exact-bytes')
        group=cache/'key/__grp__kernel.json';group.write_text(json.dumps({'child_paths':{'kernel.bin':str(binary)}}))
        proof=restore(seed,root/'restored',qualified)
        assert proof['kernel_artifact_bytes_unchanged'] and proof['copied_files']==2
        assert (root/'restored/key/kernel.bin').read_bytes()==binary.read_bytes()
        rebased=json.loads((root/'restored/key/__grp__kernel.json').read_text())
        assert rebased['child_paths']['kernel.bin']==str(root/'restored/key/kernel.bin')
        rejected(lambda:restore(seed,root/'restored',qualified))
        for name,value in [('commit','other'),('large_linear','0'),('source','/other'),('gate_library_sha256','other')]:
            rejected(lambda name=name,value=value:validate_seed(seed,dict(qualified,**{name:value})))
        (seed/'state.json').write_text(json.dumps(dict(state,cleanup_complete=False)))
        rejected(lambda:validate_seed(seed,qualified));(seed/'state.json').write_text(json.dumps(state))
        (cache/'symlink').symlink_to(binary)
        rejected(lambda:restore(seed,root/'symlink-rejected',qualified));(cache/'symlink').unlink()
        group.write_text(json.dumps({'child_paths':{'bad':str(root/'outside')}}))
        rejected(lambda:restore(seed,root/'escape-rejected',qualified))
        configs=root/'configs';configs.mkdir();db=configs/'config.db';connection=sqlite3.connect(db)
        connection.execute('PRAGMA journal_mode=WAL');connection.execute('CREATE TABLE config(key INTEGER,value TEXT)')
        connection.execute("INSERT INTO config VALUES(1,'committed-in-WAL')");connection.commit()
        assert Path(str(db)+'-wal').exists()
        backup=root/'backup';backup.mkdir();snapshot_database(db,backup/'config.db')
        assert tuning_config_fingerprint(configs)==tuning_config_fingerprint(backup)
        connection.close()
    assert 'torch' not in sys.modules
    result=dict(passed=True,cleanup_ast_preserved=True,cache_bytes_and_group_rebase=True,
                completed_matching_seed_required=True,symlink_and_escape_rejected=True,
                wal_snapshot_preserved=True,torch_imported=False)
    print(json.dumps(result))


if __name__=='__main__':
    main(Path(sys.argv[1]))
