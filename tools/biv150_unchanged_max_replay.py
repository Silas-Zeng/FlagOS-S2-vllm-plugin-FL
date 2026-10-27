"""Read and verify frozen v1 bytes only; never extract/import raw Python or cubins."""
import hashlib
import io
import json
import tarfile

from biv150_unchanged_max_ir import audit_artifacts, audit_ttgir


def load_v1(root, proof):
    evidence = root/'docs/evidence/gqa-unchanged-max-20261027'
    provenance = json.loads((evidence/'v1-provenance.json').read_text())
    archive = (evidence/provenance['archive']['path']).read_bytes()
    assert len(archive) == provenance['archive']['bytes'] <= 2*1024*1024
    assert hashlib.sha256(archive).hexdigest() == provenance['archive']['sha256']
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tar:
        members = tar.getmembers()
        assert len({m.name for m in members}) == len(members) == 27
        assert all(m.isfile() and m.size <= 2*1024*1024 for m in members)
        raw = {m.name: tar.extractfile(m).read() for m in members}
    manifest_bytes = raw['download-manifest.json']
    assert hashlib.sha256(manifest_bytes).hexdigest() == provenance['manifest_sha256']
    manifest = json.loads(manifest_bytes)
    assert len(manifest['files']) == provenance['raw_files'] == 26
    assert set(raw) == {'download-manifest.json'} | {f['relative'] for f in manifest['files']}
    for entry in manifest['files']:
        data = raw[entry['relative']]
        assert len(data) == entry['bytes']
        assert hashlib.sha256(data).hexdigest() == entry['sha256']
    assert sum(e['bytes'] for e in manifest['files']) == provenance['raw_bytes'] == 925773
    historic = json.loads(raw['out/result.json'])
    assert historic['source_commit'] == provenance['raw_run_source_commit']
    assert historic['proof']['generated_sha256'] == proof['generated_sha256'] == provenance['generated_sha256']
    assert hashlib.sha256(raw['out/generated-gqa.py']).hexdigest() == proof['generated_sha256']
    assert hashlib.sha256(raw['out/result.json']).hexdigest() == provenance['historic_result_sha256']
    assert hashlib.sha256(raw['exports/cjj-unchanged-max-v1-20261027.log']).hexdigest() == provenance['historic_log_sha256']
    assert not historic['records'][0]['passed'] and not historic['records'][0]['timing_attempted']
    assert historic['error'] == "ValueError('Missing conditional alpha/accumulator scale/explicit FMA/skip state')"
    candidate = {kind: raw['out/b64-candidate.'+kind].decode() for kind in ('ttgir','llir','asm')}
    audit = audit_artifacts(candidate, proof)
    negatives = [('v1-unconditional-baseline', raw['out/b64-baseline.ttgir'].decode())]
    old = provenance['pr71_negative']
    old_data = (evidence/old['path']).read_bytes()
    assert len(old_data) == old['bytes'] and hashlib.sha256(old_data).hexdigest() == old['sha256']
    negatives.append(('pr71-threshold8', old_data.decode()))
    for _, text in negatives:
        try:
            audit_ttgir(text)
        except ValueError:
            pass
        else:
            raise AssertionError('Old unconditional/non-FMA IR accepted')
    return candidate, dict(validator_version=2, cpu_ir_replay_passed=True,
                           raw_run_source_commit=historic['source_commit'],
                           raw_files_sha256_verified=26, raw_bytes_verified=925773,
                           archive=provenance['archive'], historic_result_unchanged=True,
                           historic_passed=False, historic_timing_attempted=False,
                           actual_ir_review=audit, old_ir_negative_labels=[label for label,_ in negatives],
                           historic_identity_primitive_passed=historic['primitive']['passed'],
                           full_gqa_numeric_checks_reached=False,
                           historic_post_launch_resources={role: {k:historic['records'][0][role][k]
                               for k in ('phase','hash','n_regs','n_spills','metadata')}
                               for role in ('baseline','candidate')},
                           gpu_numeric_acceptance=False, performance_acceptance=False), raw
