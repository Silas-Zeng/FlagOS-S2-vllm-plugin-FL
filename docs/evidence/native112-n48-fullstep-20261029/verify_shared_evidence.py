#!/usr/bin/env python3
"""Lightweight exact archive/history verification; does not rerun numerical checks."""
import hashlib,io,json,tarfile
from pathlib import Path
D=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    index=json.loads((D/'evidence-files.json').read_text())
    for row in index:
        p=D/row['relative'];b=p.read_bytes()
        assert p.parent==D and len(b)==row['bytes'] and sha(b)==row['sha256']
        assert len(b)<=2*1024*1024
    meta=json.loads((D/'metadata-package.json').read_text());packed=(D/meta['package']).read_bytes()
    assert len(packed)==meta['bytes'] and sha(packed)==meta['sha256']
    members={r['relative']:r for r in meta['members']};manifest=json.loads((D/'original-download-manifest.json').read_text());original={r['relative']:r for r in manifest['files']}
    assert len(original)==118 and sum(r['bytes'] for r in original.values())==91645512
    assert set(members)-{'download-manifest.json'}=={n for n in original if not n.startswith('out/blobs/')}
    assert len(meta['omitted_blob_paths'])==meta['omitted_full_blob_files']==78
    for row in meta['omitted_blob_paths']:
        assert row['relative'].startswith('out/blobs/')
        assert all(row[k]==original[row['relative']][k] for k in ('bytes','sha256'))
    saved={}
    with tarfile.open(fileobj=io.BytesIO(packed),mode='r:gz') as tar:
        rows=tar.getmembers();assert len(rows)==len(members)==41 and {x.name for x in rows}==set(members)
        for member in rows:
            assert member.isfile() and '..' not in Path(member.name).parts and not Path(member.name).is_absolute()
            row=members[member.name];b=tar.extractfile(member).read();assert len(b)==row['bytes'] and sha(b)==row['sha256']
            if member.name=='download-manifest.json':assert b==(D/'original-download-manifest.json').read_bytes()
            else:assert row['sha256']==original[member.name]['sha256'] and row['bytes']==original[member.name]['bytes']
            if member.name in ('out/fullstep-summary.json','exports/cjj-native112-n48-fullstep-v1-20261029-owner.json'):saved[member.name]=b
    summary=json.loads(saved['out/fullstep-summary.json']);owner=json.loads(saved['exports/cjj-native112-n48-fullstep-v1-20261029-owner.json'])
    assert summary['GPU_executed'] is False and '--gpu' in owner['command']
    report=json.loads((D/'independent-review.json').read_text());history=json.loads((D/'audit-history-and-errata.json').read_text())
    for name,pin in [('independent-review.json','2d8d5b873eb30084da7dd0e25332088462dd5cfef62c45fe45c75bccd07ede5d'),('replay_fullstep.py','5e7a4911f6535c67c32735adf11286e635c0c08f56c476d1e1cf6443db75b978'),('original-numeric-audit.json','db985c037f33727c238ba4c7b885692f6ac548c702083f88f2ec3c941006699b')]:assert sha((D/name).read_bytes())==pin
    assert report['N48_adapter_script_sha256']==sha((D/'replay_fullstep.py').read_bytes())
    assert report['original_numeric_audit_report_sha256']==sha((D/'original-numeric-audit.json').read_bytes())
    assert report['manifest_sha256']==sha((D/'original-download-manifest.json').read_bytes())
    assert report['summary_sha256']==sha(saved['out/fullstep-summary.json'])
    assert report['passed'] and report['runtime_n_spills_zero_gate_preserved'] and report['ELF_SGPR_spill_all_zero'] is False
    assert history['original_GPU_executed_false_preserved'] and history['all_numeric_tolerances_unchanged']
    assert history['phase_ideal_partial_diagnostic_failures']=={c:dict(recorded=12,failed=12) for c in ('valid41','valid42')}
    assert 'AssertionError' in (D/'v1-failure.log').read_text() and 'AssertionError' in (D/'v2-failure.log').read_text()
    print(json.dumps(dict(shared_files_verified=len(index),original_metadata_members_verified=41,omitted_full_blob_files=78,full_raw_inventory_entries=118,original_false_flag_and_both_failures_retained=True,final_report_sha256=sha((D/'independent-review.json').read_bytes()),metadata_package_sha256=meta['sha256'],numeric_or_GPU_repeated=False),indent=2,sort_keys=True))
if __name__=='__main__':main()
