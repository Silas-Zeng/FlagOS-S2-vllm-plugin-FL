#!/usr/bin/env python3
"""Reuse the exact c079 PR112 full saved-array replay; only actual113 identities adapt."""
import ast,hashlib,json,sys
from pathlib import Path
ORIGINAL='docs/evidence/native112-n48-consumer-20261029/replay_consumer.py'
ORIGINAL_SHA='47b58473c537ee6e8b1cdd06aed711ff7cf7c5c7661727da83ce3afcf3bb6e2b'
# Pending: populated only after Root supplies the complete actual c91 raw archive.
ACTUAL={'summary_sha256': '17966c98d1f8056af1b35a70d5f8639ab46ce1a0fe723833898a4a8fc5dbfa59', 'download_manifest_sha256': '1152d78231499303565697e5424f91b77e0102e1d899bc63e3f61de3c7c6ba56', 'owner': {'pid': 78898, 'start_ticks': 357135455, 'same_start': True, 'state': 'Z'}, 'raw_files': 175, 'raw_bytes': 59716288}

def adapted_source(root):
    raw=(root/ORIGINAL).read_bytes();assert hashlib.sha256(raw).hexdigest()==ORIGINAL_SHA
    edits=[
      ("FROZEN='09e502134fcf9fa7168c9f342c2a3d33754cee1c'","FROZEN='c91f1cac3b7e69e03cee8dab9f7aa7b313674d8b'"),
      ("SUMMARY='493e210f8eeaac6006b25a94ed55e08fbe932337d6481fe3d8ec6e86d0033272'","SUMMARY=ACTUAL['summary_sha256']"),
      ("CANDIDATE='2ad51a9812502e7dd8d81b99debd2160086a52510a93063f7ce4b0fbd8c837b1'","CANDIDATE='5cf698e8a60272d86b2545364f580faf4c40114e0044e91557c1a137f2521bed'"),
      ("    assert manifest['raw_files']==170 and manifest['raw_bytes']==59569068\n    assert manifest['original_owner']==dict(pid=62050,start_ticks=355994499,same_start=True,state='Z')", "    assert sha((raw/'download-manifest.json').read_bytes())==ACTUAL['download_manifest_sha256']\n    assert manifest['raw_files']==ACTUAL['raw_files'] and manifest['raw_bytes']==ACTUAL['raw_bytes']\n    assert manifest['original_owner']==ACTUAL['owner']"),
      ("    assert len(seen)==170 and sum(x['bytes'] for x in manifest['files'])==59569068", "    assert len(seen)==ACTUAL['raw_files'] and sum(x['bytes'] for x in manifest['files'])==ACTUAL['raw_bytes']"),
      ("    import probe_biv150_native_k8_compact_n48_consumer as front", "    import probe_biv150_native_k8_compact_n48_shared6k_consumer as adapter\n    front=SimpleNamespace(**adapter.frontend()[0])\n    actual_text,actual_proof=adapter.adapted_source()\n    meta=s['executed_consumer_source']\n    assert meta['relative']=='executed-consumer-source.py'\n    assert (out/meta['relative']).read_bytes()==actual_text.encode()\n    assert meta['bytes']==len(actual_text.encode()) and meta['sha256']==sha(actual_text.encode())\n    assert meta['source_adapter']==actual_proof"),
      ("      raw_files_rehashed=170,raw_bytes_rehashed=59569068,", "      raw_files_rehashed=ACTUAL['raw_files'],raw_bytes_rehashed=ACTUAL['raw_bytes'],identity_adaptation=ADAPTATION,actual_executed_source=meta,")]
    text=raw.decode();changes=[]
    for old,new in edits:
        assert text.count(old)==1 and new not in text,(old,text.count(old));text=text.replace(old,new);changes.append(dict(old=old,new=new))
    restored=text
    for old,new in reversed(edits):assert restored.count(new)==1;restored=restored.replace(new,old)
    assert restored.encode()==raw;ast.parse(text)
    # All numerical replay code from numpy import through the final counts/proxy gates is verbatim.
    begin='    import numpy as np\n';end='    report=dict(completed=True,'
    assert text[text.index(begin):text.index(end)]==raw.decode()[raw.decode().index(begin):raw.decode().index(end)]
    proof=dict(original_path=ORIGINAL,original_sha256=ORIGINAL_SHA,inverse_byte_exact=True,full_numeric_replay_body_byte_exact=True,
      original_thresholds_unchanged=True,original_fixtures_golden_metrics_check_core_bytes_unchanged=True,
      actual_executed_replay_sha256=hashlib.sha256(text.encode()).hexdigest(),edits=changes)
    return text,proof

def main():
    assert '--source' in sys.argv
    root=Path(sys.argv[sys.argv.index('--source')+1]).resolve()
    text,proof=adapted_source(root)
    if '--plan' in sys.argv:
        print(json.dumps(dict(status='pending actual consumer raw',actual=ACTUAL,adaptation=proof,arrays_replayed=False,GPU_executed=False),indent=2,sort_keys=True));return
    assert all(v is not None for v in ACTUAL.values()),'Actual113 raw identities are pending; no array replay or qualification'
    for k in ('summary_sha256','download_manifest_sha256'):assert len(ACTUAL[k])==64
    ns=dict(__name__='cjj_actual113_reused_c079_review',__file__=__file__,ACTUAL=ACTUAL,ADAPTATION=proof)
    exec(compile(text,'<actual113-c079-full-array-replay>','exec'),ns)
    ns['main']()
if __name__=='__main__':main()
