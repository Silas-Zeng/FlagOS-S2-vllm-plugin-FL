"""Read-only actual codec receipt gate; never invokes a device or framework."""
import json
from pathlib import Path
import struct
import biv150_native_k8_codec_fixture as f


def require_codec(summary_path,expected_sha):
    path=Path(summary_path);raw=path.read_bytes()
    assert type(expected_sha) is str and len(expected_sha)==64 and f.sha(raw)==expected_sha,'Actual codec receipt SHA required'
    result=json.loads(raw)
    assert result['completed'] and not result['errors'] and result['requested_codec_gates_passed']
    assert result['actual_codec_verified'] and result['Graph_verified'] and result['side_stream_verified']
    assert result['library']['sha256']==f.LIBRARY_SHA and result['library_sha256']==f.LIBRARY_SHA
    assert result['eager_op_calls']==3 and result['capture_op_calls']==1 and result['graph_replays']==1
    assert len(result['admission_rejections'])==6
    for row in result['artifact_manifest']:
        name=Path(row['relative']);assert not name.is_absolute() and '..' not in name.parts
        data=(path.parent/name).read_bytes();assert len(data)==row['bytes'] and f.sha(data)==row['sha256']
    fixture=f.fixtures()
    for stage,key in [('eagerA-after','A'),('graphB-after','B'),('eagerB-after','B'),('sideC-after','C')]:
        arrays={name:(path.parent/(stage+'-'+name+'.bin')).read_bytes() for name in ('input_full','code_full','scale_full')}
        checks,_=f.full_check(fixture[key],f.oracle(fixture[key]),arrays);assert all(checks.values()),stage
    for name in ('input_full','code_full','scale_full'):
        assert (path.parent/('admission-before-'+name+'.bin')).read_bytes()==(path.parent/('admission-after-'+name+'.bin')).read_bytes()
    n=len(fixture['B'])
    assert (path.parent/'graphB-before-input_full.bin').read_bytes()==f.guarded_input(fixture['B'])
    code=(path.parent/'graphB-before-code_full.bin').read_bytes();scale=(path.parent/'graphB-before-scale_full.bin').read_bytes()
    assert code[128:-128]==bytes([128])*(n*128) and scale[4:-4]==struct.pack('<I',f.SCALE_POISON)*n
    return dict(actual_codec_receipt_sha256=expected_sha,library_sha256=f.LIBRARY_SHA,
      all_actual_codec_snapshot_bits_independently_replayed=True,consumer_inherits_no_attention_or_model_qualification=True)
