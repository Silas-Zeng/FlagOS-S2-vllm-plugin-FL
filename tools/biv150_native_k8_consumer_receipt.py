"""Replay actual52c complete consumer snapshots before any full-step device work."""
import json
from pathlib import Path
import struct
import biv150_native_k8_consumer_v2_fixture as f


def require_consumer(path,expected_sha):
    import numpy as np
    path=Path(path);raw=path.read_bytes();assert expected_sha=='c2e13e3766172e9b74342bbcf48b1b4729d1b9604bc249b6974740f7f90cc128' and f.sha(raw)==expected_sha
    j=json.loads(raw);assert j['reference_version']=='pagewise_BF16_P_v2' and j['thresholds_changed'] is False
    assert j['final_output_hard_reference']=='unchanged ideal quantized-score FP64'
    assert j['completed'] and not j['errors'] and j['requested_consumer_gates_passed']
    assert j['consumer_numeric_verified'] and j['Graph_verified'] and j['side_stream_verified']
    assert j['library_sha256']==f.LIBRARY_SHA and j['codec_precondition']['actual_codec_receipt_sha256']=='3c9733ab79e69c77f70068cff2d3da553ad9c2b81b08ea3c396c6e0788b4f263'
    assert {tuple(r['case']) for r in j['records']}==set(f.CASES) and len(j['records'])==4
    for row in j['artifact_manifest']:
        name=Path(row['relative']);assert not name.is_absolute() and '..' not in name.parts
        b=(path.parent/name).read_bytes();assert len(b)==row['bytes'] and f.sha(b)==row['sha256']
    def load(meta):
        b=(path.parent/meta['path']).read_bytes();assert len(b)==meta['bytes'] and f.sha(b)==meta['sha256'];return b
    for record in j['records']:
        assert record['completed'] and all(all(c['checks'].values()) for c in record['checks'])
        n,s,pitch=record['case'];cases={p:f.fixture(n,s,pitch,p) for p in (0,1,2)};refs={p:f.golden(cases[p]) for p in cases}
        for stage,p,producer in [('A-producer',0,True),('A-after',0,False),('graphB-after',1,False),('eagerB-producer',1,True),('eagerB-after',1,False),('sideC-after',2,False)]:
            arrays={k:load(v) for k,v in record['snapshots'][stage].items()}
            assert all(arrays[k]==raw for k,raw in f.input_bytes(cases[p]).items())
            for field,shape in [('partial',(n,16,s,128)),('maximum',(n,16,s)),('summation',(n,16,s)),('output',(n,16,128))]:
                width=2 if field=='output' else 4;raw=arrays[field];guard=struct.pack('<H' if width==2 else '<I',f.BF16_GUARD if width==2 else f.F32_GUARD)*f.GUARD_ELEMENTS
                assert raw[:len(guard)]==raw[-len(guard):]==guard
                middle=raw[len(guard):-len(guard)]
                if field=='output':
                    bits=np.frombuffer(middle,dtype='<u2').reshape(shape)
                    if producer:assert bool((bits==f.BF16_POISON).all());continue
                    actual=(bits.astype(np.uint32)<<16).view('<f4').astype(np.float64)
                    assert f.metrics(actual,refs[p]['output'])['passed']
                    zero=np.array(cases[p]['lengths'][f.GUARD_ELEMENTS:-f.GUARD_ELEMENTS])==0;assert bool((bits[zero]==0).all())
                else:
                    bits=np.frombuffer(middle,dtype='<u4').reshape(shape);active=np.broadcast_to(refs[p]['active'][:,None,:,None] if field=='partial' else refs[p]['active'][:,None,:],shape)
                    assert bool((bits[~active]==f.F32_POISON).all());assert f.metrics(bits.view('<f4')[active],refs[p][field][active])['passed']
        assert {k:load(v) for k,v in record['snapshots']['graphB-after'].items()}=={k:load(v) for k,v in record['snapshots']['eagerB-after'].items()}
    return dict(actual_consumer_summary_sha256=expected_sha,all_full_D128_snapshots_replayed=True,
      cases=4,K8_library_sha256=f.LIBRARY_SHA,no_model_or_performance_qualification_inherited=True)
