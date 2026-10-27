#!/usr/bin/env python3
"""CPU NumPy full-reference/format/admission checks; no Torch/Triton/device."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from audit_biv150_native_k8_compile_cpu import from_archive
from biv150_native_k8_codec_receipt import require_codec
import biv150_native_k8_consumer_fixture as f


def codec_replay():
    path=f.ROOT/'docs/evidence/native-k8-qk-feasibility-20261028/codec-v1/original-codec.tar.gz'
    files=from_archive(path)
    with tempfile.TemporaryDirectory(prefix='cjj-codec-actual-replay-') as d:
        root=Path(d)
        for name,raw in files.items():p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        source=root/'out/codec-summary.json';proof=require_codec(source,f.sha(source.read_bytes()))
        try:require_codec(source,'0'*64)
        except AssertionError:pass
        else:raise AssertionError('Wrong receipt SHA admitted')
        raw=source.read_bytes();bad=raw.replace(b'"actual_codec_verified": true',b'"actual_codec_verified": false');assert bad!=raw;source.write_bytes(bad)
        try:require_codec(source,f.sha(bad))
        except AssertionError:pass
        else:raise AssertionError('Failed actual codec admitted')
    return proof

def check():
    import numpy as np
    proof=codec_replay();records=[];total=0
    for n,s,pitch in f.CASES:
        for phase in (0,1,2):
            c=f.fixture(n,s,pitch,phase);g=f.golden(c);total+=g['full_D128_integer_products']
            lens=c['lengths'][f.GUARD_ELEMENTS:-f.GUARD_ELEMENTS]
            assert np.isfinite(g['output']).all() and (g['output'][np.array(lens)==0]==0).all()
            for page in range(f.PAGES):
                for slot in range(16):
                    for head in range(2):
                        base=f.LAYER_OFFSET+page*16384
                        code=struct.unpack_from('<128b',c['cache'],base+slot*256+head*128);assert code==tuple(c['kcodes'][page][slot][head])
                        assert struct.unpack_from('<f',c['cache'],base+4096+slot*8+head*4)[0]==c['kscales'][page][slot][head]
                        assert struct.unpack_from('<128H',c['cache'],base+8192+slot*512+head*256)==tuple(c['value_bits'][page][slot][head])
            # Source full-D128 golden is not a16/32-D periodic factorization. A
            # D32 chunk permutation changes exact integer products, while a
            # scalar128-term sum crosschecks NumPy's full-width dot.
            row=next(i for i,L in enumerate(lens) if L>=17);page=c['table'][f.GUARD_ELEMENTS+row*f.COLS]
            q=c['qcodes'][row][3];k=c['kcodes'][page][0][0]
            direct=sum(a*b for a,b in zip(q,k));assert direct==int(np.array(q,dtype=np.int64)@np.array(k,dtype=np.int64))
            perm=k[32:64]+k[:32]+k[64:];assert sum(a*b for a,b in zip(q,perm))!=direct
            assert c['qscales'][row][2]!=c['qscales'][row][1] and c['qscales'][row][0]!=c['qscales'][row][4]
            for field in ('partial','maximum','summation'):
                mask=np.broadcast_to(g['active'][:,None,:,None] if field=='partial' else g['active'][:,None,:],g[field].shape)
                assert np.isfinite(g[field][mask]).all() and np.isnan(g[field][~mask]).all()
            assert f.metrics(g['output'],g['output'])['passed']
            bad=g['output'].copy();bad.reshape(-1)[-1]+=1.;assert not f.metrics(bad,g['output'])['passed']
            records.append(dict(case=[n,s,pitch],phase=phase,active_segments=int(g['active'].sum()),
              full_D128_integer_products=g['full_D128_integer_products'],format_exact=True,D32_permutation_detected=True))
    cli=subprocess.run([sys.executable,'-B',str(f.ROOT/'tools/probe_biv150_native_k8_consumer.py')],capture_output=True,text=True,check=True);assert json.loads(cli.stdout)==f.plan()
    bad=subprocess.run([sys.executable,'-B',str(f.ROOT/'tools/probe_biv150_native_k8_consumer.py'),'--gpu'],capture_output=True,text=True);assert bad.returncode!=0
    assert not {'torch','triton'}.intersection(sys.modules)
    return dict(plan=f.plan(),actual_codec_receipt=proof,records=records,total_full_D128_integer_products=total,
      no_Torch_or_Triton_import=True,consumer_device_verified=False,precision_gate=False)

if __name__=='__main__':print(json.dumps(check(),sort_keys=True,indent=2))
