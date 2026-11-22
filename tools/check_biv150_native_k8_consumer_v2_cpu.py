#!/usr/bin/env python3
"""Replay all actual v1 partial elements against independent online-BF16P reference."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tarfile
import numpy as np
from biv150_native_k8_pagewise_reference import from_inputs
import biv150_native_k8_consumer_fixture as old
import biv150_native_k8_consumer_v2_factory as factory


def pack(raw,folder):
    files={str(p.relative_to(raw)):p.read_bytes() for p in raw.rglob('*') if p.is_file()}
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w',format=tarfile.USTAR_FORMAT) as tar:
        for name,data in sorted(files.items()):
            meta=tarfile.TarInfo(name);meta.size=len(data);meta.mode=0o644;meta.mtime=0;tar.addfile(meta,io.BytesIO(data))
    data=buf.getvalue();folder.mkdir(parents=True,exist_ok=True);parts=[]
    for i,start in enumerate(range(0,len(data),1900000)):
        fragment=data[start:start+1900000];name='original-part-'+str(i).zfill(2)+'.tar.gz';inner='original.tar.fragment-'+str(i).zfill(2)
        with (folder/name).open('wb') as target:
            with gzip.GzipFile(filename='',fileobj=target,mode='wb',mtime=0) as gz:
                with tarfile.open(fileobj=gz,mode='w',format=tarfile.USTAR_FORMAT) as tar:
                    row=tarfile.TarInfo(inner);row.size=len(fragment);row.mode=0o644;tar.addfile(row,io.BytesIO(fragment))
        b=(folder/name).read_bytes();assert len(b)<=2*1024*1024
        parts.append(dict(file=name,sha256=old.sha(b),member=inner,start=start,bytes=len(fragment),fragment_sha256=old.sha(fragment)))
    (folder/'archive-index.json').write_text(json.dumps(dict(original_tar_bytes=len(data),original_tar_sha256=old.sha(data),parts=parts),sort_keys=True,indent=2)+'\n')
    return files


def unpack(folder):
    index=json.loads((folder/'archive-index.json').read_text());data=bytearray()
    for row in index['parts']:
        blob=(folder/row['file']).read_bytes();assert old.sha(blob)==row['sha256']
        with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as tar:
            members=tar.getmembers();assert len(members)==1 and members[0].name==row['member'] and members[0].isfile();fragment=tar.extractfile(members[0]).read()
        assert len(data)==row['start'] and len(fragment)==row['bytes'] and old.sha(fragment)==row['fragment_sha256'];data.extend(fragment)
    assert len(data)==index['original_tar_bytes'] and old.sha(data)==index['original_tar_sha256']
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:') as tar:
        result={}
        for row in tar:
            assert row.isfile() and not Path(row.name).is_absolute() and '..' not in Path(row.name).parts
            assert row.name not in result;result[row.name]=tar.extractfile(row).read()
        return result


def check(files):
    manifest=json.loads(files['download-manifest.json']);assert manifest['raw_files']==34 and manifest['raw_bytes']==15880353
    assert set(files)=={'download-manifest.json',*(r['relative'] for r in manifest['files'])}
    for row in manifest['files']:
        b=files[row['relative']];assert len(b)==row['bytes'] and old.sha(b)==row['sha256']
    j=json.loads(files['out/consumer-summary.json']);assert not j['completed'] and j['producer_calls']==1 and j['reducer_calls']==0
    record=j['records'][0];n,s,pitch=record['case'];assert (n,s,pitch)==(64,2,2048)
    def blob(meta):
        b=files['out/'+meta['path']];assert len(b)==meta['bytes'] and old.sha(b)==meta['sha256'];return b
    actual={k:blob(v) for k,v in record['snapshots']['A-producer'].items()};before={k:blob(v) for k,v in record['snapshots']['A-before'].items()}
    c=old.fixture(n,s,pitch,0);ideal=old.golden(c);assert all(actual[k]==v==before[k] for k,v in old.input_bytes(c).items())
    new=from_inputs({k:actual[k] for k in ('query','cache','table','lengths')},n,s,pitch)
    mask=np.broadcast_to(ideal['active'][:,None,:,None],new['partial'].shape)
    bits=np.frombuffer(actual['partial'],dtype='<u4')[64:-64].reshape(new['partial'].shape);observed=bits.view('<f4').astype(np.float64)
    assert (bits[~mask]==old.F32_POISON).all()
    proxy=old.metrics(observed[mask],ideal['partial'][mask]);implementation=old.metrics(observed[mask],new['partial'][mask])
    no_round=old.metrics(observed[mask],new['without_P_round_control'][mask]);once=old.metrics(observed[mask],new['final_max_once_round_control'][mask])
    assert not proxy['passed'] and not no_round['passed'] and not once['passed'] and implementation['passed']
    identical=float(np.max(np.abs(new['without_P_round_control'][mask]-ideal['partial'][mask])));assert identical<1e-12
    violations=int(((np.abs(observed-ideal['partial'])>.005+.005*np.abs(ideal['partial']))&mask).sum());assert violations==4080
    # Literal source proof, independent from the numeric result.
    source=(old.ROOT/'csrc/iluvatar/native_gqa_s2.cu').read_text()
    assert source.index('new_m0=fmaxf')<source.index('p0=exp2f')<source.index('__float2bfloat16_rn(p0)')<source.index('acc[n][2]=acc[n][2]*rho0')
    from biv150_native_k8_source_factory import verify as kernel_inverse
    kernel=kernel_inverse();assert kernel['inverse_byte_exact']
    fp=factory.verify()
    # Changing the supposed reference-only frontend's numerical threshold fails
    # the exact inverse; thresholds themselves remain unchanged.
    changed=factory.DEST.read_text().replace("checks[field+'_FP64']=metrics[field]['passed']","checks[field+'_FP64']=True")
    try:factory.inverse(changed)
    except AssertionError:pass
    else:raise AssertionError('Reference frontend guard mutation accepted')
    assert not {'torch','triton'}.intersection(sys.modules)
    return dict(all34_original_SHA_verified=True,source_inverse=kernel,reference_frontend_inverse=fp,
      active_full_partial_elements=int(mask.sum()),implementation_pagewise_BF16_P_FP64=implementation,
      original_ideal_FP64_partial_diagnostic=proxy,original_ideal_pointwise_failures=violations,
      without_P_round_negative_control=no_round,final_max_once_round_negative_control=once,
      unrounded_online_equivalence_to_original_ideal_max_error=identical,
      actual_v1_status='failed, preserved',actual_reducer_calls=0,actual_final_output_unverified=True,
      new_consumer_case_Graph_side_device_qualification=False,thresholds_changed=False,
      source_dtype_kernel_changed=False,precision_gate=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path);p.add_argument('--archive-dir',type=Path,default=old.ROOT/'docs/evidence/native-k8-qk-feasibility-20261028/consumer-v1-failure');p.add_argument('--pack',action='store_true');a=p.parse_args()
    if a.pack:assert a.raw;files=pack(a.raw,a.archive_dir)
    elif a.raw:files={str(q.relative_to(a.raw)):q.read_bytes() for q in a.raw.rglob('*') if q.is_file()}
    else:files=unpack(a.archive_dir)
    print(json.dumps(check(files),sort_keys=True,indent=2))

if __name__=='__main__':main()
