"""Stdlib compile admission and process-local live storage/epoch contract."""
import json
from pathlib import Path
import re

import biv150_native_k8_compact_factory as f


def artifact(root, row):
    root=Path(root).resolve();name=Path(row['relative'])
    assert not name.is_absolute() and '..' not in name.parts
    path=(root/name).resolve();assert path.is_relative_to(root)
    raw=path.read_bytes()
    assert len(raw)==row['bytes'] and f.sha(raw)==row['sha256']
    return raw


def compile_binding(summary):
    summary=Path(summary).resolve();raw=summary.read_bytes();j=json.loads(raw)
    assert j['completed'] is True and j['compile_passed'] is True and not j['errors']
    assert j['source_transform']==f.verify()
    assert all(type(j[k]) is int and j[k]==0 for k in
        ('attention_launches','encoder_launches','codec_launches','reducer_launches','input_initializations'))
    rows={}
    for row in j['artifact_manifest']:
        assert row['relative'] not in rows
        artifact(summary.parent,row);rows[row['relative']]=row
    for role in f.FILES:
        local=f.path(role)
        assert artifact(summary.parent,rows[local.name])==local.read_bytes()
    libraries=[row for name,row in rows.items() if Path(name).name==Path(j['library']['path']).name]
    assert len(libraries)==1 and libraries[0]['sha256']==j['library']['sha256']
    header=f.ROOT/'tools/kernels/biv150_native_k8_codec.cuh'
    assert artifact(summary.parent,rows[header.name])==header.read_bytes()
    import probe_biv150_native_k8_compact as frontend
    assert j['explicit_cuda_flags']==frontend.build_module().CUDA_FLAGS
    inspection_path=summary.parent/'actual-so-inspection.json'
    inspection_raw=inspection_path.read_bytes();i=json.loads(inspection_raw)
    assert i['completed'] is True and not i['errors']
    assert i['compile_summary_sha256']==f.sha(raw) and i['library']==j['library']
    assert re.fullmatch('[0-9a-f]{64}',j['library']['sha256'])
    assert len(i['commands'])==2
    for row,label in zip(i['commands'],('sass','IR')):
        assert type(row['exit_code']) is int and row['exit_code']==0
        assert row['argv']==[i['objdump'],'--'+label,j['library']['path']]
    outputs={row['relative']:row for row in i['artifact_manifest']}
    for row in outputs.values():artifact(summary.parent,row)
    required=('actual-so-sass.stdout.txt','actual-so-IR.stdout.txt')
    assert all(outputs[k]['bytes']>0 for k in required)
    return dict(compile_summary_sha256=f.sha(raw),inspection_sha256=f.sha(inspection_raw),
        library_sha256=j['library']['sha256'],actual_SO_SASS_sha256=outputs[required[0]]['sha256'],
        actual_SO_IR_sha256=outputs[required[1]]['sha256'],source_proof=f.verify())


REVIEW_FLAGS=('actual_machine_code_bound_to_SO','actual_compact_addresses_and_SME_descriptor_verified',
              'actual_wait_tail_QK_PV_math_verified','actual_codec_preserved')


def review_template(summary):
    return dict(version=1,**compile_binding(summary),reviewer='CJJ1008',
        **{k:False for k in REVIEW_FLAGS},
        static_resources={key:dict(vgpr=None,sgpr=None,shared_bytes=None,private_segment_bytes=None,
            vgpr_spills=None,sgpr_spills=None) for key in ('S2','S4')},
        notes='UNREVIEWED: final SO addresses, wait/tail/maths, codec and resources require Root review')


def admission(summary,review_path,review_sha,library):
    binding=compile_binding(summary);raw=Path(review_path).read_bytes();j=json.loads(raw)
    assert f.sha(raw)==review_sha
    for key,value in binding.items():assert j[key]==value,key
    assert type(j['version']) is int and j['version']==1 and j['reviewer']=='CJJ1008'
    assert all(j[k] is True for k in REVIEW_FLAGS),'Actual compact SO remains unreviewed'
    assert type(j['notes']) is str and len(j['notes'])>=20 and not j['notes'].startswith('UNREVIEWED')
    assert set(j['static_resources'])=={'S2','S4'}
    for resource in j['static_resources'].values():
        assert set(resource)=={'vgpr','sgpr','shared_bytes','private_segment_bytes','vgpr_spills','sgpr_spills'}
        for key,value in resource.items():
            assert type(value) is int and value >= (1 if key in ('vgpr','sgpr') else 0),key
        assert resource['shared_bytes']==8192
    assert f.sha(Path(library).read_bytes())==binding['library_sha256']
    return dict(**binding,review_sha256=review_sha,review=j,
                qualification_inherited_from_original22e=False)


class LiveEpoch:
    """Hold real tensors strongly; a copied receipt/JSON is never an admission.

    The device caller owns allocation/upload and supplies an observer and actual
    GPU->CPU readback. activate() reads the complete live cache before granting a
    fresh, process-local identity lease. Replays explicitly revalidate that lease.
    """
    def __init__(self,raw,views,observe,readback,*,pages,layer_offset,format_id):
        assert format_id==f.FORMAT
        self.raw=raw;self.views=dict(views);self.observe=observe;self.readback=readback
        self.pages=pages;self.layer_offset=layer_offset;self.format_id=format_id
        self.initial=observe(raw);self.active=True;self.token=None;self.generation=0
        assert self.initial['dtype']=='uint8' and self.initial['shape']==(layer_offset+pages*f.PAGE+128,)
        assert self.initial['strides']==(1,) and self.initial['device']=='cuda:0'
        assert self.initial['pointer']==self.initial['storage_base'] and self.initial['storage_offset']==0
        assert self.initial['storage_bytes']==self.initial['shape'][0]
        self.expected_bytes=None

    def _roles(self):
        result={}
        for key,tensor in self.views.items():
            d=self.observe(tensor)
            assert d['device']=='cuda:0'
            assert d['storage_base']==self.initial['storage_base'] and d['storage_bytes']==self.initial['storage_bytes']
            width={'int8':1,'float32':4,'bfloat16':2}[d['dtype']]
            assert type(d['storage_offset']) is int and d['storage_offset']>=0
            assert d['pointer']==d['storage_base']+d['storage_offset']*width
            result[key]={k:d[k] for k in ('dtype','shape','strides','pointer','storage_id')}
        return result

    def activate(self,expected):
        assert self.active and type(expected) is bytes and len(expected)==self.initial['storage_bytes']
        self.token=None;self.expected_bytes=None
        assert self.readback(self.raw)==expected,'Live cache full readback differs from CPU upload'
        self.expected_bytes=expected;self.generation+=1;self.token=object()
        self.admit(self.token)
        return self.token

    def admit(self,token,*,readonly=(),outputs=()):
        assert self.active and self.format_id==f.FORMAT
        assert token is not None and token is self.token,'Stale/missing/foreign epoch identity'
        assert self.observe(self.raw)==self.initial,'Live allocation identity/layout changed'
        receipt=dict(format=self.format_id,epoch=self.generation,purpose='Root-owned synthetic component fixture',
            storage_id=self.initial['storage_id'],layer_base=self.initial['pointer']+self.layer_offset,
            pages=self.pages,layer_bytes=self.pages*f.PAGE,layout_checked=self.expected_bytes is not None)
        f.admit_fixture(receipt,self._roles(),epoch=self.generation)
        other=[self.observe(t) for t in (*readonly,*outputs)]
        cache=(self.initial['storage_base'],self.initial['storage_base']+self.initial['storage_bytes'])
        ranges=[]
        for d in other:
            assert d['device']=='cuda:0'
            span=(d['storage_base'],d['storage_base']+d['storage_bytes'])
            assert max(cache[0],span[0])>=min(cache[1],span[1]),'Cache/readonly/output backing alias'
            assert all(max(span[0],a)>=min(span[1],b) for a,b in ranges),'Other input/output backing alias'
            ranges.append(span)
        return dict(**receipt,raw=self.initial,roles=self._roles(),
            live_cache_readback_sha256=f.sha(self.expected_bytes),lease_is_process_local=True)

    def close(self):
        self.active=False;self.token=None
