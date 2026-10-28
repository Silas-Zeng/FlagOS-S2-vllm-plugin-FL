"""Stdlib-only component admission; model and real service capacity remain unverified."""
import hashlib
import os
from pathlib import Path
import re

ENV='VLLM_FL_BIV150_KONLY_COMPACT_NATIVE_DECODE'
PATH_ENV='VLLM_FL_BIV150_KONLY_COMPACT_NATIVE_LIBRARY'
SHA_ENV='VLLM_FL_BIV150_KONLY_COMPACT_NATIVE_LIBRARY_SHA256'
FORMAT='cjj-konly-compact12544-rn32-rne-v1'
OPERATOR='cjj_native_k8_compact_probe::produce'
QUALIFICATION={
    'state':'component-reviewed', 'format':FORMAT, 'operator':OPERATOR,
    'page_bytes':12544,
    'library_sha256':'40f9b17c634d2344ca54b9f1202e4e10932bf3f3f632cd41d793ddafd0934453',
    'actual_SO_review_sha256':'3b3e94c9af183b77df5be794746f2b783c1b925f643b5d1d450e13da3a3ddc28',
    'consumer_summary_sha256':'9d2bcd2f803626fca16aae5c73533ce1436d8bb98010f00fa6583061dfa90df8',
    'fullstep_summary_sha256':'53880d7cad7beb47a07a4551056b616afdba45e6fca4703da98aaea9816f036f',
    'fullstep_review_sha256':'96262cd2414ce85a3648f0d45e1727f93b243e754310e31e1160fc30639ebdb0',
    'source_CUDA_sha256':'3bd67edd58f1ea822a908dcc9217aceeef93124a7e9e00e6d7815dcdffe7a81d',
    'writer_source_sha256':'a0cacf3f13f450f1f633d0bf562b3b67239cdc4b4e2a9b1159bcbde4b3fa290f',
    'views_source_sha256':'1ab738e4b8178ee0d8ce12536ac7e6428ed09f7252ea7349c8d71f73bae64b1e',
    'scopes':[[64,2,2048],[64,2,2560],[32,4,2048],[32,4,2560]],
}


def need(value,message):
    if not value:raise RuntimeError('Compact NativeK8: '+message)


def pin():
    q=QUALIFICATION
    need(type(q) is dict and q.get('state')=='component-reviewed',
         'PENDING qualification; compact-native ON is forbidden before worker/library loading')
    required={'state','format','operator','page_bytes','library_sha256','actual_SO_review_sha256',
              'consumer_summary_sha256','fullstep_summary_sha256','fullstep_review_sha256','source_CUDA_sha256',
              'writer_source_sha256','views_source_sha256','scopes'}
    need(set(q)==required,'complete reviewed source pin required')
    need(q['format']==FORMAT and q['operator']==OPERATOR,'qualified format/namespace')
    for key in ('library_sha256','actual_SO_review_sha256','consumer_summary_sha256',
                'fullstep_summary_sha256','fullstep_review_sha256','source_CUDA_sha256','writer_source_sha256','views_source_sha256'):
        need(type(q[key]) is str and re.fullmatch('[0-9a-f]{64}',q[key]) is not None,'known nonempty SHA '+key)
    need(q['source_CUDA_sha256']=='3bd67edd58f1ea822a908dcc9217aceeef93124a7e9e00e6d7815dcdffe7a81d','PR100 address-only CUDA source pin')
    need(type(q['page_bytes']) is int and q['page_bytes']==12544,'compact physical page bytes')
    need(q['writer_source_sha256']=='a0cacf3f13f450f1f633d0bf562b3b67239cdc4b4e2a9b1159bcbde4b3fa290f'
         and q['views_source_sha256']=='1ab738e4b8178ee0d8ce12536ac7e6428ed09f7252ea7349c8d71f73bae64b1e',
         'unchanged dc5 encoder and compact role sources')
    need(type(q['scopes']) is list and all(type(row) is list and len(row)==3 and all(type(x) is int for x in row) for row in q['scopes']),'typed scope triples')
    need(q['scopes']==[[64,2,2048],[64,2,2560],[32,4,2048],[32,4,2560]],'only reviewed compact scopes')
    return dict(q)


def preflight(environ=None):
    env=os.environ if environ is None else environ
    if env.get(ENV,'0')!='1':return None
    q=pin()  # Deliberately before paths/hashes/config: missing None cannot pass.
    need(env.get('VLLM_FL_BIV150_KONLY_COMPACT_SERVICE','0')=='1','requires compact12544 service')
    need(env.get('VLLM_FL_BIV150_KONLY_ENCODED_SERVICE','0')!='1','two cache formats enabled')
    need(env.get('VLLM_FL_BIV150_KONLY_NATIVE_DECODE','0')!='1','16KiB native adapter is a different format')
    if env.get('VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST','0')=='1':
        # Source admission only. Every fresh JIT object must still pass the
        # unchanged compact adapter's binder/IR/native/resource gates at runtime.
        # Load standalone: C3 invokes this before any vllm_fl/framework import.
        import importlib.util
        source=Path(__file__).parent/'patches'/'biv150_konly_compact_identity.py'
        spec=importlib.util.spec_from_file_location('_cjj_compact_hoist_source_gate',source)
        identity=importlib.util.module_from_spec(spec);spec.loader.exec_module(identity)
        record=identity.verify_package()
        need(record.get('q32_hoist_opt_in_adapter_available') is True,
             'compact Q32 source adapter unavailable; old16KiB proof is invalid')
        for role,expected in {
            'q32':'a991c573a699c85d3d5ec2e9e85442871cc36992b51385750062160992883a63',
            'hoist_proposal':'1289fed8303b438a4150778964e5980446389cf260ba4ce707eec68e3bcff3d9',
            'q32_adapter':'68c13b3c0635062df74fe64f34d16021eaebf1d4750d6d4e0cda2ae7eaa0868d',
        }.items():
            item=record['generated'][role]
            need(item['path']=='vllm_fl/patches/biv150_konly_compact_'+role+'.py'
                 and item['sha256']==expected, 'unchanged 6c compact '+role+' source')
        need(record['hoist_dependencies']=={
            'biv150_konly_encoded_q32_adapter.py':'bb19f67d4e9ff80ca8510fe01ca5b1ad023d639ec267e5b51cc2d0d17bbfa04b',
            'biv150_konly_encoded_hoist_native.py':'2027e1ce4a489f6b47c97a5253c1482d2dc767e62e9b10ec029e3f192c559ef2',
        }, 'unchanged fresh object/binder/native/resource admission sources')
    path=Path(env.get(PATH_ENV,''))
    need(path.is_absolute() and path.is_file(),'existing absolute compact library path')
    need(type(env.get(SHA_ENV)) is str and env[SHA_ENV]==q['library_sha256'],'explicit actual library SHA pair')
    path=path.resolve()
    need(hashlib.sha256(path.read_bytes()).hexdigest()==q['library_sha256'],'actual compact library contents')
    for role in ('writer','views'):
        source=Path(__file__).parent/'patches'/('biv150_konly_compact_'+role+'.py')
        need(hashlib.sha256(source.read_bytes()).hexdigest()==q[role+'_source_sha256'],
             'actual compact '+role+' source contents')
    return dict(path=str(path),sha256=q['library_sha256'],qualification=q)
