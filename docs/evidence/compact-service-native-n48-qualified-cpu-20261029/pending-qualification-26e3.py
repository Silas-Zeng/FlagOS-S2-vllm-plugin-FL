"""Stdlib-only component admission; model and real service capacity remain unverified."""
import hashlib
import os
from pathlib import Path
import re

ENV='VLLM_FL_BIV150_KONLY_COMPACT_NATIVE_DECODE'
PATH_ENV='VLLM_FL_BIV150_KONLY_COMPACT_NATIVE_LIBRARY'
SHA_ENV='VLLM_FL_BIV150_KONLY_COMPACT_NATIVE_LIBRARY_SHA256'
FORMAT='cjj-konly-compact12544-rn32-rne-v1'
OPERATOR='cjj_native_k8_compact_v_overlap_probe::produce'
N48_ENV='VLLM_FL_BIV150_KONLY_COMPACT_NATIVE_N48'
N48_PATH_ENV='VLLM_FL_BIV150_KONLY_COMPACT_NATIVE_N48_LIBRARY'
N48_SHA_ENV='VLLM_FL_BIV150_KONLY_COMPACT_NATIVE_N48_LIBRARY_SHA256'
N48_OPERATOR='cjj_native_k8_compact_n48_s4_probe::produce'
# Complete these records only in a reviewed commit with the corresponding actual
# consumer/fullstep evidence. Compile or parent/service results cannot fill None.
QUALIFICATION={'state': 'component-reviewed',
 'format': 'cjj-konly-compact12544-rn32-rne-v1',
 'operator': 'cjj_native_k8_compact_v_overlap_probe::produce',
 'page_bytes': 12544,
 'library_sha256': '7892a6aad09d247a322a74906312f250a9d253160202fee23259adac7b7260d4',
 'actual_SO_review_sha256': '0b3d0098f3e8b01b553fc75b44dbe6c5eb4b898d61076149f96e02fc09697708',
 'consumer_summary_sha256': '6f719fe84e83e547d0e42f7c2be5bdb408e64cdb107cc455994a681705e34c7c',
 'consumer_review_sha256': '896c7c992aefcbfa36ed7445c3cf6fb1d40135bde610b5d83693ed27ec8fadf1',
 'fullstep_summary_sha256': 'f12332f0b2397ed17baaadbad5e5d189c326188e1b6b9a05c8d4084ef7667478',
 'fullstep_review_sha256': '0cd4cb1682fa359f566ac7bf2f076c7861761a4cb8783600ea2cdf551acf051d',
 'source_CUDA_sha256': '99a3f0a9868ed08d260956bef39315b99face6b1069ee23d8ebf6e58f6f1a3ce',
 'writer_source_sha256': 'a0cacf3f13f450f1f633d0bf562b3b67239cdc4b4e2a9b1159bcbde4b3fa290f',
 'views_source_sha256': '1ab738e4b8178ee0d8ce12536ac7e6428ed09f7252ea7349c8d71f73bae64b1e',
 'scopes': [[64, 2, 2048], [64, 2, 2560], [32, 4, 2048], [32, 4, 2560]],
 'fullstep_geometry': {'physical_pages': 34276, 'new_KV_row_pitch': 256, 'service_verified': False}}
N48_QUALIFICATION={'state': 'pending',
 'format': 'cjj-konly-compact12544-rn32-rne-v1',
 'operator': 'cjj_native_k8_compact_n48_s4_probe::produce',
 'page_bytes': 12544,
 'library_sha256': '2ad51a9812502e7dd8d81b99debd2160086a52510a93063f7ce4b0fbd8c837b1',
 'actual_SO_review_sha256': '30c4c7daf2ccf3839bfabc9936f0a845400d26242fe93cc95bec99d270e8f398',
 'consumer_summary_sha256': '493e210f8eeaac6006b25a94ed55e08fbe932337d6481fe3d8ec6e86d0033272',
 'consumer_review_sha256': 'c07937c3d6c3ffc78a5c7b213271e4f864a9463775fe932f9f2aadbff08ef4d0',
 'fullstep_summary_sha256': None,
 'fullstep_review_sha256': None,
 'source_CUDA_sha256': '1aca8c4be8d8d8e8d818cad529c0716d16b1e30a871ee3c8ec48b12d62338ab7',
 'writer_source_sha256': 'a0cacf3f13f450f1f633d0bf562b3b67239cdc4b4e2a9b1159bcbde4b3fa290f',
 'views_source_sha256': '1ab738e4b8178ee0d8ce12536ac7e6428ed09f7252ea7349c8d71f73bae64b1e',
 'scopes': [[48, 4, 2048], [48, 4, 2560]],
 'fullstep_geometry': None}


def need(value,message):
    if not value:raise RuntimeError('Compact NativeK8: '+message)


def pin(scope='base'):
    need(scope in ('base','n48'),'known qualification scope')
    q=QUALIFICATION if scope=='base' else N48_QUALIFICATION
    need(type(q) is dict and q.get('state')=='component-reviewed',
         'PENDING '+scope+' qualification; compact-native ON is forbidden before worker/library loading')
    required={'state','format','operator','page_bytes','library_sha256','actual_SO_review_sha256',
              'consumer_summary_sha256','consumer_review_sha256','fullstep_summary_sha256','fullstep_review_sha256',
              'source_CUDA_sha256','writer_source_sha256','views_source_sha256','scopes','fullstep_geometry'}
    need(set(q)==required,'complete reviewed source pin required')
    expected=(('cjj_native_k8_compact_v_overlap_probe::produce',
      '7892a6aad09d247a322a74906312f250a9d253160202fee23259adac7b7260d4',
      '99a3f0a9868ed08d260956bef39315b99face6b1069ee23d8ebf6e58f6f1a3ce',
      [[64,2,2048],[64,2,2560],[32,4,2048],[32,4,2560]]) if scope=='base' else
      ('cjj_native_k8_compact_n48_s4_probe::produce',
      '2ad51a9812502e7dd8d81b99debd2160086a52510a93063f7ce4b0fbd8c837b1',
      '1aca8c4be8d8d8e8d818cad529c0716d16b1e30a871ee3c8ec48b12d62338ab7',
      [[48,4,2048],[48,4,2560]]))
    need(q['format']==FORMAT and (q['operator'],q['library_sha256'],q['source_CUDA_sha256'],q['scopes'])==expected,
         'exact actual library/source/namespace/scopes')
    for key in required-{'state','format','operator','page_bytes','scopes','fullstep_geometry'}:
        need(type(q[key]) is str and re.fullmatch('[0-9a-f]{64}',q[key]) is not None,'known nonempty SHA '+key)
    need(type(q['page_bytes']) is int and q['page_bytes']==12544,'compact physical page bytes')
    geometry=q['fullstep_geometry']
    need(type(geometry) is dict and set(geometry)=={'physical_pages','new_KV_row_pitch','service_verified'}
         and type(geometry['physical_pages']) is int and geometry['physical_pages']==(34276 if scope=='base' else 44769)
         and type(geometry['new_KV_row_pitch']) is int and geometry['new_KV_row_pitch']==(256 if scope=='base' else 2560)
         and geometry['service_verified'] is False,'component fullstep geometry, not shared model allocation/service qualification')
    need(q['writer_source_sha256']=='a0cacf3f13f450f1f633d0bf562b3b67239cdc4b4e2a9b1159bcbde4b3fa290f'
         and q['views_source_sha256']=='1ab738e4b8178ee0d8ce12536ac7e6428ed09f7252ea7349c8d71f73bae64b1e',
         'unchanged dc5 encoder and compact role sources')
    need(type(q['scopes']) is list and all(type(row) is list and len(row)==3 and all(type(x) is int for x in row) for row in q['scopes']),'typed scope triples')
    import copy
    return copy.deepcopy(q)


def library_pair(env,path_key,hash_key,qualification):
    path=Path(env.get(path_key,''))
    need(path.is_absolute() and path.is_file(),'existing absolute compact library path')
    need(type(env.get(hash_key)) is str and env[hash_key]==qualification['library_sha256'],'explicit actual library SHA pair')
    path=path.resolve()
    need(hashlib.sha256(path.read_bytes()).hexdigest()==qualification['library_sha256'],'actual compact library contents')
    return dict(path=str(path),sha256=qualification['library_sha256'],qualification=qualification)


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
    result=library_pair(env,PATH_ENV,SHA_ENV,q)
    result['n48']=None
    if env.get(N48_ENV,'0')=='1':
        result['n48']=library_pair(env,N48_PATH_ENV,N48_SHA_ENV,pin('n48'))
    for role in ('writer','views'):
        source=Path(__file__).parent/'patches'/('biv150_konly_compact_'+role+'.py')
        need(hashlib.sha256(source.read_bytes()).hexdigest()==q[role+'_source_sha256'],
             'actual compact '+role+' source contents')
    return result
