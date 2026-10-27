"""Stdlib-only pre-worker admission. Source pin is deliberately PENDING."""
import hashlib
import os
from pathlib import Path
import re

ENV='VLLM_FL_BIV150_KONLY_COMPACT_NATIVE_DECODE'
PATH_ENV='VLLM_FL_BIV150_KONLY_COMPACT_NATIVE_LIBRARY'
SHA_ENV='VLLM_FL_BIV150_KONLY_COMPACT_NATIVE_LIBRARY_SHA256'
FORMAT='cjj-konly-compact12544-rn32-rne-v1'
OPERATOR='cjj_native_k8_compact_probe::produce'
QUALIFICATION={'state':'pending','reason':'No actual compact SO SHA/static review/consumer receipt is qualified'}


def need(value,message):
    if not value:raise RuntimeError('Compact NativeK8: '+message)


def pin():
    q=QUALIFICATION
    need(type(q) is dict and q.get('state')=='component-reviewed',
         'PENDING qualification; compact-native ON is forbidden before worker/library loading')
    required={'state','format','operator','library_sha256','actual_SO_review_sha256',
              'consumer_summary_sha256','source_CUDA_sha256','scopes'}
    need(set(q)==required,'complete reviewed source pin required')
    need(q['format']==FORMAT and q['operator']==OPERATOR,'qualified format/namespace')
    for key in ('library_sha256','actual_SO_review_sha256','consumer_summary_sha256','source_CUDA_sha256'):
        need(type(q[key]) is str and re.fullmatch('[0-9a-f]{64}',q[key]) is not None,'known nonempty SHA '+key)
    need(q['source_CUDA_sha256']=='3bd67edd58f1ea822a908dcc9217aceeef93124a7e9e00e6d7815dcdffe7a81d','PR100 address-only CUDA source pin')
    need(type(q['scopes']) is list and all(type(row) is list and len(row)==3 and all(type(x) is int for x in row) for row in q['scopes']),'typed scope triples')
    need(q['scopes']==[[64,2,2048],[64,2,2560],[32,4,2048],[32,4,2560]],'only reviewed compact scopes')
    return dict(q)


def preflight(environ=None):
    env=os.environ if environ is None else environ
    if env.get(ENV,'0')!='1':return None
    q=pin()  # Deliberately before paths/hashes/config: missing None cannot pass.
    need(env.get('VLLM_FL_BIV150_KONLY_COMPACT_SERVICE','0')=='1','requires compact12544 service')
    need(env.get('VLLM_FL_BIV150_KONLY_ENCODED_SERVICE','0')!='1','two cache formats enabled')
    need(env.get('VLLM_FL_BIV150_KONLY_NATIVE_DECODE','0')!='1','old22e adapter is a different format')
    need(env.get('VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST','0')!='1','old16KiB hoist proof is invalid')
    path=Path(env.get(PATH_ENV,''))
    need(path.is_absolute() and path.is_file(),'existing absolute compact library path')
    need(type(env.get(SHA_ENV)) is str and env[SHA_ENV]==q['library_sha256'],'explicit actual library SHA pair')
    path=path.resolve()
    need(hashlib.sha256(path.read_bytes()).hexdigest()==q['library_sha256'],'actual compact library contents')
    return dict(path=str(path),sha256=q['library_sha256'],qualification=q)
