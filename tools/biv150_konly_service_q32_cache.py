"""Read exact runtime-selected 359 cache and verified static source; CPU only."""
import ast
import io
import json
from pathlib import Path
import re
import tarfile
import biv150_konly_service_q32_hoist as h
ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'docs/evidence/konly-service-q32-hoist-20261027/service359-cache-v1'
ARCHIVE_SHA='a73d7f5295ae251d4b9caacbff07b9f9f8ae2c70c5d91790c9307d3022298e2a'
SOURCE_SHA='ac0ee4caaeed8dfd211a57cd4d3d130d8437a2a0129c1c915e1b19ed4fc66b4b'
RUNTIME_SHA='a4f0de54223d7c2a088d84be94f974438f0688599be169a4d69cec750b2fbe01'


def original():
    b=(DIRECTORY/'original-cache.tar.gz').read_bytes();h.need(h.sha(b)==ARCHIVE_SHA,'actual selected-cache archive SHA')
    with tarfile.open(fileobj=io.BytesIO(b)) as tar:
        rows=tar.getmembers();h.need(len(rows)==9 and all(r.isfile() and '/' not in r.name for r in rows),'exact cache members')
        files={r.name:tar.extractfile(r).read() for r in rows}
    m=json.loads(files['download-manifest.json'])
    h.need(m['source_native_snapshot_rechecked_unchanged'] is True and m['live_compiled_object_introspection'] is False,'cache provenance kind')
    h.need(set(files)=={r['name'] for r in m['files']}|{'download-manifest.json'},'all original cache files')
    for r in m['files']:h.need((len(files[r['name']]),h.sha(files[r['name']]))==(r['bytes'],r['sha256']),'original cache member SHA')
    c=json.loads(files['kernel_cjj_konly_q32.json']);e=m['selected_event']
    h.need(c['hash']==m['actual_worker_cache_matching_hash']==e['hash'] and e['profile']=='q32'
           and e['query_rows']==2048 and e['sequences']==1 and e['format']==h.FORMAT,'actual runtime selected/cache binding')
    source=(DIRECTORY/'verified359-q32.py.txt').read_text();h.need(h.sha(source)==SOURCE_SHA,'verified359 q32 module SHA')
    runtime=(DIRECTORY/'verified359-runtime.py.txt').read_text();h.need(h.sha(runtime)==RUNTIME_SHA,'verified359 route module SHA')
    for text in ("if n == 1 and qrows >= 1024 and runtime.q32_enabled and not capture:",
                 "profile, bm, bq, warps = 'q32', 256, 32, 16",
                 'selected.update(BLOCK_M=bm, BLOCK_Q=bq, TILE_SIZE=64, num_warps=warps, num_stages=1)',
                 'target_grid = (qrows // bq + n, 2)',"kwargs.get('num_query_heads') == 16", "kwargs.get('num_queries_per_kv') == 8"):
        h.need(text in runtime,'verified359 static route assertion')
    fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==c['name'])
    function=''.join(source.splitlines(True)[fn.lineno-1:fn.end_lineno]);text=files['kernel_cjj_konly_q32.ttgir'].decode()
    path=re.search(r'#loc = loc\("([^\"]+)":34:0\)',text)[1]
    h.need(path.endswith('/vllm_fl/patches/biv150_konly_encoded_q32.py'),'actual service debug/source path')
    meta=dict(schema=h.SCHEMA,enabled=True,profile='q32',format=h.FORMAT,target=c['target'],
        launch_options=dict(num_warps=c['num_warps'],num_stages=c['num_stages']),
        constants=dict(BLOCK_M=256,BLOCK_Q=32,TILE_SIZE=64,num_seqs=e['sequences'],num_query_heads=16,num_queries_per_kv=8,KV_QUANT_MODE=0,IS_3D=False),
        query_rows=e['query_rows'],grid=[e['query_rows']//32+1,2],kernel_hash=c['hash'],function_name=c['name'],
        source_digest=h.sha(function),ttgir_sha256=h.sha(text),pins=h.PINS)
    override,proof=h.propose(text,meta)
    provenance=dict(runtime_selected_hash_bound_cache=True,live_compiled_object_introspection=False,
        source_commit='359c3e11c9cf7551fbdda649bfb5a331b4822157',module_file_sha256=SOURCE_SHA,
        source_digest_kind='static function text from verified359 module, not live JIT object',
        implementation_sha256=e['implementation_sha256'],actual_source_path=path,
        static_scope_assertions='verified359 facade Q32 route plus actual TTGIR dimensions/module and cache options',
        missing_live_fields=['src.signature','src.constants','src.attrs'],
        reconstruction_requires_fresh_unhoisted_TTGIR_and_LLVM_exact_before_override=True)
    return files,meta,override,dict(proposal=proof,provenance=provenance),source
