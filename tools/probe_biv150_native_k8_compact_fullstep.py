#!/usr/bin/env python3
"""CPU default; fixed compact12544 full step, reusing the frozen 3eb ABBA body."""
import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
PARENT = '3ebdbebf61e0e90735cd8ad83eb1605627b6ef46'
WRITER_COMMIT = 'dc5fd2e5b86a3d57aceaad26e8556d169ec7462a'
LIBRARY = '40f9b17c634d2344ca54b9f1202e4e10932bf3f3f632cd41d793ddafd0934453'
CONSUMER = '9d2bcd2f803626fca16aae5c73533ce1436d8bb98010f00fa6583061dfa90df8'
REPLAY_FILE = 'docs/evidence/native-k8-compact-device-v1-20261028/consumer-root-replay.py'
REPLAY_SHA = 'a1e30ce992c911b0547f80537f56b1f71415d39b7c0ba91329f6a3d66a802c86'
FORMAT = 'cjj-konly-compact12544-rn32-rne-v1'
PAGE = 12544
FROZEN = {
    'writer': ('vllm_fl/patches/biv150_konly_compact_writer.py', 'a0cacf3f13f450f1f633d0bf562b3b67239cdc4b4e2a9b1159bcbde4b3fa290f'),
    'views': ('vllm_fl/patches/biv150_konly_compact_views.py', '1ab738e4b8178ee0d8ce12536ac7e6428ed09f7252ea7349c8d71f73bae64b1e'),
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(commit, name):
    return subprocess.check_output(['git', '-C', str(ROOT), 'show', commit+':'+name])


def prepare(parent):
    """Load the actual frozen reference tree; never relax its old BF16 pins."""
    parent = Path(parent).resolve()
    assert parent.is_dir() and parent != ROOT
    files = ['tools/probe_biv150_native_k8_fullstep.py', 'tools/biv150_native_k8_fullstep_fixture.py',
             'tools/biv150_native_k8_consumer_receipt.py', 'tools/biv150_native_k8_source_factory.py',
             'tools/biv150_native_tl_pair_contract.py', 'tools/biv150_konly_strided_writer_contract.py',
             'tools/biv150_native_k8_consumer_fixture.py', 'tools/biv150_native_k8_consumer_v2_fixture.py',
             'tools/biv150_native_k8_codec_fixture.py', 'tools/biv150_native_k8_codec_receipt.py',
             'tools/biv150_native_k8_pagewise_reference.py']
    for name in files:
        assert (parent/name).read_bytes() == blob(PARENT, name), name
        module_name = Path(name).stem
        if module_name in sys.modules:
            assert Path(sys.modules[module_name].__file__).resolve() == parent/name
    sys.path.insert(0, str(parent/'tools'))
    old = importlib.import_module('probe_biv150_native_k8_fullstep')
    old.f.pins()
    return old


def compact_chunk(canonical):
    """Address remap only; original scalar codec/oracle remains unchanged."""
    import numpy as np
    assert canonical.dtype == np.uint8 and canonical.ndim == 2 and canonical.shape[1] == 16384
    result = np.full((len(canonical), PAGE), 0xa5, dtype=np.uint8)
    result[:, :4224] = canonical[:, :4224]
    result[:, 4352:] = canonical[:, 8192:]
    return result


def compact_contract(original):
    # Same View classes, validation and slot checks; only actual cache geometry differs.
    ns = dict(vars(original), PAGE=PAGE, FORMAT=FORMAT)
    qualify = types.FunctionType(original.qualify.__code__, ns, original.qualify.__name__)
    return types.SimpleNamespace(View=original.View, SlotSnapshot=original.SlotSnapshot, qualify=qualify)


def materialize(out, role):
    name, expected = FROZEN[role]
    raw = blob(WRITER_COMMIT, name)
    assert sha(raw) == expected
    folder = out/'frozen-sources'; folder.mkdir(exist_ok=True)
    path = folder/Path(name).name
    assert not path.exists()
    path.write_bytes(raw)
    return path


def validate_roles(views, arenas, cases, pages, torch):
    def observe(t):
        s = t.untyped_storage()
        return views.View(str(t.dtype).removeprefix('torch.'), tuple(t.shape), tuple(t.stride()),
            int(s._cdata), s.data_ptr(), s.nbytes(), t.storage_offset(), t.data_ptr(), str(t.device))
    raw = arenas['k8']['raw']
    layer = raw.view(torch.bfloat16).as_strided((pages,2,16,2,128),(6272,2176,256,128,1))
    expected = views.parameters(observe(layer), 'decode79', format_id=FORMAT)
    for field, role in [('key','key_cache_ptr'), ('scale','key_scale_ptr'), ('value','value_cache_ptr')]:
        assert observe(cases['k8'][field]) == expected[role]
    expected = views.parameters(observe(layer), 'writer83', format_id=FORMAT)
    for role, t in [('code_full',raw.view(torch.int8)), ('scale_full',raw.view(torch.float32)),
                    ('value_full',raw.view(torch.bfloat16))]:
        assert observe(t) == expected[role]
    return dict(format=FORMAT, page_bytes=PAGE, pages=pages, V_consumer_offset=4352,
                writer_three_full_bases=True, frozen_views_sha256=FROZEN['views'][1])


def initial_sources(torch, source, pages, guard, rec, timed):
    # The old cold source interleaves K/V by page; the existing writer takes a
    # constant token stride. Pack two guarded linear sources outside step timing.
    buffers = {}; values = {}
    for name, index in [('key',0), ('value',1)]:
        full = torch.full((pages*8192+2*guard,),109,dtype=torch.uint8,device='cuda')
        value = full.narrow(0,guard,pages*8192).view(torch.bfloat16).reshape(pages*16,2,128)
        rec['initial']['compact_initial_'+name+'_linear_pack'] = timed(
            lambda: value.reshape(pages,16,2,128).copy_(source[:,index]), 1)
        buffers[name] = full; values[name] = value
    full = torch.full((pages*16*8+2*guard,),109,dtype=torch.uint8,device='cuda')
    value = full.narrow(0,guard,pages*16*8).view(torch.int64)
    rec['initial']['compact_initial_slots'] = timed(
        lambda: value.copy_(torch.arange(pages*16,dtype=torch.int64,device='cuda')),1)
    buffers['slots'] = full; values['slots'] = value
    return buffers, values


def check_initial_sources(buffers, cold, raw, pages, guard, rec, stage):
    import numpy as np
    rows = []
    for name, full in buffers.items():
        size = pages*(16*8 if name == 'slots' else 8192)
        assert raw(full.narrow(0,0,guard))+raw(full.narrow(0,guard+size,guard)) == bytes([109])*(2*guard)
        chunks = []
        for start in range(0,pages,128):
            count = min(128,pages-start)
            expected = (np.arange(start*16,(start+count)*16,dtype='<i8').tobytes() if name == 'slots'
                        else cold[start:start+count,0 if name == 'key' else 1].tobytes())
            width = 128 if name == 'slots' else 8192
            got = raw(full.narrow(0,guard+start*width,count*width))
            assert got == expected, 'complete compact initial source readonly'
            chunks.append(dict(start=start,bytes=len(got),sha256=sha(got)))
        rows.append(dict(source=name,all_bytes_compared=size,guards_exact=True,chunks=chunks))
    rec.setdefault('compact_initial_source_checks',[]).append(dict(stage=stage,sources=rows))


def adapted_source():
    text = blob(PARENT, 'tools/probe_biv150_native_k8_fullstep.py').decode()
    node = next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name == 'run')
    original = ast.get_source_segment(text,node)
    edits = [
        ('f.sha(a.k8_library.read_bytes())==f.LIBRARY_SHA', 'f.sha(a.k8_library.read_bytes())==COMPACT_LIBRARY'),
        ('result=dict(f.plan(),actual_consumer=pre,', 'result=dict(compact_plan(),actual_consumer=pre,'),
        ("writer=module(f.ROOT/'tools/kernels/biv150_konly_strided_writer.py','cjj_fullstep_exact83_writer')",
         "writer=module(materialize(out,'writer'),'cjj_fullstep_compact91_writer')\n        views=module(materialize(out,'views'),'cjj_fullstep_compact91_views')"),
        ("contract=module(f.ROOT/'tools/biv150_konly_strided_writer_contract.py','cjj_fullstep_83_contract')",
         "contract=compact_contract(module(f.ROOT/'tools/biv150_konly_strided_writer_contract.py','cjj_fullstep_83_contract'))"),
        ("full=torch.full((f.PAGES*16384+2*f.GUARD,),90,dtype=torch.uint8,device='cuda');resident=full.narrow(0,f.GUARD,f.PAGES*16384)",
         "page_bytes=16384 if arm=='bf16' else 12544\n                full=torch.full((f.PAGES*page_bytes+2*f.GUARD,),90,dtype=torch.uint8,device='cuda');resident=full.narrow(0,f.GUARD,f.PAGES*page_bytes)"),
        ("key=resident.view(torch.int8).as_strided((f.PAGES,16,2,128),(16384,256,128,1));scale=resident.view(torch.float32)",
         "key=resident.view(torch.int8).as_strided((f.PAGES,16,2,128),(12544,256,128,1));scale=resident.view(torch.float32)"),
        ("value=full.view(torch.bfloat16).as_strided((f.PAGES,16,2,128),(8192,256,128,1),storage_offset=(f.GUARD+8192)//2)",
         "value=full.view(torch.bfloat16).as_strided((f.PAGES,16,2,128),(6272,256,128,1),storage_offset=(f.GUARD+4352)//2)"),
        ("and value.data_ptr()==resident.data_ptr()+8192", "and value.data_ptr()==resident.data_ptr()+4352"),
        ('initial_and_consumer_V_is_offset_bytes=', 'consumer_V_is_offset_bytes='),
        ("            def reset(arm):", "            initial_backs,initial_values=initial_sources(torch,gpu_source,f.PAGES,f.GUARD,rec,timed)\n            rec['compact_roles']=validate_roles(views,arenas,cases,f.PAGES,torch)\n            def reset(arm):"),
        ("support.encode_initial[(f.PAGES*32,)](gpu_source[:,0],gpu_source[:,1],cases[arm]['key'],cases[arm]['scale'],cases[arm]['value'],f.PAGES,num_warps=1,num_stages=1)",
         "writer.encode_append[(f.PAGES*16,2)](initial_values['key'],initial_values['value'],initial_values['slots'],arenas[arm]['raw'].view(torch.int8),cases[arm]['scale'],arenas[arm]['raw'].view(torch.bfloat16),256,256,f.PAGES*16,f.PAGES*16,f.PAGES*16,f.PAGES,num_warps=1,num_stages=1)"),
        ("support.encode_initial.warmup(gpu_source[:,0],gpu_source[:,1],cases['k8']['key'],cases['k8']['scale'],cases['k8']['value'],f.PAGES,grid=(f.PAGES*32,),num_warps=1,num_stages=1)",
         "writer.encode_append.warmup(initial_values['key'],initial_values['value'],initial_values['slots'],arenas['k8']['raw'].view(torch.int8),cases['k8']['scale'],arenas['k8']['raw'].view(torch.bfloat16),256,256,f.PAGES*16,f.PAGES*16,f.PAGES*16,f.PAGES,grid=(f.PAGES*16,2),num_warps=1,num_stages=1)"),
        ("existing_initial_encoder_JIT_load_wall_seconds", "existing_compact91_writer_initial_JIT_load_wall_seconds"),
        ("                z=controls[p];return contract.qualify(", "                validate_roles(views,arenas,cases,f.PAGES,torch)\n                z=controls[p];return contract.qualify("),
        ("writer.encode_append[(n,2)](nk,nv,slots,z['key'],z['scale'],", "writer.encode_append[(n,2)](nk,nv,slots,arenas[arm]['raw'].view(torch.int8),z['scale'],"),
        ('torch.ops.cjj_native_k8_qk_probe.produce', 'torch.ops.cjj_native_k8_compact_probe.produce'),
        ("z=arenas[arm];guard=raw(z['full'].narrow(0,0,f.GUARD))+raw(z['full'].narrow(0,f.GUARD+f.PAGES*16384,f.GUARD));assert guard==bytes([90])*(2*f.GUARD)",
         "z=arenas[arm];page_bytes=16384 if arm=='bf16' else 12544\n                guard=raw(z['full'].narrow(0,0,f.GUARD))+raw(z['full'].narrow(0,f.GUARD+f.PAGES*page_bytes,f.GUARD));assert guard==bytes([90])*(2*f.GUARD)"),
        (";actual=raw(z['raw'].narrow(0,start*16384,count*16384))", ";expected=compact_chunk(expected) if arm=='k8' else expected;actual=raw(z['raw'].narrow(0,start*page_bytes,count*page_bytes))"),
        ("all_bytes_compared=f.PAGES*16384,guards_exact=True,chunks=chunks", "all_bytes_compared=f.PAGES*page_bytes,guards_exact=True,chunks=chunks"),
        ("            def check_source(stage):", "            def check_source(stage):\n                check_initial_sources(initial_backs,cold,raw,f.PAGES,f.GUARD,rec,stage)"),
        ("for role in ('k8_append','bf16_append','reducer'):", "for role in ('initial_encoder','k8_append','bf16_append','reducer'):"),
        (";torch.cuda.empty_cache()", ";del initial_backs,initial_values;torch.cuda.empty_cache()"),
    ]
    changed = original
    for old,new in edits:
        assert changed.count(old) == 1, old
        changed = changed.replace(old,new)
    restored = changed
    for old,new in reversed(edits):
        assert restored.count(new) == 1, new
        restored = restored.replace(new,old)
    assert restored == original
    ast.parse(changed)
    return changed, dict(parent_commit=PARENT,parent_function_sha256=sha(original.encode()),
        adapted_function_sha256=sha(changed.encode()),inverse_byte_exact=True,
        edits=[dict(old=a,new=b) for a,b in edits])


def replay_consumer(a):
    """Reuse the already reviewed complete saved-array replay; paths only differ."""
    path = ROOT/REPLAY_FILE; text = path.read_text(); assert sha(path.read_bytes()) == REPLAY_SHA
    assert sha((a.compact_consumer_raw/'out/consumer-summary.json').read_bytes()) == CONSUMER
    prefix, tail = text.split("path=Path('/private/tmp/cjj-compact100-consumer-actual-review-20261028.json')",1)
    substitutions = {
        '/private/tmp/flagos-cjj-native-k8-compact-consumer-frozen-51d-20261028': str(ROOT),
        '/private/tmp/cjj-native-k8-compact-consumer-v1-raw-20261028': str(a.compact_consumer_raw),
        '/private/tmp/cjj-native-k8-compact-compile-v1-raw-20261028/out': str(a.compile_summary.parent),
        '/private/tmp/cjj-native-k8-codec-v1-raw-20261028/out/codec-summary.json': str(a.codec_summary),
        '/private/tmp/cjj-native-k8-compile-v1-raw-20261028/out/build/cjj_native_k8_qk_probe.so': str(a.original_library),
    }
    for old,new in substitutions.items():
        assert prefix.count(repr(old)) == 1
        prefix = prefix.replace(repr(old),repr(new))
    # The replay's library lookup is fixed to the actual compile archive. Require
    # the runner to load those same bytes, not an unrelated passed summary.
    assert a.compile_summary.name == 'compile-summary.json'
    assert sha(a.k8_library.read_bytes()) == LIBRARY
    ns = dict(__file__=str(path),__name__='cjj_compact_fullstep_saved_array_replay')
    exec(compile(prefix,str(path)+'<paths-only>','exec'),ns)
    assert ns['admission']['library_sha256'] == LIBRARY
    assert not {'torch','triton'}.intersection(sys.modules)
    return ns['report']


def execute(a):
    assert a.root_owned_exclusive
    assert all(p.is_absolute() for k,p in vars(a).items() if isinstance(p,Path))
    assert not a.out.exists()
    old = prepare(a.parent_source)
    actual = replay_consumer(a)
    source, transformation = adapted_source()
    def compact_plan():
        plan = old.f.plan()
        plan.update(component='compact12544 complete writer + native40f9 + original reducer vs aace BF16; not service',
            K8_library_sha256=LIBRARY, namespace='cjj_native_k8_compact_probe',format=FORMAT,
            page_bytes={'bf16':16384,'k8':PAGE},arena_bytes={'bf16':old.f.PAGES*16384,'k8':old.f.PAGES*PAGE},
            K8_append='unchanged dc5 compact writer with actual separate pitches/row extents/slot_count',
            initial_K8_encode='same dc5 compact writer on guarded linear K/V and full-base V',
            actual_compact_consumer=actual,compact_fullstep_adapter=transformation,
            compact_writer_sources=FROZEN,compact_writer_source_commit=WRITER_COMMIT,
            compact_initial_linear_pack_costs_separate=True,capacity_reclaimed=False,
            service_source_pitch2560_covered=False)
        return plan
    ns = dict(vars(old), COMPACT_LIBRARY=LIBRARY,compact_plan=compact_plan,
        materialize=materialize,compact_contract=compact_contract,compact_chunk=compact_chunk,
        initial_sources=initial_sources,check_initial_sources=check_initial_sources,validate_roles=validate_roles)
    exec(compile(source,'<compact-address-writer-adapted-original-fullstep>','exec'),ns)
    return ns['run'](a)


def plan():
    _, transform = adapted_source()
    for name,digest in FROZEN.values(): assert sha(blob(WRITER_COMMIT,name)) == digest
    return dict(parent_commit=PARENT,writer_commit=WRITER_COMMIT,library_sha256=LIBRARY,
        consumer_summary_sha256=CONSUMER,fixed_pages=34276,compact_page_bytes=PAGE,
        cases={'4k':[64,4096,2,2048],'16k':[32,16384,4,2560]},new_KV_pitch=256,
        service_source_pitch2560_covered=False,transform=transform,frozen_sources=FROZEN,
        exact_original_FP64_Graph_side_ABBA_timing_preserved=True,
        frozen_parent_source_required=True,complete_saved_consumer_replay_before_Torch=True,
        CPU_only=True,GPU_executed=False,device_numeric_verified=False,timing_executed=False,
        capacity_reclaimed=False,model_verified=False,precision_gate=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',action='store_true');p.add_argument('--root-owned-exclusive',action='store_true')
    paths = ('parent-source','compact-consumer-raw','compile-summary','codec-summary','original-library',
             'consumer-summary','k8-library','bf16-library','out')
    for name in paths:p.add_argument('--'+name,type=Path)
    p.add_argument('--consumer-proof-sha256');p.add_argument('--case',choices=('4k','16k'),action='append')
    a=p.parse_args()
    if not a.gpu:
        if any(v for k,v in vars(a).items() if k!='gpu'):p.error('Device arguments require --gpu')
        print(json.dumps(plan(),sort_keys=True,indent=2));return 0
    if not a.root_owned_exclusive or not all(getattr(a,k.replace('-','_')) for k in paths) or not a.consumer_proof_sha256:
        p.error('Root ownership, frozen reference, actual compact raw+compile+codec, original consumer and all three exact libraries required')
    if a.case and len(a.case)!=len(set(a.case)):p.error('Duplicate case')
    return execute(a)


if __name__=='__main__':raise SystemExit(main())
