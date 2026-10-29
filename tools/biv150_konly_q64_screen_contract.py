"""Narrow CPU admission for one Q32/Q64 screen; original math is imported by SHA."""
import ast
import json
import re
from pathlib import Path
import sys
import types
import biv150_konly_q64_compile_source as c

REFERENCE_COMMIT = 'e4dd64e453fbe1bdeba4d8b2a1b551167e9446c4'
REFERENCES = {
    'biv150_konly_i8_qk_reference.py':'63d8f901b3648dad52f9dceab251e1d451def2aaa96f2bc30bd0aaff102e64e3',
    'biv150_konly_decode_quant_reference.py':'38807ab17bf28e64afee6fb581af92d4e715856d954dc42db4fdd1dd6d91bc18',
    'biv150_konly_q32_numeric_contract.py':'9c75e7a81c0d77fba0690d25db355cb4dfd669b9aed963930697b7951e6c46e8',
    'bench_biv150_konly_service_q32_bitwise.py':'985f07044f35a9d0496d642c0fe527912354596876be33e1212bc18650a08c7f',
}
MANIFEST = '9711f65a2d260478292e5a1702f4de1be009b7930267f8d7d010b574c488d926'
SUMMARY = '4af012637630257c1310d68f3d0e648a2d6a14197d637b4260ff2ba534e608cb'
REVIEW = '256c95d945b44bfdd3dc0acd1f6a83da791f1a72ab8b0ba3d5fc6f052606caa0'
COMPILE_COMMIT = '844a69f10539c551500e29e2304061aef7d8671b'
EXECUTION_SOURCES = {
    'spec/iluvatar/triton/compiler/compiler.py':'a97f067023873acf2e348c28f61494e59170ca84c3ee2314f20e491d3e833dfd',
    'backends/iluvatar/compiler.py':'a0d2bbda2f779b528147a238ff6fc558d026478f72d3d12d51d27f48f1aa570a',
    'runtime/cache.py':'faf84021196925228e20fdab44b3372c5ac40a57f653c212c0655e8fb97757d0',
}
LOADED_COMPILER = Path('/usr/local/lib/python3.12/site-packages/triton/spec/iluvatar/triton/compiler/compiler.py')
FIXTURE = '289afc781f17e80465626eef045cd4bfb3db5ed7c516ce662d8e75c06a698132'
WRITER = '16afc46585f950f4b9c06c831b0e0dbf5515ee941a476a9ed300c492ebf947f9'
ROLES = ('q32','q64')
MODES = ('reader','append-reader')
# runtime and ELF scalar-spill fields deliberately remain separate.
RESOURCES = {'q32':dict(n_regs=122,n_spills=0,shared=33792,ELF_sgpr_spills=96),
             'q64':dict(n_regs=237,n_spills=0,shared=67584,ELF_sgpr_spills=485)}
ROWS_BEFORE = 'rows=[0,1,15,16,31,32,1024,2047];length=6144'
ROWS_AFTER = 'rows=[0,1,15,16,31,32,63,64,1024,2047];length=6144'


def plan():
    return dict(schema='cjj-q64-one-component-screen-v1',query_rows=2048,prefix=4096,total_KV=6144,pages=773,
        query_pitch=2048,writer_source_pitch=2560,table_pitch=8192,page_bytes=16384,patterns=[0,1],
        roles=list(ROLES),modes=list(MODES),resources=RESOURCES,
        compile_source=COMPILE_COMMIT,review_sha256=REVIEW,reference_commit=REFERENCE_COMMIT,reference_pins=REFERENCES,execution_source_pins=EXECUTION_SOURCES,
        numeric_rows_pattern1=[0,1,15,16,31,32,63,64,1024,2047],pattern0_all_rows=True,
        Graph_B_before_fresh_eager_B=True,Graph_reader_is_actual_hoisted_handle=True,
        ABBA_order=['q32','q64','q64','q32'],captured_steps=4,timed_graph_replays=16,warmup_graph_replays=8,
        GPU_executed=False,numeric_verified=False,Graph_verified=False,side_stream_verified=False,
        performance_measured=False,precision_gate=False,service_verified=False,official_score=False)


def reference_sources(folder):
    result={}
    for name,digest in REFERENCES.items():
        data=(folder/name).read_bytes();c.need(c.sha(data)==digest,'original reference '+name);result[name]=data.decode()
    text=result['biv150_konly_q32_numeric_contract.py']
    c.need(text.count(ROWS_BEFORE)==1 and ROWS_AFTER not in text,'reference row-list anchor')
    expanded=text.replace(ROWS_BEFORE,ROWS_AFTER)
    c.need(expanded.replace(ROWS_AFTER,ROWS_BEFORE)==text,'reference exact row-list inverse')
    return result,expanded


def load_reference(folder):
    sources,expanded=reference_sources(folder)
    for name in ('biv150_konly_i8_qk_reference','biv150_konly_decode_quant_reference'):
        c.need(name not in sys.modules,'unrelated reference module already loaded')
        c.load_file(name,folder/(name+'.py'))
    reference=c.module_from_source('cjj_q64_original_reference_plus_two_rows',expanded)
    tree=ast.parse(sources['bench_biv150_konly_service_q32_bitwise.py'])
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='policy')
    ns={'h':types.SimpleNamespace(need=c.need)}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'e4dd-original-policy','exec'),ns)
    return reference,ns['policy'],expanded


def preflight(raw,review_path,reference_root):
    c.need(c.sha((raw/'download-manifest.json').read_bytes())==MANIFEST,'actual compile manifest')
    manifest=json.loads((raw/'download-manifest.json').read_text())
    c.need(manifest['raw_files']==len(manifest['files'])==80 and manifest['raw_bytes']==9410399,'compile file geometry')
    for row in manifest['files']:
        p=raw/row['relative'];c.need(p.resolve().is_relative_to(raw.resolve()),'raw member escapes root')
        data=p.read_bytes();c.need(len(data)==row['bytes'] and c.sha(data)==row['sha256'],'compile member '+row['relative'])
    c.need(c.sha((raw/'out/summary.json').read_bytes())==SUMMARY,'actual summary')
    c.need(c.sha(review_path.read_bytes())==REVIEW,'actual independently replayed review')
    summary=json.loads((raw/'out/summary.json').read_text());review=json.loads(review_path.read_text())
    c.need(summary['completed'] is True and summary['compile_passed'] is True and summary['errors']==[],'compile completion')
    c.need(summary['commit']==review['source_commit']==COMPILE_COMMIT and review['completed'] is True and review['errors']==[],'review source')
    c.need(review['summary_sha256']==SUMMARY and review['download_manifest_sha256']==MANIFEST,'review raw binding')
    c.need(review['scope']['actual_backend_QCode_hoist_and_bank_reuse_evidence'] is True,'actual backend review')
    c.need(c.sha((c.ROOT/'tools/biv150_prefill_probe_fixtures.py').read_bytes())==FIXTURE,'original fixture')
    c.need(c.sha((c.ROOT/'vllm_fl/patches/biv150_konly_encoded_writer.py').read_bytes())==WRITER,'unchanged accepted writer')
    _,expanded=reference_sources(reference_root)
    owner=json.loads(next((raw/'exports').glob('*owner.json')).read_text())
    c.need(owner['source_commit']==COMPILE_COMMIT and owner['pid']==82478 and owner['start_ticks']==358039812,'actual compile owner')
    original_source=Path(owner['command'][1]).parents[1];original_output=Path(owner['out'])
    runtime={};records={}
    for role in ROLES:
        record=json.loads((raw/'out'/role/'hoisted/record.json').read_text());records[role]=record
        c.need(record==summary['records'][role]['hoisted'],'compile record mirror')
        g=review['groups'][role+'/hoisted'];r=RESOURCES[role]
        c.need(g['artifacts']==record['artifacts'],'review artifact binding')
        c.need(g['ELF']['.vgpr_count']==r['n_regs'] and g['ELF']['.sgpr_spill_count']==r['ELF_sgpr_spills'],'review register binding')
        c.need(g['ELF']['.vgpr_spill_count']==0 and g['ELF']['.private_segment_fixed_size']==0 and g['ELF']['.group_segment_fixed_size']==0,'ELF resources')
        c.need(all(g['runtime_handle'][k]==r[k] for k in ('n_regs','n_spills','shared')),'runtime resources')
        runtime[role]=dict(source=str(original_source/'vllm_fl/patches/biv150_konly_encoded_q32.py') if role=='q32' else str(original_output/'sources/candidate_q64.py'),
            override=str(original_output/role/'override.ttgir'))
    return dict(plan=plan(),raw_manifest_sha256=MANIFEST,compile_summary_sha256=SUMMARY,actual_review_sha256=REVIEW,
        expanded_reference_sha256=c.sha(expanded),records=records,runtime_paths=runtime)


def execution_sources(root):
    # Actual backend: ASM is a separately saved translation of LLIR; the
    # stage loop does not feed it onward. CompiledKernel loads binary_ext=cubin.
    for relative,digest in EXECUTION_SOURCES.items():
        c.need(c.sha((root/relative).read_bytes())==digest,'compiler/loader source changed '+relative)
    return dict(pins=EXECUTION_SOURCES,ASM_is_separate_translation=True,loaded_binary_extension='cubin',
        spec_stage_skip_lines=[392,393],backend_asm_lines=[460,466],backend_cubin_lines=[468,475],
        binary_selection_lines=[474,480],load_binary_argument_lines=[517,518],
        cache_hit_lines=[301,312],cache_miss_add_stages_line=324,cache_group_absolute_paths_lines=[95,112])


def execution_snapshot(snapshot):
    c.need(type(snapshot) is dict and type(snapshot.get('artifacts')) is dict,'snapshot/artifacts absent')
    artifacts=dict(snapshot['artifacts']);diagnostic=artifacts.pop('asm',None)
    c.need(type(diagnostic) is str and re.fullmatch('[0-9a-f]{64}',diagnostic) is not None,'actual diagnostic ASM SHA absent')
    return dict(snapshot,artifacts=artifacts),diagnostic


def check_loaded(role,kernel,snapshot,expected):
    # Only non-executed ASM text may differ. LLVM, full cubin, source, typed
    # ABI, kernel identity, other artifacts and runtime resources stay exact.
    actual,asm=execution_snapshot(snapshot);reviewed,old_asm=execution_snapshot(expected)
    c.need(actual==reviewed,'fresh executable identity differs for '+role)
    binary=getattr(kernel,'kernel',None)
    c.need(type(binary) is bytes and c.sha(binary)==expected['artifacts']['cubin'],'actual loader argument differs from final reviewed cubin')
    kernel._init_handles();r=RESOURCES[role]
    c.need(kernel.kernel==binary,'loader changed bound final cubin')
    observed=dict(n_regs=getattr(kernel,'n_regs',None),n_spills=getattr(kernel,'n_spills',None),shared=getattr(kernel.metadata,'shared',None))
    c.need(all(type(observed[k]) is int and observed[k]==r[k] for k in observed),'actual loaded resource mismatch '+role)
    c.need((kernel.metadata.num_warps,kernel.metadata.num_stages,kernel.metadata.warp_size)==(16,1,64),'loaded geometry')
    return dict(observed,diagnostic_ASM=dict(actual_sha256=asm,reviewed_sha256=old_asm,bytes_identical=asm==old_asm,used_as_execution_input=False),loader_cubin_sha256=c.sha(binary),ELF_sgpr_spills=r['ELF_sgpr_spills'],ELF_vgpr_spills=0,ELF_private_bytes=0,
        ELF_static_shared_bytes=0,threads=1024,resource_binding='exact reviewed final cubin plus actual runtime handle',occupancy_claimed=False)
