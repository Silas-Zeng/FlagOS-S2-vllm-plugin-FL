"""Private two-sequence coordinates and exact PR123 compiled-object admission."""
import ast
import json
from pathlib import Path
import re
import sys
import types
import biv150_konly_multi2_hoist_source as c
import biv150_konly_multi2_numeric_source as source

MAIN='899ce77470b0b539aebc108fd122f668c2d5894d'
COMPILE_COMMIT='f667e2c7150662460d60c6c870444f53e52eff6d'
MANIFEST='8b6af5b1aa715d449b95ae4826793afb8d6627de4d990eed3cba4cb2765b2391'
SUMMARY='048f02b18fa27b1f96f1705408e431ed1fe47512e1831c5b5893cf6c73bedd9d'
REVIEW='8737facf61e20be7afcb265e7df113059a5eebc18c4cb536014079668b9e1729'
ROLES=('unhoisted','hoisted')
MODES=('reader',)
RESOURCES={'unhoisted':dict(n_regs=135,n_spills=0,shared=57344,ELF_sgpr_spills=98),
           'hoisted':dict(n_regs=122,n_spills=0,shared=33792,ELF_sgpr_spills=96)}
# All phases keep the actual compiled total Q, cache allocation and descriptor
# strides. B/C swap a ragged request boundary and have partial final KV pages.
PHASES=(((1024,1024),(3072,3072)),((31,2017),(2079,4063)),((2017,31),(4063,2079)))
exec(compile(source.inherited_contract(),'executed-116-loader-checks','exec'),globals())
reviewed_cache=c.module_from_source('cjj_private_multi2_reviewed_cache',source.cache_source())
REFERENCE_EDITS=(
    ('def references(cpu,code,scale,torch,np):','def references(cpu,code,scale,torch,np,*,rows,length,prefix):'),
    ('    rows=[0,1,15,16,31,32,1024,2047];length=6144','    # Exact per-sequence coordinates supplied by the caller.'),
    ("ids=cpu['table'][0,:384].long()","ids=cpu['table'][0,:(length+15)//16].long()"),
    ('<4096+torch.tensor(rows)[:,None]+1','<prefix+torch.tensor(rows)[:,None]+1'),
)


def phase_geometry(qlens,klens):
    c.need(type(qlens) in (tuple,list) and type(klens) in (tuple,list) and len(qlens)==len(klens)==2,'exact two sequences')
    c.need(all(type(v) is int for v in (*qlens,*klens)),'integer non-bool geometry')
    c.need(sum(qlens)==2048 and all(0<q<=k for q,k in zip(qlens,klens)),'total Q2048 and nonempty valid causal ranges')
    c.need(2*sum((k+15)//16 for k in klens)+5==773,'original fixed 773-page fixture')
    return dict(qlens=list(qlens),klens=list(klens),prefixes=[k-q for q,k in zip(qlens,klens)],
        cu_q=[0,qlens[0],2048],query_tiles=sum((q+31)//32 for q in qlens),grid=[66,2])


def plan():
    return dict(schema='cjj-private-multi2-hoist-numeric-v1',main=MAIN,compile_source=COMPILE_COMMIT,
        dependency=source.DEPENDENCY,dependency_pins=source.PINS,compile_manifest_sha256=MANIFEST,
        compile_summary_sha256=SUMMARY,compile_review_sha256=REVIEW,resources=RESOURCES,
        roles=list(ROLES),modes=list(MODES),phases=[phase_geometry(*p) for p in PHASES],
        phase_labels=['A uniform balanced','B random ragged, Graph before fresh eager','C changed random reversed ragged on distinct side stream'],
        total_query_rows=2048,num_seqs=2,query_pitch=2048,output_pitch=2048,page_bytes=16384,pages=773,table_pitch=8192,
        reference_commit=REFERENCE_COMMIT,reference_pins=REFERENCES,original_thresholds_unchanged=True,
        pattern_A_reference_all_rows=True,other_patterns_original_per_sequence_row_selection=True,
        full_output_bitwise=True,whole_cache_and_readonly_bytes=True,Graph_B_before_fresh_eager_B=True,
        reviewed_cache_only=True,new_reader_backend_compile_prohibited=True,writer_launches=0,
        default_enabled=False,GPU_executed=False,numeric_verified=False,Graph_verified=False,
        side_stream_verified=False,precision_gate=False,performance_measured=False,service_verified=False,
        old_compile_Root_admission_granted=False,device_admission_requires_new_source_review=True,
        future_measurement=dict(order=['unhoisted','hoisted','hoisted','unhoisted'],complete_reader=True,
            captured_steps=4,warmup_graph_replays=8,timed_graph_replays=16,append_writer_included=False,
            includes=['Q RN32/RNE','Q/K/V loads and staging','QK','softmax','PV','output epilogue'],
            prerequisite='all original numeric, bitwise, Graph B and side C gates pass; separate Root-approved timing entry',
            no_performance_execution_in_this_frontend=True))


def reference_sources(folder):
    result={}
    for name,digest in REFERENCES.items():
        body=(folder/name).read_bytes();c.need(c.sha(body)==digest,'original numerical reference '+name);result[name]=body.decode()
    return result,c.transform(result['biv150_konly_q32_numeric_contract.py'],REFERENCE_EDITS)


def load_reference(folder):
    sources,expanded=reference_sources(folder)
    for name in ('biv150_konly_i8_qk_reference','biv150_konly_decode_quant_reference'):
        c.need(name not in sys.modules,'unrelated reference module already loaded');c.load_file(name,folder/(name+'.py'))
    reference=c.module_from_source('cjj_multi2_original_reference_with_sequence_coordinates',expanded)
    tree=ast.parse(sources['bench_biv150_konly_service_q32_bitwise.py'])
    policy=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='policy')
    ns={'h':types.SimpleNamespace(need=c.need)}
    exec(compile(ast.Module(body=[policy],type_ignores=[]),'e4dd-original-policy','exec'),ns)
    return reference,ns['policy'],expanded


def references(reference,fixture,cpu,code,scale,qlens,klens,pattern,torch,np):
    phase_geometry(qlens,klens);selected=[];original=[];quantized=[];start=0
    for seq,(qrows,length) in enumerate(zip(qlens,klens)):
        rows=fixture.row_indices(qrows,False)
        one=dict(cpu,q=cpu['q'][start:start+qrows],table=cpu['table'][seq:seq+1])
        refs=reference.references(one,code,scale,torch,np,rows=rows,length=length,prefix=length-qrows)
        if pattern==0:
            # Same FP64 uniform-softmax shortcut as executed 116; only the
            # sequence, length and prefix coordinates are generalized.
            values=cpu['packed'][cpu['table'][seq,:(length+15)//16].long(),1].reshape(-1,2,128)[:length].double()
            visible=length-qrows+torch.arange(qrows)+1
            full=(values.cumsum(0)[visible-1]/visible[:,None,None]).repeat_interleave(8,dim=1)
            refs=dict(refs,rows=list(range(qrows)),original=full,quantized=full)
        selected.extend(start+r for r in refs['rows']);original.append(refs['original']);quantized.append(refs['quantized']);start+=qrows
    return dict(rows=selected,original=torch.cat(original),quantized=torch.cat(quantized),
        quantized_reference='original INT64 QK / FP64 scales and closed softmax / BF16 V, stitched per sequence',
        probability_rounding_modeled=False,online_softmax_rounding_modeled=False)


def preflight(raw,review_path,reference_root):
    c.plan();plan()  # original GPU/proposal/helper pins and CPU geometry first
    body=(raw/'download-manifest.json').read_bytes();c.need(c.sha(body)==MANIFEST,'actual PR123 download manifest')
    manifest=json.loads(body);c.need(manifest['raw_files']==len(manifest['files'])==44 and manifest['raw_bytes']==3818887,'actual compile archive geometry')
    for row in manifest['files']:
        p=raw/row['relative'];c.need(p.resolve().is_relative_to(raw.resolve()),'raw path outside root')
        body=p.read_bytes();c.need(len(body)==row['bytes'] and c.sha(body)==row['sha256'],'actual raw '+row['relative'])
    body=(raw/'out/summary.json').read_bytes();c.need(c.sha(body)==SUMMARY,'actual compile summary')
    summary=json.loads(body);c.need(c.sha(review_path.read_bytes())==REVIEW,'independently replayed actual compile review')
    review=json.loads(review_path.read_text())
    c.need(summary['completed'] is True and summary['compile_passed'] is True and summary['errors']==[],'original compile completed')
    c.need(summary['commit']==review['source_commit']==COMPILE_COMMIT and review['review_passed'] is True,'compile review source')
    c.need(review['summary_sha256']==SUMMARY and review['manifest_sha256']==MANIFEST and review['all_original_hashes_verified'] is True,'actual review bindings')
    c.need(review['complete_ELF_ABI_equal'] is True and review['complete_typed_ABI_equal'] is True,'actual ABI equality')
    c.need(review['actual_unmodified_native_auditor_result']['actual_I8_operand_hoist_verified'] is True,'actual multi2 operand hoist review')
    c.need(review['Root_admission_granted'] is False and review['numeric_verified'] is False and review['service_verified'] is False,'compile-only qualification boundaries')
    c.need(summary['attention_launches']==summary['writer_launches']==summary['encoder_launches']==0,'original compile-only counters')
    c.need(c.sha((c.ROOT/'tools/biv150_prefill_probe_fixtures.py').read_bytes())==FIXTURE,'same accepted fixture')
    _,expanded=reference_sources(reference_root)
    owner=json.loads(next((raw/'exports').glob('*owner.json')).read_text())
    c.need(owner['source_commit']==COMPILE_COMMIT and owner['pid']==179282 and owner['start_ticks']==362567924,'actual original compile owner')
    output=Path(owner['out']);records={};runtime={}
    c.need((raw/'out/sources/candidate_multi2.py').read_text()==c.candidate_source(),'original actual GPU source still byte-identical')
    for role in ROLES:
        record=json.loads((raw/'out/multi2'/role/'record.json').read_text());records[role]=record
        c.need(record==summary['records']['multi2'][role],'original record mirror')
        actual=review['binary'][role];resource=RESOURCES[role];elf=actual['ELF_metadata']
        c.need(actual['cubin_sha256']==record['artifacts']['cubin'],'review final cubin')
        c.need(elf['.vgpr_count']==resource['n_regs'] and elf['.sgpr_spill_count']==resource['ELF_sgpr_spills'],'ELF register/spill fields')
        c.need(elf['.vgpr_spill_count']==elf['.private_segment_fixed_size']==elf['.group_segment_fixed_size']==0,'original ELF zero fields')
        c.need(all(actual['original_loaded_resources'][k]==summary['records']['multi2']['handle_resources'][role][k]==resource[k] for k in ('n_regs','n_spills','shared')),'actual loaded resources')
        runtime[role]=dict(source=str(output/'sources/candidate_multi2.py'),override=str(output/'multi2/override.ttgir') if role=='hoisted' else None)
    return dict(plan=plan(),raw_manifest_sha256=MANIFEST,compile_summary_sha256=SUMMARY,actual_review_sha256=REVIEW,
        original_compile_Root_admission_granted=False,expanded_reference_sha256=c.sha(expanded),records=records,runtime_paths=runtime)
