"""Verify completed multi2 raw records, recompute CPU summaries and deterministic archive."""
import argparse
import ast
import gzip
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
MAIN = '241220723a09061c182764f8f738e5ec01190367'
MODEL = 'e08c79456f9b943dbc4d9d02d0a4e030b4eceac1'
SDK = '54c495dcea404582a1526a96ae7e472965cbd03c8fa0931c71ef72223bf04e1f'
AUDITOR = 'cea62aeaea0915ec9a4b04c05879888014517ae70308130b07d995aee945b2dc'
RUNS = {'cjj-prefill-multi2-k64-v2-20261027': ('254b7a09e81b6efe85033ed01d0819c52c62f1e6', 4),
        'cjj-prefill-multi2-scope-v3-20261027': ('e936576a22ed7d55b0e230e01711735680f4603d', 18)}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def close(a, b):
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)


def git(commit, path):
    return subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+path])


def pure_proof(sdk_path):
    sdk = Path(sdk_path).read_bytes()
    need(sha(sdk) == SDK, 'SDK source identity differs')
    normalizer = git(MODEL,'vllm_fl/patches/biv150_prefill_normalized.py').decode()
    namespace = {'ast':ast,'re':re}
    nodes = [n for n in ast.parse(normalizer).body if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id=='_STRIDES' for t in n.targets)
             or isinstance(n,ast.FunctionDef) and n.name=='_specialize_strides']
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pure-stride-helper>','exec'),namespace)
    source = sdk.decode(); node = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='kernel_unified_attention')
    first = min([node.lineno]+[x.lineno for x in node.decorator_list])
    normalized = ''.join(source.splitlines(keepends=True)[first-1:node.end_lineno])
    normalized = normalized.replace('L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)','L = tl.zeros([BLOCK_M], dtype=tl.float32)')
    normalized = normalized.replace('def kernel_unified_attention(','def kernel_biv150_prefill_normalized(',1)
    normalized = namespace['_specialize_strides'](normalized)
    need(sha(normalized.encode())=='fa6ddc420a95102d116714fcbea41bc81aa9c09379d68518af65755c015b8744','Normalized source changed')
    lines = set()
    for text in (source,normalized):
        for index,line in enumerate(text.splitlines(),1):
            if 'query_ptr + query_offset' in line:
                lines.update((index-1,index,index+1))
    module = git(MODEL,'vllm_fl/patches/biv150_q_fragment_service.py').decode()
    nodes = [n for n in ast.parse(module).body if isinstance(n,ast.FunctionDef) and n.name=='lowered_proof']
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pure-lowered-proof>','exec'),namespace)
    return namespace['lowered_proof'], lines


def manager(proof):
    need(proof['active'] is True and proof['private_jit_cache'] is True,'Manager inactive/shared cache')
    need(proof['record_errors']==[] and proof['omitted_records']==0,'Incomplete Manager records')
    need(proof['fallback_calls']==proof['fallback_compiles']==proof['original_3d_calls']==0,'Manager fallback/3D')
    for record in proof['compile_records']:
        p = record['mechanism']
        need(record['mode']=='selected' and p['mechanism_pass'] is True and p['ttgir_q_load_outside_loop'] is True
             and p['llvm_q_memory_reads_in_loop']==0 and p['llvm_q_memory_reads_total']>0,'Q-hoist proof failed')
    return {row['candidate_hash']:row for row in proof['compile_records']}


def run(snapshot, tag, commit, count, lowered, q_lines):
    folder = snapshot/tag; data = read(folder/'result.json')
    owner = read(snapshot/'exports'/(tag+'-owner.json'))
    need(data['source_commit']==owner['source_commit']==commit,'Source/owner mismatch')
    script = git(commit,'tools/bench_biv150_prefill_multi2.py')
    need(sha(script)==data['script_sha256'],'Original measured script changed')
    need(data['completed'] is True and data['errors']==[] and len(data['cases'])==count,'Incomplete component')
    need(all(data[k] is False for k in ('official_score','precision_gate','service_accuracy_validated','service_installed')),'Component mislabeled as service acceptance')
    need(data['baseline_mode']=='current-multi2' and data['baseline_config']==[128,8,64,1],'Wrong two-sequence baseline')
    need(data['production_transform_unchanged'] is True and data['adapted_manager_sha256'] is None,'Modified production transform')
    need(data['candidate_tile']==[256,32] and (data['candidate_warps'],data['candidate_kv_tile'],data['candidate_stages'])==(16,64,1),'Wrong candidate')
    need(all(v['installed'] is True for v in data['installations'].values()),'Original parent not installed')
    # These frozen sources exercise a non-default stream, poison output and capture once before changed inputs.
    source = script.decode()
    for marker in ('stream=torch.cuda.Stream()', "out.fill_(float('nan'))", 'for pattern in (0,1):',
                   'if label not in graphs:', 'graphs[label].replay()', 'with torch.cuda.stream(stream):'):
        need(marker in source,'Missing original side/changed-Graph protocol: '+marker)
    rows = []; exceptions = []
    for case in data['cases']:
        q, k = case['qlens'],case['klens']; total = sum(q); wide = total>=1024
        need(case['completed'] is True and len(q)==len(k)==2 and min(q)>=0 and all(b>=a for a,b in zip(q,k)),'Invalid shape/completion')
        need(case['case_contract']['total_query_rows']==total and case['case_contract']['num_sequences']==2,'Wrong shape contract')
        need([x['pattern'] for x in case['checks']]==[0,1],'Missing input pattern')
        need(case['checks'][0]['cpu_input_sha256']['q']!=case['checks'][1]['cpu_input_sha256']['q']
             and case['checks'][0]['cpu_input_sha256']['packed']!=case['checks'][1]['cpu_input_sha256']['packed'],'Graph inputs did not change')
        checks = []
        for check in case['checks']:
            direct, guard = check['direct'],check['reference_guard']
            need(direct['passed'] is True and direct['finite'] is True and direct['numeric_equal'] is True
                 and direct['max_abs_diff']==direct['mismatched_elements']==direct['outside_tolerance']==0,'Direct numerical mismatch')
            need(direct['elements']==total*16*128,'Incomplete direct output coverage')
            need(guard['passed'] is True and guard['finite'] is True and guard['baseline_exceptions_over_one_bf16_ulp']==0
                 and guard['new_or_worsened_violations']==0,'Reference guard failed')
            need([v['route'] for v in check['graphs']]==['baseline','candidate'],'Missing Graph route')
            need(all(v['passed'] is True and v['finite'] is True and v['numeric_equal'] is True
                     and v['atol']==v['rtol']==v['max_abs_diff']==0 for v in check['graphs']),'Changed Graph mismatch')
            cpu = {key:check[key] for key in ('baseline_cpu','candidate_cpu')}
            need(all(v['finite'] is True and v['atol']==v['rtol']==.005 for v in cpu.values()),'CPU oracle finite/tolerance changed')
            if guard['existing_baseline_violations']:
                exceptions.append(dict(case=case['name'],pattern=check['pattern'],baseline_cpu=cpu['baseline_cpu'],
                    candidate_cpu=cpu['candidate_cpu'],reference_guard=guard))
            checks.append(dict(pattern=check['pattern'],direct=direct,**cpu,reference_guard=guard,
                               graph_routes=check['graphs'],reference_rows='all uniform rows' if check['pattern']==0 else 'selected random rows'))
        for label, bm, bq, warps in [('baseline',128,16,8),('candidate',256 if wide else 128,32 if wide else 16,16 if wide else 8)]:
            selection = case[label+'_selection']; records = manager(case[label+'_proof'])
            need((selection['BLOCK_M'],selection['BLOCK_Q'],selection['num_warps'],selection['TILE_SIZE'],selection['num_stages'])==(bm,bq,warps,64,1),'Wrong selected geometry')
            need(selection['IS_3D'] is False and selection['num_seqs']==2 and selection['grid']==[total//bq+2,2],'Wrong generic two-sequence grid')
            need(selection['hash'] in records and selection['loaded_resources']['hash']==selection['hash'],'No actual compiled kernel link')
            parents=selection.get('preserved_default_parents',[])
            need(label=='candidate' and wide or any(p['name']=='q32' for p in parents),'Missing preserved Q32 parent evidence')
            for parent in parents:
                need(parent['active'] is True and parent['disabled'] is False and parent['selected_calls']==0,'Q32 parent bypassed/selected on two seq')
        if 'private_policy_selects_wide' in case:
            need(case['private_policy_selects_wide'] is wide and case['wide_manager_selected_calls_delta']==(2 if wide else 0),'Private policy scope mismatch')
            need(case['private_policy']['condition']=='num_sequences == 2 and total_query_rows >= 1024'
                 and case['private_policy']['host_reads_from_device'] is False and case['private_policy']['production_installed'] is False,'Scope reads or service mislabel')
        rounds = case['rounds']; need(len(rounds)==3,'Missing timing rounds')
        for index, order in enumerate((['baseline','candidate'],['candidate','baseline'],['baseline','candidate'])):
            timing = rounds[index]
            need(timing['index']==index and timing['order']==order,'Timing order changed')
            need(timing['baseline_ms']>0 and timing['candidate_ms']>0
                 and close(timing['speedup'],timing['baseline_ms']/timing['candidate_ms']),'Timing arithmetic differs')
        median = statistics.median(r['speedup'] for r in rounds)
        need(close(median,case['median_speedup']),'Stored median differs')
        rows.append(dict(name=case['name'],qlens=q,klens=k,wide=wide,rounds=rounds,median_speedup=median,
                         baseline_selection=case['baseline_selection'],candidate_selection=case['candidate_selection'],checks=checks))
    if count==18:
        policy=data['terminal_private_policy']
        need(policy['selected_calls']==32 and policy['fallback_calls']==4 and policy['host_reads_from_device'] is False
             and policy['production_installed'] is False,'Terminal private-policy scope count differs')
    compiled = {}
    for label,bm in (('baseline',128),('candidate',256)):
        terminal = data['terminal_manager_proofs'][label]; records = manager(terminal)
        actual = terminal['actual_loaded_resources']; record = records[actual['hash']]
        ir = folder/'compiler-ir'/label
        for suffix in ('ttgir','llir'):
            need(sha((ir/('actual.'+suffix)).read_bytes())==actual['asm_sha256'][suffix],'Saved actual IR mismatch')
        need(sha((ir/record['override_filename']).read_bytes())==record['override_sha256'],'Override IR mismatch')
        recomputed = lowered((ir/'actual.ttgir').read_text(),(ir/'actual.llir').read_text(),q_lines,bm)
        need(recomputed==record['mechanism'],'Saved actual Q-hoist mechanism differs')
        compiled[label] = dict(actual_loaded_resources=actual,recomputed_q_hoist=recomputed,
                               saved_files=['actual.ttgir','actual.llir',record['override_filename']],
                               asm_cubin_bytes_saved=False,asm_cubin_hashes_only=True)
    wide_speeds = [v['median_speedup'] for v in rows if v['wide']]
    fallback = [v['median_speedup'] for v in rows if not v['wide']]
    return dict(source_commit=commit,script_sha256=data['script_sha256'],owner=owner,completed=True,
                component_guards_passed=True,official_score=False,precision_gate=False,
                wide_cases=len(wide_speeds),fallback_cases=len(fallback),
                wide_speedup_range=[min(wide_speeds),max(wide_speeds)],fallback_speedups=fallback,
                cpu_oracle_inherited_exceptions=exceptions,compiled=compiled,cases=rows)


def archive(files):
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw,mode='w',format=tarfile.USTAR_FORMAT) as tar:
        for name,value in sorted(files.items()):
            info = tarfile.TarInfo(name); info.size=len(value); info.mode=0o644
            info.mtime=info.uid=info.gid=0; info.uname=info.gname=''
            tar.addfile(info,io.BytesIO(value))
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output,mode='wb',filename='',mtime=0,compresslevel=9) as stream:
        stream.write(raw.getvalue())
    return output.getvalue()


def main(args):
    manifest = read(args.snapshot/'download-manifest.json'); originals = {}
    for row in manifest['files']:
        name = row['relative']; need(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe raw path')
        value = (args.snapshot/name).read_bytes()
        need(name not in originals and len(value)==row['bytes'] and sha(value)==row['sha256'],'Raw download SHA/size mismatch')
        originals[name]=value
    need(len(originals)==18,'Incomplete original set')
    need({str(p.relative_to(args.snapshot)) for p in args.snapshot.rglob('*') if p.is_file()}==set(originals)|{'download-manifest.json'},'Uninventoried snapshot file')
    originals['download-manifest.json']=(args.snapshot/'download-manifest.json').read_bytes()
    lowered,q_lines=pure_proof(args.sdk_source)
    runs={tag:run(args.snapshot,tag,*values,lowered,q_lines) for tag,values in RUNS.items()}
    audit_path=ROOT/'tools/audit_biv150_multi2_pair.py'
    need(sha(audit_path.read_bytes())==AUDITOR,'Frozen final auditor changed')
    spec=importlib.util.spec_from_file_location('_frozen_multi2_audit',audit_path); audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    baseline=read(args.baseline_audit); arm=baseline['baseline']
    need(baseline['baseline_only'] is True and baseline['pair_complete'] is False and baseline['candidate_not_audited'] is True
         and baseline['model_rerun'] is False and baseline['gpu_work'] is False and baseline['audit_script_sha256']==AUDITOR
         and baseline['source_commit']==MAIN and baseline['baseline_all_gates_passed'] is True,'Baseline-only audit identity/status differs')
    scores=arm['accuracy']['question_scores']; need(len(scores)==105 and sum(scores.values())==arm['accuracy']['correct']==102,'Baseline105 computed scores differ')
    need(arm['accuracy']['passed'] is True and arm['state']['complete'] is True and arm['state']['cleanup_complete'] is True,'Baseline incomplete')
    for length in ('4096','16384'):
        need(audit.working_gate(arm['scores'][length],length)==arm['working_gates'][length] and arm['working_gates'][length]['passed'] is True,'Baseline working gate differs')
    packed=archive(originals); need(len(packed)<=2*1024*1024,'Archive exceeds publisher limit')
    need(packed==archive(originals),'Archive is not deterministic')
    with tarfile.open(fileobj=io.BytesIO(packed),mode='r:gz') as tar:
        need({m.name for m in tar.getmembers()}==set(originals),'Archive member set differs')
        for member in tar.getmembers():
            need(member.isfile() and tar.extractfile(member).read()==originals[member.name],'Archived bytes differ')
    args.out.mkdir(parents=False,exist_ok=False)
    (args.out/'completed-components-raw.tar.gz').write_bytes(packed)
    (args.out/'baseline-independent-v1.json').write_bytes(args.baseline_audit.read_bytes())
    inventory=dict(archive='completed-components-raw.tar.gz',archive_bytes=len(packed),archive_sha256=sha(packed),
        deterministic_gzip_mtime=0,deterministic_tar_metadata=True,original_files=18,archive_members=19,
        files=[dict(relative=name,bytes=len(value),sha256=sha(value)) for name,value in sorted(originals.items())],
        baseline_only_audit_sha256=sha(args.baseline_audit.read_bytes()))
    (args.out/'archive-manifest.json').write_text(json.dumps(inventory,indent=2)+'\n')
    summary=dict(snapshot_utc=manifest['utc'],cpu_only=True,gpu_rerun=False,original_files_sha_reverified=18,
        summary_script_sha256=sha(Path(__file__).read_bytes()),model_source_frozen=MODEL,precision_gate=False,
        pair_audited=False,service_gain_claimed=False,model_snapshot=manifest['model'],runs=runs,
        baseline_only=dict(export_sha256=sha(args.baseline_audit.read_bytes()),correct=102,count=105,
            gates_passed=True,pair_complete=False,candidate_not_audited=True,
            scores={length:{k:arm['scores'][length][k] for k in ('total_token_throughput','mean_ttft_ms','completed','failed','total_input_tokens','total_output_tokens')} for length in ('4096','16384')}),
        limits=['CPU rechecks preserved result guards and hashes; original tensors were not saved and are not recomputed.',
                'Timing covers captured resident-input 2D prefill kernels; excludes host dispatch, input copies and the model.',
                'Uniform reference covers all query rows; random reference covers selected query rows only.',
                'Inherited one-BF16-ULP baseline oracle exceptions remain raw allclose=false.',
                'Non-default side stream and changed-input poisoned Graph protocol follow the frozen measured source.',
                'Only actual TTGIR/LLIR and override IR bytes are saved; ASM/cubin have metadata hashes only.',
                'Two-sequence BM128 baseline preserves current Q32 parent: its num_seqs==1 eligibility does not match.',
                'Baseline arm passed; candidate accuracy/pair acceptance were unfinished at this snapshot.'])
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(raw_files=18,archive_bytes=len(packed),archive_sha256=sha(packed),component_cases=sum(len(v['cases']) for v in runs.values()),baseline_only=True,pair_audited=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--baseline-audit',type=Path,required=True)
    p.add_argument('--sdk-source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
