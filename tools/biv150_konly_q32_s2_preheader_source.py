"""One fixed post-pipeline S2 Q-read relocation. Stdlib/default CPU only."""
import ast
import json
from pathlib import Path
import re
import biv150_konly_q32_s2_compile_source as raw

ROOT=raw.ROOT
BASE=raw.BASE
PARENT='cb9088cabef96615604a891089d76e2eadccdbb6'
FACTORY_SHA='ddeb922f473e54d446d915c8429ac1595f191d34641853b8f3088f4c61ee1445'
OVERRIDE_SHA='9f45fd740d0eca78f00e80f43b890a1a624301f12aee84664c3707a41e9326db'
NAME=raw.NAME
FUNCTION_SHA=raw.FUNCTION_SHA
need=raw.need
sha=raw.sha
importlib=raw.importlib
kwargs=raw.kwargs
geometry=raw.geometry
candidate_source=raw.candidate_source
snapshot_source=raw.snapshot_source
load_file=raw.load_file
expected_sdk_pins=raw.expected_sdk_pins
ORIGINALS={
 'summary.json':'6e5d4877056b3c11d86824d88bf905dcce27b5f3e1b4a16167cec85fb775ea09',
 'raw-q32-stage2/raw-stage2/actual.ttgir':'02e0549f8f01e154ad2f66ccbc6db43ec9d886578d8e5bbd70cf89c431f93444',
 'raw-q32-stage2/raw-stage2/actual.llir':'440f3ac5e5b41585fa7ed31d50e9421786738f67c03ea5b1f563c5676fb04236',
 'raw-q32-stage2/raw-stage2/actual.asm':'6f20d35132c473a2843222a518e0aad2a91ce7ec2bb1600c1bea7e1ae78d694f',
 'raw-q32-stage2/raw-stage2/actual.cubin':'d6264d6f480933a5d596339c464f97ad04be32d3e7625137a7c9fbae9637852f',
}


def hoist(text):
    """Only this complete actual module is admitted; no general async grammar."""
    need(type(text) is str and sha(text)==ORIGINALS['raw-q32-stage2/raw-stage2/actual.ttgir'],
         'only exact reviewed cb908 actual S2 TTGIR')
    lines=text.splitlines(keepends=True)
    alloc,header,qload,kload,dot=(lines[i-1] for i in (314,425,439,440,441))
    need('ttg.local_alloc %4' in alloc and 'mutable' not in alloc,'immutable initialized Q')
    need([i for i,l in enumerate(lines,1) if re.search(r'%5(?![\w])',l)]==[314,439], 'sole Q allocation/read; no alias')
    need([i for i,l in enumerate(lines,1) if re.search(r'%10(?![\w])',l)]==[439,441], 'sole Q result/dot use')
    need('scf.for ' in header and 'iter_args(' in header and '%V_load_279 = %V_load_254' in header,'original S2 header/V carry')
    need('ttg.local_load %5 :' in qload and 'token' not in qload and 'opIdx = 0' in qload,'independent Q operand')
    need('token %K_load_285' in kload and 'tt.dot %10, %K_load_293' in dot,'unchanged dependent K read/dot')
    fresh=qload[2:].replace('%10 =','%cjj_s2_q_operand =',1)
    need('%cjj_s2_q_operand' not in text,'fresh SSA name')
    edits=[(header,fresh+header),(qload+kload,kload),(dot,dot.replace('%10,','%cjj_s2_q_operand,',1))]
    transformed=raw.transform(text,edits)
    need(sha(transformed)==OVERRIDE_SHA,'exact sole-relocation output')
    # Exact reversal above covers all async operations, masks, V, math, yield and locations.
    need(transformed.count('ttg.local_load %5 :')==1 and
         transformed.index('ttg.local_load %5 :')<transformed.index(header),'only preheader Q read')
    return transformed,dict(original_sha256=sha(text),override_sha256=sha(transformed),
        operation='one Q local_load439 to directly before loop425 plus fresh SSA/one dot operand rename',
        edits=edits,byte_exact_inverse=True,pipeline_operations_unchanged=True,
        early_Q=False,new_actual_hoist_verified=False,synchronization_verified=False)


def reviewed(out):
    need(type(out) is Path or isinstance(out,Path),'explicit original compile output Path')
    need(out.is_absolute() and out.is_dir(),'absolute existing original output')
    for name,digest in ORIGINALS.items():need(sha((out/name).read_bytes())==digest,'original artifact '+name)
    summary=json.loads((out/'summary.json').read_bytes())
    need(summary['commit']==PARENT and summary['completed'] is True and summary['errors']==[] and
         summary['attention_launches']==summary['writer_launches']==summary['encoder_launches']==0,'original compile-only provenance')
    need(summary['num_stages']==2 and summary['num_seqs']==1 and summary['query_rows']==2048,
         'only actual N1/Q2048 S2')
    before=summary['records']['raw-q32-stage2']['raw-stage2']
    need(before['function_name']==NAME and before['source_digest']==FUNCTION_SHA,'original Q32 source identity')
    text=(out/'raw-q32-stage2/raw-stage2/actual.ttgir').read_text()
    override,proof=hoist(text)
    return dict(original_out=str(out),original_source=PARENT,original_artifacts=ORIGINALS,
                original_function_digest=FUNCTION_SHA,override=override,proof=proof,
                original_locations_preserved=True,original_source_root_is_not_new_checkout=True)


def driver_edits():
    return [
      ('Root-owned raw Q32/stage2 single warm compile, zero launches.', 'Root-owned fixed S2 Q-read preheader warm compile, zero launches.'),
      ('import biv150_konly_q32_s2_compile_source as c','import biv150_konly_q32_s2_preheader_source as c'),
      ('from biv150_konly_q32_s2_compile_source import kwargs','from biv150_konly_q32_s2_preheader_source import kwargs'),
      ("    args.out.mkdir(parents=True)","    reviewed = c.reviewed(args.reviewed_out)\n    c.need(not args.out.resolve().is_relative_to(args.reviewed_out.resolve()), 'new output must be separate from readonly original')\n    args.out.mkdir(parents=True)\n    override = args.out/'q-local-preheader.ttgir'\n    override.write_text(reviewed['override'])\n    (args.out/'original-reviewed.ttgir').write_bytes((args.reviewed_out/'raw-q32-stage2/raw-stage2/actual.ttgir').read_bytes())"),
      ("    checkpoint()\n    try:","    result['reviewed_stage2'] = {k:v for k,v in reviewed.items() if k!='override'}\n    result['override_path'] = str(override.resolve())\n    checkpoint()\n    try:"),
      ('# No hoist proposal, override, native proof or qualification is used.', '# Only the fixed S2 Q-read override; no S1/native qualification is inherited.'),
      ("'raw scout forbids an override argument'", "'ABI mapper must supply no external override'"),
      ("role = 'raw-q32-stage2'","role = 'preheader-q32-stage2'"),
      ("                'actual compiler override manager must be disabled')","                'actual compiler override manager must be disabled')\n            launch['ir_override'] = str(override.resolve())"),
      ("before = preserve(first,'raw-stage2')","before = preserve(first,'preheader-stage2')"),
      ("hasattr(first.metadata,'ir_override') and first.metadata.ir_override is None,\n                'actual compiled metadata must explicitly contain no override'","hasattr(first.metadata,'ir_override') and first.metadata.ir_override == str(override.resolve()),\n                'actual compiled metadata must bind this exact S2 override path'"),
      ("for label, kernel in (('raw-stage2',first),):","for label, kernel in (('preheader-stage2',first),):"),
      ("        result.update(completed=True, compile_passed=True,","        c.need(c.reviewed(args.reviewed_out)==reviewed and c.sha(override.read_bytes())==reviewed['proof']['override_sha256'], 'original artifacts/override unchanged')\n        result.update(completed=True, compile_passed=True,"),
      ("    parser.add_argument('--commit'); parser.add_argument('--out', type=Path)","    parser.add_argument('--commit'); parser.add_argument('--out', type=Path)\n    parser.add_argument('--reviewed-out', type=Path)"),
      ("any((args.commit,args.out,args.root_owned_exclusive))","any((args.commit,args.out,args.reviewed_out,args.root_owned_exclusive))"),
    ]


def driver_source():return raw.transform(raw.driver_source(),driver_edits())


def driver():return raw.module_from_source('cjj_private_q32_s2_preheader_driver',driver_source())


def plan():
    need(sha((ROOT/'tools/biv150_konly_q32_s2_compile_source.py').read_bytes())==FACTORY_SHA,'unchanged cb908 source factory')
    value=raw.plan();driver_source()
    value.update(schema='cjj-q32-s2-preheader-compile-falsifier-v1',parent=PARENT,
        raw_stage2_artifact_pins=ORIGINALS,expected_override_sha256=OVERRIDE_SHA,driver_sha256=sha(driver_source()),
        override_allowed='only exact actual S2 module with sole Q-read relocation',
        hoist_proposal_used=True,new_actual_Q_hoist_verified=False,
        pipeline_byte_inverse_required=True,new_sync_and_ELF_audit_pending=True,
        source_root_distinction='new checkout generates original Q32 source; fixed override preserves cb908 original IR/debug locations',
        scope='N1/Q2048 only; one warmup,0 reader/writer/encoder launches; compile receipt is not mechanism/numerical/performance/service admission')
    return value
