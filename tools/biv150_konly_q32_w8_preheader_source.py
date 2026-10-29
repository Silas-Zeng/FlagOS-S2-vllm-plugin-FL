"""One exact bound w8 QCode operand relocation; no inherited w16 qualification."""
import importlib.util
from pathlib import Path
import re
import json
import biv150_konly_q32_w8_compile_source as raw

ROOT, BASE = raw.ROOT, raw.BASE
PARENT = '7209a3c76869d00d44037dae7933fb678f26ef81'
RAW_HELPER_SHA = 'c20937f6860b4d727d88e00e124316f39198e6ef9ac6db1043f3b08e3a8607a5'
INPUT_SHA = '24e221001418328374a32c398cbd363fc73bf0b7eb03ab72c357a8b53ee1825d'
OVERRIDE_SHA = 'dfdc2c2af60b93153170ebbcc9477e8bf200fe243a03069f0e23aff4f7467aec'
SUMMARY_SHA = '44f8e64248152fc7415ddb6024d8f63ccded5a999659bb10967b46471fda8000'
RECORD_SHA = '8a0377f15fa7bb0fc53ecefbad03021949deaee25305445eb3210ab695fae702'
ORIGINAL_OUT = Path('/workspace/experiments/minicpm-s2/cjj-q32-i8-w8-compile-v1-20261030')
for _name in ('need','sha','NAME','FUNCTION_SHA','FILES','PARENT_FILES','DEPENDENCY',
              'kwargs','geometry','original','candidate_source','snapshot_source',
              'expected_sdk_pins','load_file','module_from_source','transform',
              'preserve_unqualified','observe_memory'):
    globals()[_name] = getattr(raw, _name)
need(sha(Path(raw.__file__).read_bytes()) == RAW_HELPER_SHA, 'frozen7209 compiler helper')


def hoist(text):
    need(sha(text) == INPUT_SHA, 'only the bound actual w8 input TTGIR is accepted')
    lines = text.splitlines(keepends=True)
    need(len(lines) == 908 and '%5 = ttg.local_alloc %4' in lines[311]
         and '%acc:3 = scf.for ' in lines[351] and '%10 = ttg.local_load %5 :' in lines[420],
         'actual allocation312 / loop352 / operand421')
    uses = lambda value: [i+1 for i,line in enumerate(lines)
        if re.search(re.escape(value)+r'(?![-.A-Za-z0-9_#])',line)]
    need(uses('%5') == [312,421] and uses('%10') == [421,423], 'unique Q memory and operand consumers')
    old = lines.pop(420)
    lines.insert(351, '    '+old.lstrip(' '))
    result = ''.join(lines)
    restored = lines.copy(); del restored[351]; restored.insert(420,old)
    need(''.join(restored) == text and sha(result) == OVERRIDE_SHA, 'one-line exact inverse / override SHA')
    return result, dict(input_sha256=INPUT_SHA,override_sha256=OVERRIDE_SHA,moved_operations=1,
        Q_allocation_line=312,old_Q_operand_line=421,old_KV_loop_line=352,
        new_Q_operand_line=352,new_KV_loop_line=353,exact_byte_inverse=True,
        all_other_text_math_masks_memory_and_layout_unchanged=True,
        old_w16_grammar_applied=False,actual_compiled_hoist_verified=False,
        Q_publish_bank_reuse_sync_verified=False,numerical_verified=False)


def bind_reviewed(directory):
    need(directory is not None and directory.is_absolute() and directory == ORIGINAL_OUT,
         'original7209 compile output/source/debug root required')
    body = (directory/'summary.json').read_bytes(); need(sha(body) == SUMMARY_SHA, 'actual raw w8 summary')
    summary = json.loads(body)
    need(summary['commit'] == PARENT and summary['completed'] is True and summary['errors'] == [], 'raw compile terminal/source')
    folder = directory/'raw-q32-w8-stage1/raw-w8-stage1'
    body = (folder/'record.json').read_bytes(); need(sha(body) == RECORD_SHA, 'actual complete w8 typed record')
    record = json.loads(body)
    for ext,pin in record['artifacts'].items(): need(sha((folder/('actual.'+ext)).read_bytes()) == pin, 'actual raw w8 '+ext)
    source = directory/'sources/candidate_q32_w8.py'
    need(source.read_text() == candidate_source() and record['source_digest'] == FUNCTION_SHA, 'original Q32 GPU/JIT source')
    input_ttgir = (folder/'actual.ttgir').read_text()
    override, proof = hoist(input_ttgir)
    return dict(original_out=str(directory),source=str(source),summary_sha256=SUMMARY_SHA,
        record_sha256=RECORD_SHA,record=record,tensor_descriptors=summary['tensor_descriptors'],
        original_handle=summary['records']['raw-q32-w8-stage1']['handle_resources']['raw-w8-stage1'],
        proof=proof,override=override,input_ttgir=input_ttgir)


def actual_ttgir(kernel, override):
    # Only SSA normalization is reused; no old target-specific hoist/sync grammar.
    path=ROOT/'vllm_fl/patches/biv150_konly_encoded_hoist_native.py'
    need(sha(path.read_bytes()) == '2027e1ce4a489f6b47c97a5253c1482d2dc767e62e9b10ec029e3f192c559ef2', 'SSA utility pin')
    helper=module_from_source('w8_preheader_alpha_only',path.read_text())
    need(helper.alpha(kernel.asm['ttgir']) == helper.alpha(override.read_text()), 'actual override IR differs beyond scoped SSA')


def driver_edits():
    return [
        ('raw Q32/w8/stage1 single warm compile','exact Q32/w8 preheader single warm compile'),
        ('import biv150_konly_q32_w8_compile_source as c','import biv150_konly_q32_w8_preheader_source as c'),
        ('from biv150_konly_q32_w8_compile_source import kwargs','from biv150_konly_q32_w8_preheader_source import kwargs'),
        ('        expected_pins = c.expected_sdk_pins()',
         "        reviewed = c.bind_reviewed(args.reviewed_out)\n        (args.out/'original-reviewed-w8.ttgir').write_text(reviewed.pop('input_ttgir'))\n        override = args.out/'q-local-preheader-w8.ttgir'\n        override.write_text(reviewed.pop('override'))\n        result['reviewed_w8'] = reviewed\n        result['override_path'] = str(override.resolve())\n        checkpoint()\n        expected_pins = c.expected_sdk_pins()"),
        ("modules = {32:c.load_file('cjj_private_q32_w8_scout', source)}",
         "modules = {32:c.load_file('cjj_private_q32_w8_scout', Path(reviewed['source']))}"),
        ('# No hoist proposal, override, native proof or qualification is used.',
         '# Only the fixed w8 relocation is used; no native/sync/numeric qualification is inherited.'),
        ("        for bq in (32,):", "        c.need(result['tensor_descriptors'] == reviewed['tensor_descriptors'], 'same actual raw w8 descriptors')\n        for bq in (32,):"),
        ("role = 'raw-q32-w8-stage1'", "role = 'preheader-q32-w8-stage1'"),
        ('first = fn.warmup(grid=grid, **launch)', 'first = fn.warmup(grid=grid, **launch, ir_override=str(override.resolve()))'),
        ("before = preserve(first,'raw-w8-stage1')", "before = preserve(first,'preheader-w8-stage1')\n            c.need(before['ABI'] == reviewed['record']['ABI'], 'complete actual typed ABI unchanged')\n            c.actual_ttgir(first,override)\n            record['actual_TTGIR_matches_fixed_override_modulo_SSA'] = True"),
        ("hasattr(first.metadata,'ir_override') and first.metadata.ir_override is None",
         "hasattr(first.metadata,'ir_override') and first.metadata.ir_override == str(override.resolve())"),
        ("'actual compiled metadata must explicitly contain no override'", "'actual compiled metadata must bind this fixed override path'"),
        ("for label, kernel in (('raw-w8-stage1',first),):", "for label, kernel in (('preheader-w8-stage1',first),):"),
        ("parser.add_argument('--commit'); parser.add_argument('--out', type=Path)",
         "parser.add_argument('--commit'); parser.add_argument('--out', type=Path)\n    parser.add_argument('--reviewed-out', type=Path)"),
        ('(args.commit,args.out,args.root_owned_exclusive)', '(args.commit,args.out,args.root_owned_exclusive,args.reviewed_out)'),
    ]


def driver_source(): return transform(raw.driver_source(), driver_edits())
def driver(): return module_from_source('cjj_w8_preheader_compile_driver',driver_source())


def plan():
    value=raw.plan();driver_source()
    value.update(schema='cjj-private-I8-Q32-w8-exact-preheader-compile-v1',parent=PARENT,
        driver_sha256=sha(driver_source()),planned_JIT_warmups=1,override_allowed=True,
        expected_raw_input_sha256=INPUT_SHA,expected_override_sha256=OVERRIDE_SHA,
        original_raw_summary_sha256=SUMMARY_SHA,original_raw_record_sha256=RECORD_SHA,
        original_source_debug_root=str(ORIGINAL_OUT),
        kernel_change_from_accepted_Q32_S1='w8 same-source reader plus one exact local_load relocation over its own actual w8 TTGIR; no w16 qualification',
        scope='One exact actual-w8 Q local_load421 to direct preheader352. Fresh actual Q/sync/resources review pending; zero numerical/timing/service permission.')
    return value
