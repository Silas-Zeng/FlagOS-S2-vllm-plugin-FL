#!/usr/bin/env python3
"""No Torch/JIT/GPU: verify provenance, source gates, pitched address/byte contract."""
import argparse
import ast
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

from biv150_konly_decode_quant_reference import quantize_v2
from biv150_konly_i8_qk_reference import f32_bits
from biv150_konly_strided_writer_contract import SlotSnapshot, need, oracle, qualify, scatter_model
from biv150_konly_strided_writer_fixtures import SPECS, codec_vectors, make_case

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT/'tools/kernels/biv150_konly_strided_writer.py'
ARCHIVE = ROOT/'docs/evidence/konly-strided-writer-20261027/upstream79-codec-v1.tar.gz'
ARCHIVE_SHA = '278cc3be8228a1eff50a6af4a55d8e86564dff6b711850a4f9d3de1cfa5641ea'


def function_text(source, name):
    f = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    start = min([f.lineno]+[n.lineno for n in f.decorator_list])
    return ''.join(source.splitlines(keepends=True)[start-1:f.end_lineno])


def provenance():
    data = ARCHIVE.read_bytes()
    need(hashlib.sha256(data).hexdigest() == ARCHIVE_SHA, 'upstream archive SHA')
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
        members = tar.getmembers()
        need(len(members) == 5 and all(m.isfile() for m in members), 'upstream members')
        files = {m.name:tar.extractfile(m).read() for m in members}
    manifest = json.loads(files.pop('manifest.json'))
    need(set(manifest) == set(files), 'upstream inventory')
    for n, b in files.items():
        need(hashlib.sha256(b).hexdigest() == manifest[n]['sha256'] and len(b) == manifest[n]['bytes'], 'upstream file SHA')
    for target, old in [('biv150_konly_i8_qk_reference.py','primitive-reference.py'),
                        ('biv150_konly_decode_quant_reference.py','quant-reference.py')]:
        need((ROOT/'tools'/target).read_bytes() == files[old], 'unchanged old CPU codec')
    source = KERNEL.read_text()
    helper = function_text(source, 'quantize_bf16_rne')
    need(helper == function_text(files['support.py'].decode(), 'quantize_bf16_rne'), 'old device codec body')
    fixture = (ROOT/'tools/biv150_konly_strided_writer_fixtures.py').read_text()
    need(function_text(fixture, 'codec_vectors') == function_text(files['numeric-contract.py'].decode(), 'codec_vectors'), 'old141 vectors')
    return dict(archive_sha256=ARCHIVE_SHA, files=manifest, codec_function_sha256=hashlib.sha256(helper.encode()).hexdigest())


def check_kernel_source(source):
    f = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'encode_append')
    args = f.args.args
    need([a.arg for a in args] == ['key_source','value_source','slots','code_full','scale_full','value_full',
         'key_row_stride','value_row_stride','key_rows','value_rows','slot_count','cache_pages'], 'raw parameters')
    need(all(ast.unparse(a.annotation) == 'tl.int64' for a in args[6:]), 'I64 scalar ABI')
    guards = [n for n in f.body if isinstance(n, ast.If)]
    need(len(guards) == 2 and all(len(g.body) == 1 and isinstance(g.body[0], ast.Return) for g in guards), 'early-return guards')
    source_loads = [n for n in ast.walk(f) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'tl.load'
                    and ast.unparse(n.args[0]).startswith(('key_source','value_source'))]
    need(len(source_loads) == 2 and all(n.lineno > guards[1].end_lineno for n in source_loads), 'negative before source loads')
    need(ast.unparse(guards[0].test) == '(token >= slot_count) | (token >= key_rows) | (token >= value_rows)', 'extent guard')
    need(ast.unparse(guards[1].test) == '(slot < 0) | (slot >= cache_pages * 16)', 'negative/OOB guard')
    need([ast.unparse(n.args[0]) for n in source_loads] == [
        'key_source + token * key_row_stride + head * 128 + dim',
        'value_source + token * value_row_stride + head * 128 + dim'], 'independent pitches')
    stores = [n for n in ast.walk(f) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'tl.store']
    need([ast.unparse(n.args[0]) for n in stores] == [
        'code_full + page * 16384 + column * 256 + head * 128 + dim',
        'scale_full + page * 4096 + 1024 + column * 2 + head + tl.arange(0, 1)',
        'value_full + page * 8192 + 4096 + column * 256 + head * 128 + dim'], 'full-base stores')
    need(ast.unparse(stores[2].args[1]) == 'values', 'opaque BF16 V')
    return True


def run():
    prov = provenance()
    source = KERNEL.read_text()
    check_kernel_source(source)
    records = []
    for spec in SPECS:
        for phase in (0, 1):
            case = make_case(spec, phase)
            actual, binding = scatter_model(case)
            expected = oracle(case['before'], case['snapshot'].values, case['logical_keys'], case['logical_values'])
            need(actual == expected, 'pitched versus logical byte oracle')
            records.append(dict(name=case['name'], phase=phase, binding=binding,
                                expected_sha256=hashlib.sha256(expected).hexdigest(),
                                bytes=len(expected), shared_readonly_sources=case['shared_readonly_sources']))
    faults = []
    def reject(name, action):
        try: action()
        except (ValueError, TypeError, StopIteration): faults.append(name)
        else: raise AssertionError('accepted '+name)
    c = make_case(SPECS[2])
    def admission(**kw):
        d = {n:c[n] for n in ('key','value','slots','cache','pages','snapshot')};d.update(kw)
        return qualify(**d)
    for name, changed in [
        ('Kshort', dict(key=replace(c['key'], shape=(16,2,128)))),
        ('Vshort', dict(value=replace(c['value'], shape=(16,2,128)))),
        ('rowpitch255', dict(key=replace(c['key'], strides=(255,128,1)))),
        ('innerpad', dict(value=replace(c['value'], strides=(640,129,1)))),
        ('wrongD', dict(key=replace(c['key'], shape=(20,2,127)))),
        ('FP32source', dict(key=replace(c['key'], dtype='uint8'))),
        ('storage_short', dict(value=replace(c['value'], storage_bytes=300))),
        ('alias_cache', dict(key=replace(c['key'], storage_base=c['cache'].storage_base))),
        ('alias_slots', dict(key=replace(c['key'], storage_base=c['slots'].storage_base))),
        ('slotstride2', dict(slots=replace(c['slots'], strides=(2,)))),
        ('unknown_slots', dict(snapshot=None)),
        ('stale_unknown_provenance', dict(snapshot=replace(c['snapshot'], provenance='gpu-only-unknown'))),
        ('duplicate_slot', dict(snapshot=replace(c['snapshot'], values=(0,0)+c['snapshot'].values[2:]))),
        ('positive_oob', dict(snapshot=replace(c['snapshot'], values=(48,)+c['snapshot'].values[1:]))),
        ('missing_slot_value', dict(snapshot=replace(c['snapshot'], values=c['snapshot'].values[1:]))),
        ('bool_slot', dict(snapshot=replace(c['snapshot'], values=(True,)+c['snapshot'].values[1:]))),
        ('overflow_stride', dict(key=replace(c['key'], strides=(2**64,128,1)))),
    ]: reject(name, lambda changed=changed: admission(**changed))
    for name, old, new in [('shared_pitch','token * value_row_stride','token * key_row_stride'),
                           ('double_scale_offset','scale_full + page * 4096 + 1024','scale_full + page * 4096 + 2048'),
                           ('missing_V_base','value_full + page * 8192 + 4096','value_full + page * 8192'),
                           ('bad_slot_mask','(slot < 0) | (slot >= cache_pages * 16)','(slot >= cache_pages * 16)')]:
        need(old in source, 'mutation site')
        reject(name, lambda old=old,new=new: check_kernel_source(source.replace(old,new)))
    vectors = codec_vectors()
    coded = [quantize_v2(v) for v in vectors]
    bits = b''.join(bytes(c & 255 for c in code)+f32_bits(scale).to_bytes(4,'little') for code,scale in coded)
    need(not {'torch','triton','numpy'}.intersection(sys.modules), 'framework imported')
    return dict(cpu_only=True, passed=True, base_main='56f1838192abc940b3fb9accef490022ac125000',
                dependencies=[79,80,82], provenance=prov, kernel_sha256=hashlib.sha256(source.encode()).hexdigest(),
                cases=records, old_codec_vectors=len(vectors), old_codec_code_scale_sha256=hashlib.sha256(bits).hexdigest(),
                rejected=faults, device_entry_executed=False, live_slot_provenance_closed=False,
                service_installed=False, full_lifecycle_verified=False, precision_gate=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--out', type=Path);args = p.parse_args()
    text = json.dumps(run(), sort_keys=True, indent=2)+'\n'
    if args.out:
        need(not args.out.exists(), 'fresh output required');args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(text)
    else: print(text,end='')
