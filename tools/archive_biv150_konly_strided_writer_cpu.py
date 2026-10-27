#!/usr/bin/env python3
"""Stdlib-only replay of saved PR83 writer bytes/ABI/resources; never a device run."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import tarfile

sys.dont_write_bytecode = True
from bench_biv150_konly_strided_writer import EXPECTED_SIGNATURE, FRONTEND
from biv150_konly_decode_quant_reference import quantize_v2
from biv150_konly_strided_writer_contract import PAGE, SlotSnapshot, View, bf16_value, need, qualify
from biv150_konly_strided_writer_fixtures import GUARD, SPECS, make_case
from check_biv150_konly_strided_writer_cpu import KERNEL, ROOT, run

COMMIT = '58bace0b73614d2785d55b70a8376b91cb5e893d'
TAG = 'cjj-konly-strided-writer-v1-20261027'
DOWNLOAD_MANIFEST_SHA = '2280fec62481ef5cd69f563957de4829cf66782a8e00be6857a80da1fbb00347'
EVIDENCE = ROOT/'docs/evidence/konly-strided-writer-20261027/device-v1'
IR_TYPES = ['!tt.ptr<bf16>', '!tt.ptr<bf16>', '!tt.ptr<i64>', '!tt.ptr<i8>',
            '!tt.ptr<f32>', '!tt.ptr<bf16>']+['i64']*6


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True)+'\n').encode()


def canonical(value):
    return json.loads(json.dumps(value))


def safe_name(name):
    p = PurePosixPath(name)
    need(type(name) is str and not p.is_absolute() and '..' not in p.parts
         and str(p) == name and name != '.', 'unsafe raw member')
    return name


def from_directory(root):
    need(root.is_dir(), 'raw directory')
    need(not any(p.is_symlink() for p in root.rglob('*')), 'raw symlink')
    return {safe_name(p.relative_to(root).as_posix()): p.read_bytes()
            for p in sorted(root.rglob('*')) if p.is_file()}


def from_archive(path):
    files = {}
    with tarfile.open(path, 'r:gz') as tar:
        for m in tar.getmembers():
            name = safe_name(m.name)
            need(m.isfile() and name not in files, 'archive regular unique members')
            files[name] = tar.extractfile(m).read()
    return files


def archive_bytes(files):
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode='w', format=tarfile.PAX_FORMAT) as tar:
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(safe_name(name)); info.size = len(data)
            info.mode = 0o644; info.mtime = 0; info.uid = info.gid = 0
            info.uname = info.gname = ''; tar.addfile(info, io.BytesIO(data))
    out = io.BytesIO()
    with gzip.GzipFile(filename='', fileobj=out, mode='wb', mtime=0, compresslevel=9) as gz:
        gz.write(raw.getvalue())
    need(len(out.getvalue()) <= 2*1024*1024, 'publisher archive exceeds 2 MiB')
    return out.getvalue()


def view(raw):
    return View(**{**raw, 'shape': tuple(raw['shape']), 'strides': tuple(raw['strides'])})


def replay_cache(case, key_bytes, value_bytes, slots):
    """Read saved pitched source bytes; scatter old RN32/RNE codec by byte offset.

    This uses the unchanged old CPU codec, not saved expected cache/scale/code.
    V is copied as 256 opaque source bytes, including NaN payloads and signed 0.
    """
    output = bytearray(case['before']); valid = 0
    for token, physical in enumerate(slots):
        if physical < 0:
            continue
        valid += 1; page, slot = divmod(physical, 16)
        for head in range(2):
            ko = case['key'].byte_offset+2*(token*case['key'].strides[0]+head*128)
            vo = case['value'].byte_offset+2*(token*case['value'].strides[0]+head*128)
            bits = struct.unpack_from('<128H', key_bytes, ko)
            codes, scale = quantize_v2([bf16_value(b) for b in bits])
            a = page*PAGE+slot*256+head*128
            output[a:a+128] = bytes(c & 255 for c in codes)
            struct.pack_into('<f', output, page*PAGE+4096+slot*8+head*4, scale)
            a = page*PAGE+8192+slot*512+head*256
            output[a:a+256] = value_bytes[vo:vo+256]
    return bytes(output), valid


def inventory(files):
    need(digest(files['download-manifest.json']) == DOWNLOAD_MANIFEST_SHA, 'immutable collector manifest SHA')
    manifest = json.loads(files['download-manifest.json'])
    need(manifest['tag'] == TAG and manifest['read_only'] is True
         and manifest['model_rerun'] is False, 'download provenance')
    need(manifest['original_owner'] == dict(pid=636247, start_ticks=345046048,
         same_start=True, state='Z'), 'collector owner snapshot')
    inventory = manifest['files']; names = [safe_name(f['relative']) for f in inventory]
    need(len(names) == len(set(names)) == manifest['raw_files'] == 160, 'raw inventory count')
    need(set(files) == set(names)|{'download-manifest.json'}, 'exact raw inventory')
    need(sum(len(files[n]) for n in names) == manifest['raw_bytes'] == 4751338, 'raw byte count')
    for f in inventory:
        b = files[f['relative']]
        need(len(b) == f['bytes'] and digest(b) == f['sha256'], 'raw SHA '+f['relative'])
    return manifest


def content(files, cpu, manifest):
    owner = json.loads(files['exports/'+TAG+'-owner.json'])
    need((owner['pid'], owner['start_ticks'], owner['source_commit'], owner['official_score'])
         == (636247,345046048,COMMIT,False), 'original owner')
    need(owner['draft_pr'].endswith('/pull/83') and '--run-gpu' in owner['command']
         and owner['command'][owner['command'].index('--commit')+1] == COMMIT, 'owner command')
    result = json.loads(files['out/result.json'])
    for name in ('completed','passed','component_only','requested_writer_gates_passed'):
        need(result[name] is True, 'required component gate '+name)
    for name in ('performance_measured','precision_gate','model_or_service_verified',
                 'live_slot_lifecycle_verified','full_service_or_lifecycle_verified'):
        need(result[name] is False, 'unearned service/performance claim '+name)
    need(result['commit'] == COMMIT and result['errors'] == [], 'source/result failure')
    need(result['cpu'] == canonical(cpu), 'frozen CPU source/admission/old-codec replay')
    need(result['target'] == dict(arch=71, backend='corex', warp_size=64), 'actual target')
    need(result['frontend_sha256'] == FRONTEND, 'frontend pins')
    need(files['out/kernel-source.py'] == KERNEL.read_bytes(), 'actual saved writer source')
    records = result['records']; need([r['name'] for r in records] == [s[0] for s in SPECS], 'case matrix')
    checks = []; resources = []; hashes = set(); ir_files = set(); typed = []
    for spec, rec in zip(SPECS, records):
        case = make_case(spec,0); name = case['name']
        views = {n:view(v) for n,v in rec['actual_views'].items()}
        need(set(views) == {'key','value','slots','raw'}, 'actual view inventory')
        for role in ('key','value','slots'):
            old = case[role]; actual = views[role]
            need((actual.dtype,actual.shape,actual.strides,actual.storage_bytes,actual.byte_offset,actual.device)
                 == (old.dtype,old.shape,old.strides,old.storage_bytes,old.byte_offset,old.device), 'source layout '+role)
        c = case['cache']; v = views['raw']
        need((v.dtype,v.shape,v.strides,v.storage_bytes,v.byte_offset,v.device)
             == (c.dtype,c.shape,c.strides,c.storage_bytes,c.byte_offset,c.device), 'cache full-base layout')
        ptrs = rec['full_base_pointers']
        need(set(ptrs) == {'raw','code','scale','vfull'} and len(set(ptrs.values())) == 1
             and ptrs['raw'] == v.storage_base+v.byte_offset, 'all typed cache pointers share page0')
        need(rec['shared_readonly_sources'] is case['shared_readonly_sources'], 'shared QKV flag')
        need((views['key'].storage_base == views['value'].storage_base) is case['shared_readonly_sources'], 'readonly QKV backing')
        plan = qualify(views['key'],views['value'],views['slots'],v,case['pages'],case['snapshot'])
        need(rec['binding'] == canonical(plan) and rec['launched'] is plan['launch'], 'actual admission plan')
        labels = [('direct',0)] if plan['launch'] else [('empty-no-launch',0)]
        if name in ('tail17_k384_v640','shared_qkv_2560'):
            labels += [('changed-graph',1),('changed-side',1)]
        need([(c['label'],c['phase']) for c in rec['checks']] == labels, 'changed Graph/side coverage')
        for check in rec['checks']:
            current = make_case(spec,check['phase']); label = check['label']
            prefix = 'out/'+name+'/'+label+'-'
            kb = files[prefix+'key-source.bin']; vb = files[prefix+'value-source.bin']
            need(kb == current['buffers']['key'] and vb == current['buffers']['value'], 'saved fixture source/padding')
            sb = files[prefix+'slots.bin']; count = len(current['snapshot'].values)
            need(sb[:GUARD] == sb[-GUARD:] == b'\x6d'*GUARD and len(sb) == count*8+2*GUARD, 'saved slot snapshot guards')
            slots = struct.unpack_from('<'+'q'*count,sb,GUARD)
            need(slots == current['snapshot'].values and list(slots) == check['slot_values'], 'changed slot snapshot')
            changed_plan = qualify(views['key'],views['value'],views['slots'],v,current['pages'],
                                   SlotSnapshot(slots,'component-owned-cpu-preupload',check['phase']))
            need(changed_plan['slot_epoch'] == check['phase'], 'phase freshness')
            expected, valid = replay_cache(current,kb,vb,slots)
            actual = files[prefix+'actual.bin']; saved_expected = files[prefix+'expected.bin']
            need(actual == expected == saved_expected, 'independent raw cache-byte codec replay '+name+'/'+label)
            need(check['sha256'] == digest(actual) and check['complete_cache_bytes'] == len(actual)
                 and check['all_bytes_exact'] is True and check['source_slot_canaries_unchanged'] is True, 'raw check mirror')
            checks.append(dict(case=name,label=label,phase=check['phase'],cache_bytes=len(actual),
                               valid_slots=valid,negative_slots=count-valid,sha256=digest(actual)))
        if not plan['launch']:
            need('pre_launch' not in rec and 'post_launch' not in rec, 'empty case no device launch')
            continue
        need(rec['pre_launch']['hash'] == rec['post_launch']['hash'], 'warm/loaded kernel identity')
        for phase in ('pre_launch','post_launch'):
            record = rec[phase]; need(record['phase'] == phase, 'resource phase')
            need(record['signature'] == EXPECTED_SIGNATURE, 'typed actual signature')
            md = record['metadata']
            need(md['num_warps'] == 1 and md['num_stages'] == 1 and md['warp_size'] == 64, 'launch options')
            need(type(md['shared']) is int and md['shared'] >= 0, 'shared resource known')
            need(set(record['files']) >= {'source','ttir','ttgir','llir','asm','cubin'}, 'actual artifact set')
            for kind, info in record['files'].items():
                path = 'out/'+name+'/compiler/'+safe_name(info['file']); data = files[path]
                need(info['bytes'] == len(data) > 0 and info['sha256'] == digest(data), 'actual artifact SHA')
                ir_files.add(path)
                if kind in ('ttir','ttgir'):
                    line = next((s for s in data.decode().splitlines() if 'tt.func public @encode_append(' in s), '')
                    actual_types = re.findall(r'%\w+: (!tt\.ptr<\w+>|i64)',line)
                    need(actual_types == IR_TYPES, 'actual '+kind+' 6-pointer/6-I64 ABI')
            if phase == 'post_launch':
                need(record['loaded_resources_verified'] is True and type(record['n_regs']) is int
                     and record['n_regs'] > 0 and type(record['n_spills']) is int and record['n_spills'] == 0,
                     'actual loaded resource gate')
                resources.append(dict(case=name,hash=record['hash'],n_regs=record['n_regs'],
                                      n_spills=record['n_spills'],**md))
                hashes.add(record['hash']); typed.append(name)
    need(len(checks) == 13 and len(resources) == 8 and len(hashes) == 5, 'observed matrix totals')
    return dict(component_replay_passed=True,tested_commit=COMMIT,
                download_manifest_sha256=digest(files['download-manifest.json']),
                raw_files=160,raw_bytes=4751338,owner_snapshot=manifest['original_owner'],
                result_sha256=digest(files['out/result.json']),writer_sha256=digest(KERNEL.read_bytes()),
                case_count=len(records),launched_cases=8,empty_no_launch_cases=1,
                cache_byte_checks=checks,checked_cache_bytes=sum(c['cache_bytes'] for c in checks),
                changed_graph_checks=2,changed_side_checks=2,actual_kernel_hashes=sorted(hashes),
                actual_saved_artifacts_checked=len(ir_files),loaded_resources=resources,
                actual_ttir_ttgir_6pointer_6I64_verified=typed,
                cpu_admission_rejections=cpu['rejected'],old_codec_vectors=cpu['old_codec_vectors'],
                old_codec_valid_head_rows=cpu['codec141_valid_head_rows'],
                independently_replayed='saved actual full cache bytes from saved CPU fixture source/slots',
                guard_evidence='original bench asserted GPU source/slot/canary readback; saved source/slots are CPU expected bytes; outer canaries not separately saved',
                component_only=True,precision_gate=False,performance_measured=False,
                full_service_or_lifecycle_verified=False,model_or_service_verified=False,
                device_or_framework_executed_by_this_replay=False)


def audit(files,cpu):
    return content(files,cpu,inventory(files))


def faults(files,cpu):
    rejected = {}
    reasons = dict(output_code='raw cache-byte',reserved_cache_byte='raw cache-byte',
                   slot_value='slot snapshot',typed_signature='typed actual signature',
                   loaded_spill='loaded resource',bad_source_stride='source layout',
                   missing_graph='Graph/side coverage',unearned_precision='unearned service',
                   typed_TTIR='actual ttir')
    for name in ('output_code','reserved_cache_byte','slot_value','typed_signature',
                 'loaded_spill','bad_source_stride','missing_graph','unearned_precision','typed_TTIR'):
        f = dict(files); r = json.loads(f['out/result.json'])
        if name in ('output_code','reserved_cache_byte'):
            p = 'out/one_slot_extra_rows/direct-actual.bin'; b = bytearray(f[p]); b[0 if name == 'output_code' else 5000] ^= 1; f[p] = bytes(b)
        elif name == 'slot_value':
            p='out/tail17_k384_v640/changed-graph-slots.bin'; b=bytearray(f[p]); b[GUARD] ^= 1; f[p]=bytes(b)
        elif name == 'typed_signature': r['records'][1]['post_launch']['signature'][3]='*bf16'
        elif name == 'loaded_spill': r['records'][1]['post_launch']['n_spills']=1
        elif name == 'bad_source_stride': r['records'][2]['actual_views']['key']['strides'][0]=256
        elif name == 'missing_graph': r['records'][2]['checks'].pop(1)
        elif name == 'unearned_precision': r['precision_gate']=True
        else:
            record=r['records'][1]['post_launch']; info=record['files']['ttir']
            p='out/one_slot_extra_rows/compiler/'+info['file']
            f[p]=f[p].replace(b'%code_full: !tt.ptr<i8>',b'%code_full: !tt.ptr<u8>')
            for phase in ('pre_launch','post_launch'):
                r['records'][1][phase]['files']['ttir'].update(bytes=len(f[p]),sha256=digest(f[p]))
        f['out/result.json']=encoded(r)
        # Inject into content after the normal immutable inventory has passed.
        # Assert the intended semantic gate rejects, not an incidental SHA/size.
        try: content(f,cpu,json.loads(files['download-manifest.json']))
        except (ValueError,KeyError,IndexError,StopIteration) as exc:
            need(reasons[name] in str(exc), 'fault hit unintended gate '+name+': '+str(exc))
            rejected[name]=str(exc)
        else: raise AssertionError('accepted evidence fault '+name)
    return rejected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir',type=Path);p.add_argument('--archive',type=Path)
    p.add_argument('--build-evidence',type=Path);p.add_argument('--out',type=Path)
    a=p.parse_args();need(not (a.raw_dir and a.archive),'one source')
    files=from_directory(a.raw_dir) if a.raw_dir else from_archive(a.archive or EVIDENCE/'raw-v1.tar.gz')
    cpu=run(); summary=audit(files,cpu); summary['evidence_faults_rejected']=faults(files,cpu)
    packed=archive_bytes(files)
    summary['archive']=dict(file='raw-v1.tar.gz',bytes=len(packed),sha256=digest(packed),members=len(files))
    if a.build_evidence:
        need(not a.build_evidence.exists(),'fresh evidence output');a.build_evidence.mkdir(parents=True)
        (a.build_evidence/'raw-v1.tar.gz').write_bytes(packed)
        (a.build_evidence/'summary.json').write_bytes(encoded(summary))
        (a.build_evidence/'archive-manifest.json').write_bytes(encoded(dict(archive=summary['archive'],
                 summary_sha256=digest(encoded(summary)),members=[dict(file=n,bytes=len(b),sha256=digest(b)) for n,b in sorted(files.items())])))
    if a.out:a.out.write_bytes(encoded(summary))
    print(json.dumps(dict(component_replay_passed=True,raw_files=160,case_count=9,
                         cache_byte_checks=13,precision_gate=False,performance_measured=False)))


if __name__ == '__main__':main()
