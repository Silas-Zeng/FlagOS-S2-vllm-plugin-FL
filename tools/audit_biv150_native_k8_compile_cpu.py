#!/usr/bin/env python3
"""Independent stdlib actual-source/IR/ELF identities; never loads an ELF or framework."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import tarfile

from biv150_native_k8_source_factory import ROOT,verify

LIBRARY_SHA='22e4a5e2a0b9d9a36ea390a11126417e1bf58e3c27f6b96ba0913b27690f09dd'

def sha(b):return hashlib.sha256(b).hexdigest()

def msgpack(raw):
    at=0
    def take(n):
        nonlocal at
        data=raw[at:at+n];assert len(data)==n;at+=n;return data
    def number(n,signed=False):return int.from_bytes(take(n),'big',signed=signed)
    def read():
        byte=number(1)
        if byte<=127:return byte
        if byte>=224:return byte-256
        if 160<=byte<=191:return take(byte&31).decode()
        if 144<=byte<=159:return [read() for _ in range(byte&15)]
        if 128<=byte<=143:return {read():read() for _ in range(byte&15)}
        if byte==192:return None
        if byte in (194,195):return byte==195
        if byte in (204,205,206,207):return number(1<<(byte-204))
        if byte in (208,209,210,211):return number(1<<(byte-208),True)
        if byte in (217,218,219):return take(number(1<<(byte-217))).decode()
        if byte in (220,221):return [read() for _ in range(number(2<<(byte-220)))]
        if byte in (222,223):return {read():read() for _ in range(number(2<<(byte-222)))}
        if byte in (202,203):return struct.unpack('>f' if byte==202 else '>d',take(4 if byte==202 else 8))[0]
        raise ValueError('Unsupported metadata msgpack opcode '+str(byte))
    result=read();assert at==len(raw);return result

def elf(raw):
    assert raw[:6]==b'\x7fELF\x02\x01'
    hdr=struct.unpack_from('<16sHHIQQQIHHHHHH',raw);offset,entry,n,index=hdr[6],hdr[11],hdr[12],hdr[13]
    assert entry==64 and 0<n<10000 and offset+n*entry<=len(raw)
    rows=[struct.unpack_from('<IIQQQQIIQQ',raw,offset+i*entry) for i in range(n)]
    names=raw[rows[index][4]:rows[index][4]+rows[index][5]];sections={}
    extent=offset+n*entry
    for row in rows:
        start=row[0];name=names[start:names.index(b'\0',start)].decode()
        if row[1]!=8:
            assert row[4]+row[5]<=len(raw)
            data=raw[row[4]:row[4]+row[5]];extent=max(extent,row[4]+row[5])
        else:data=b''
        sections[name]=dict(type=row[1],bytes=data)
    return dict(machine=hdr[2],sections=sections,extent=extent)

def embedded(raw):
    host=elf(raw);fatbin=host['sections']['.nv_fatbin']['bytes'];result=[]
    for match in re.finditer(b'\x7fELF',fatbin):
        blob=fatbin[match.start():];gpu=elf(blob);assert gpu['machine']==248
        blob=blob[:gpu['extent']];note=gpu['sections']['.note']['bytes'];at=0;metadata=None
        while at<len(note):
            namesz,descsz,kind=struct.unpack_from('<III',note,at);at+=12
            name=note[at:at+namesz].rstrip(b'\0');at=(at+namesz+3)//4*4
            desc=note[at:at+descsz];at=(at+descsz+3)//4*4
            if name==b'Iluvatar' and kind==10:metadata=msgpack(desc)
        assert metadata and 'iluvatar.kernels' in metadata
        kernels=[{k:v for k,v in row.items() if k!='.args'} for row in metadata['iluvatar.kernels']]
        result.append(dict(sha256=sha(blob),text_sha256=sha(gpu['sections']['.text']['bytes']),
            text_bytes=len(gpu['sections']['.text']['bytes']),kernels=kernels))
    assert result;return result

def llvm_functions(text):
    starts=list(re.finditer(r'^define .*?@([^ (]+)\(.*?\{\n',text,re.M))
    return {m.group(1):text[m.start():text.index('\n}',m.end())+2] for m in starts}

def audit(files):
    manifest=json.loads(files['download-manifest.json']);assert manifest['raw_files']==28
    assert set(files)=={'download-manifest.json',*(row['relative'] for row in manifest['files'])}
    owner=json.loads(files['exports/cjj-native-k8-compile-v1-20261028-owner.json'])
    assert owner['source_commit']=='37713909f1f25b69d5ef6c155fad3b87351046fb'
    assert owner['pid']==688550 and owner['start_ticks']==347221367
    assert manifest['original_owner']['same_start'] and manifest['original_owner']['state']=='Z'
    for row in manifest['files']:
        raw=files[row['relative']];assert len(raw)==row['bytes'] and sha(raw)==row['sha256'],row['relative']
    assert sum(row['bytes'] for row in manifest['files'])==906281==manifest['raw_bytes']
    summary=json.loads(files['out/compile-summary.json']);assert summary['completed'] and summary['compile_passed'] and not summary['errors']
    assert all(summary[k]==0 for k in ('attention_launches','encoder_launches','codec_launches','reducer_launches','input_initializations'))
    assert summary['torch']['cuda_initialized_before'] is False and summary['torch']['cuda_initialized_after'] is False
    assert len(summary['commands'])==4 and all(row['exit_code']==0 for row in summary['commands'])
    for row in summary['artifact_manifest']:
        raw=files['out/'+row['relative']];assert len(raw)==row['bytes'] and sha(raw)==row['sha256']
    for name,expected in summary['private_source'].items():
        assert sha(files['out/'+Path(name).name])==expected
        assert (ROOT/name).read_bytes()==files['out/'+Path(name).name]
    source=verify();assert source==summary['source_transform']
    lib=files['out/build/cjj_native_k8_qk_probe.so'];assert sha(lib)==LIBRARY_SHA==summary['library']['sha256']
    codec=files['out/biv150_native_k8_codec.ll'].decode();qk=files['out/biv150_native_k8_qk.ll'].decode()
    codec_fn=next(iter(llvm_functions(codec).values()))
    assert len(re.findall(r'= fdiv float ',codec_fn))==3 and codec_fn.count('@llvm.nvvm.f2i.rn(')==2
    assert all('= fdiv float ' in line for line in (codec+qk).splitlines() if '= fdiv ' in line)
    assert '0x3810000000000000' in codec_fn and '1.000000e+00' in codec_fn and 'i32 -127' in codec_fn and 'i32 127' in codec_fn
    profiles=[]
    for name,fn in llvm_functions(qk).items():
        if 'corex_native_k8_gqa_producer' not in name:continue
        counts=dict(fdiv=len(re.findall(r'= fdiv float ',fn)),rne=fn.count('@llvm.nvvm.f2i.rn('),
          signed_I8_QK=fn.count('@llvm.bi.matrix.mad.i32x4.i8x8('),BF16_PV=fn.count('@llvm.bi.matrix.mad.f32x4.bf16x4('),
          signed_C_to_F32=len(re.findall(r'= sitofp i32 ',fn)))
        assert counts==dict(fdiv=18,rne=16,signed_I8_QK=8,BF16_PV=16,signed_C_to_F32=4),counts
        # Counts include the full-page and partial-page source paths; not16 PV/page.
        profiles.append(dict(symbol=name,static_counts=counts))
    assert len(profiles)==2
    asm_codec=files['out/biv150_native_k8_codec.s'].decode();asm_qk=files['out/biv150_native_k8_qk.s'].decode()
    assert len(re.findall(r'^\s*ml_matrix_mad_i32_i8_m16n16k32 ',asm_qk,re.M))==16
    assert len(re.findall(r'^\s*ml_matrix_mad_f32_bf16_m16n16k16 ',asm_qk,re.M))==32
    assert asm_codec.count('ml_rcp_f32')==1 and asm_codec.count('ml_cvt_i32_f32')==2 and asm_codec.count('ml_fma_f32')>=6
    actual=embedded(lib);objects=[]
    for name in ('biv150_native_k8_qk.cuda.o','biv150_native_k8_codec.cuda.o'):
        objects.extend(embedded(files['out/build/'+name]))
    assert sorted(row['sha256'] for row in actual)==sorted(row['sha256'] for row in objects)
    return dict(original_files=28,original_bytes=906281,all_collector_and_compile_manifest_hashes_verified=True,
      library_sha256=LIBRARY_SHA,extension_embedded_gpu_code_identical_to_object=True,
      actual_extension_gpu_ELFs=actual,standalone_llvm_profiles=profiles,
      standalone_codec=dict(unflagged_F32_divides=3,explicit_RNE_conversions=2,
        backend='rcp plus quotient/error FMA correction and class selection; no lone rcp claim',
        bit_exact_RN32_FTZ_device_conformance=False),
      standalone_and_extension_compiler_flags_identical=False,
      difference='Extension adds Torch includes/defines, ABI flags, -cl-single-precision-constant, -mllvm bonus-inst-threshold=0/skip-double, -fclang-abi-compat=16; standalone LLVM is not extension-object IR.',
      resources_scope='Actual embedded binary metadata only, no loaded-handle/postlaunch observation or occupancy claim',
      device_launches=0,CUDA_initialization=False,numerical_device_verified=False,model_verified=False,
      precision_gate=False,performance_measured=False,source_transform=source)

def from_raw(path):return {str(p.relative_to(path)):p.read_bytes() for p in path.rglob('*') if p.is_file()}

def from_archive(path):
    with tarfile.open(path,'r:gz') as tar:
        result={}
        for row in tar:
            assert row.isfile() and row.name not in result and not Path(row.name).is_absolute() and '..' not in Path(row.name).parts
            result[row.name]=tar.extractfile(row).read()
        return result

def archive(files,path):
    with path.open('wb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as gz:
            with tarfile.open(fileobj=gz,mode='w',format=tarfile.USTAR_FORMAT) as tar:
                for name,data in sorted(files.items()):
                    row=tarfile.TarInfo(name);row.size=len(data);row.mode=0o644;row.mtime=0;row.uid=row.gid=0;row.uname=row.gname='';tar.addfile(row,io.BytesIO(data))
    assert path.stat().st_size<=2*1024*1024

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--raw',type=Path);g.add_argument('--archive',type=Path)
    p.add_argument('--write-archive',type=Path);args=p.parse_args();files=from_raw(args.raw) if args.raw else from_archive(args.archive);result=audit(files)
    if args.write_archive:archive(files,args.write_archive)
    print(json.dumps(result,sort_keys=True,indent=2))

if __name__=='__main__':main()
