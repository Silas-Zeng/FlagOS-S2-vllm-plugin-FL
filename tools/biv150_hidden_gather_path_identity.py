"""Fixed 4a9f ELF identity modulo its known source-directory debug string only.

No compilation, binary modification on disk, CUDA or general ELF admission. Raw
hashes remain evidence; a future cold object must satisfy this complete comparison.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import struct

PINS=json.loads(Path(__file__).with_name('biv150_hidden_gather_sme1_pins.json').read_bytes())
OLD_ROOT='/workspace/experiments/minicpm-s2/code/cjj-native122-service-smoke-v1-20261030'
SUFFIX='/vllm_fl/patches'
EXTS=('source','ttir','ttgir','llir','asm','cubin')
COUNTS=dict(source=37,ttir=34,ttgir=34,llir=1,asm=2)
RULE='gather-4a9f-source-directory-dwarf-comment-v2'
HEADER=(b'\x7fELF\x02\x01\x01B\x01\x00\x00\x00\x00\x00\x00\x00',2,248,1,70976,64,3208,2,64,56,5,64,12,10)
PH=((6,4,64,65600,65600,280,280,8),(1,4,0,65536,65536,1344,1344,4096),(1,5,1344,70976,70976,1040,1040,4096),(1685382481,6,0,0,0,0,0,0),(4,4,344,65880,65880,880,880,4))
# Exact original section headers, including names/order, flags, VA and links.
SH=((0,0,0,0,0,0,0,0,0,0),(1,7,2,65880,344,880,0,0,4,0),(7,1,2,66816,1280,64,0,0,64,0),(15,1,6,70976,1344,1040,0,0,8,0),(21,1,0,0,2384,20,0,0,1,0),(35,1,0,0,2404,175,0,0,1,0),(47,1,0,0,2584,40,0,0,8,0),(60,1,0,0,2624,240,0,0,1,0),(72,1,48,0,2864,113,0,0,1,1),(81,2,0,0,2984,72,11,2,8,24),(89,3,0,0,3056,107,0,0,1,0),(99,3,0,0,3163,42,0,0,1,0))


def need(ok,message):
    if not ok:raise ValueError(message)


def file_id(data):return dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))


def root_bytes(root):
    need(type(root) is str and 1<len(root)<=4096 and re.fullmatch(r'/[A-Za-z0-9_./-]+',root),'source root syntax')
    need(str(PurePosixPath(root))==root and '..' not in PurePosixPath(root).parts,'source root canonical path')
    return root.encode()


def layout(info_size):
    need(11<info_size<8192,'bounded debug_info size')
    pos=344;rows=[SH[0]]
    for i,original in enumerate(SH[1:],1):
        row=list(original);pos+=(-pos)%row[8];row[4]=pos;row[5]=info_size if i==5 else row[5]
        pos+=row[5];rows.append(tuple(row))
    pos+=(-pos)%8
    return rows,pos


def pack(sections):
    rows,shoff=layout(len(sections[5]));header=list(HEADER);header[6]=shoff
    out=bytearray(shoff+12*64);out[:64]=struct.pack('<16sHHIQQQIHHHHHH',*header)
    for i,p in enumerate(PH):struct.pack_into('<IIQQQQQQ',out,64+i*56,*p)
    for i,(row,data) in enumerate(zip(rows,sections)):
        need(len(data)==row[5],'section payload size');out[row[4]:row[4]+row[5]]=data
        struct.pack_into('<IIQQQQIIQQ',out,shoff+i*64,*row)
    return bytes(out)


def canonical_cubin(raw,root):
    prefix=root_bytes(root);info_size=175+len(prefix)-len(OLD_ROOT)
    rows,shoff=layout(info_size);expected=list(HEADER);expected[6]=shoff
    need(len(raw)==shoff+12*64,'exact ELF size/no tail')
    need(struct.unpack_from('<16sHHIQQQIHHHHHH',raw)==tuple(expected),'fixed ELF header/derived shoff')
    need(tuple(struct.unpack_from('<IIQQQQQQ',raw,64+i*56) for i in range(5))==PH,'all program headers')
    need(tuple(struct.unpack_from('<IIQQQQIIQQ',raw,shoff+i*64) for i in range(12))==tuple(rows),'fixed section metadata/derived offsets')
    for row in rows[4:8]:
        need(row[2]==row[3]==0 and all(not(row[4]<p[2]+p[5] and p[2]<row[4]+row[5]) for p in PH if p[0]==1),'debug unmapped/non-ALLOC')
    sections=[raw[r[4]:r[4]+r[5]] for r in rows]
    need(pack(sections)==raw,'all gaps zero; no unaccounted bytes')
    info=sections[5];directory=prefix+SUFFIX.encode()+b'\0'
    need(info[:11]==struct.pack('<IHI',len(info)-4,2,0)+b'\x08','exact DWARF2 unit header')
    need(info.count(directory)==1 and info.find(directory)==64 and info.count(prefix)==1,'unique exact comp_dir at64')
    info=info.replace(directory,OLD_ROOT.encode()+SUFFIX.encode()+b'\0')
    sections[5]=struct.pack('<I',len(info)-4)+info[4:]
    canonical=pack(sections)
    need(file_id(canonical)==PINS['files']['cubin'],'complete canonical ELF differs')
    return canonical,info_size-4


def describe(artifacts,source_root):
    # Call only with a root obtained from the already verified Python module/code.
    raw_files={k:file_id(v) for k,v in artifacts.items()}
    result=dict(rule=RULE,rule_source_sha256=file_id(Path(__file__).read_bytes())['sha256'],source_root=source_root,reference_root=OLD_ROOT,raw_files=raw_files,verified=False)
    try:
        need(set(artifacts)==set(EXTS) and all(type(v) is bytes and len(v)<1048576 for v in artifacts.values()),'bounded exact artifact set')
        prefix=root_bytes(source_root);binary,unit_length=canonical_cubin(artifacts['cubin'],source_root)
        normalized={};counts={}
        for ext in EXTS[:-1]:
            data=artifacts[ext];data.decode('utf-8');counts[ext]=data.count(prefix)
            need(counts[ext]==COUNTS[ext],'exact source path count: '+ext)
            if ext=='asm':
                comment=re.compile(rb'(?m)^(\t.byte\t1 +; Abbrev \[1\] 0xb:)(0x[0-9a-f]+)( DW_TAG_compile_unit)$')
                found=list(comment.finditer(data))
                need(len(found)==1 and int(found[0][2],16)==unit_length-7,'ASM unique compile-unit comment must equal actual ELF unit_length-7')
                comment_proof=dict(original=found[0][2].decode(),reference='0xa4',actual_ELF_unit_length=unit_length,
                                   derivation='unit_length-7',changed=found[0][2]!=b'0xa4')
                data=comment.sub(lambda m:m[1]+b'0xa4'+m[3],data)
                matches=list(re.finditer(rb'(?m)^([ \t]*\.long[ \t]+)([0-9]+)([ \t]+; Length of Unit)$',data))
                need(len(matches)==1 and int(matches[0][2])==unit_length,'ASM unit length must equal actual ELF')
                a,b=matches[0].span(2);data=data[:a]+b'171'+data[b:]
            normalized[ext]=file_id(data.replace(prefix,OLD_ROOT.encode()))
        normalized['cubin']=file_id(binary)
        need(normalized==PINS['files'],'complete normalized text/ELF differs')
        result.update(verified=True,canonical_files=normalized,path_counts=counts,ELF_actual_unit_length=unit_length,ASM_compile_unit_comment=comment_proof)
    except (ValueError,struct.error) as error:
        result['reason']=str(error)
    return result
