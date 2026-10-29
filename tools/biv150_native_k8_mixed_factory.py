"""Host-only exact-row admission; original accepted CUDA/codec files are unchanged."""
import hashlib
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE='a455a3de364051fbea80226870c239de3a48e716'
NAMESPACE='cjj_native_k8_mixed_probe'
OLD_NAMESPACE='cjj_native_k8_v_overlap_probe'
FILES={'qk.cpp':'biv150_native_k8_v_overlap_qk.cpp','qk.cu':'biv150_native_k8_v_overlap_qk.cu',
       'codec.cu':'biv150_native_k8_v_overlap_codec.cu','codec.cuh':'biv150_native_k8_codec.cuh'}
OLD=' const std::int64_t rows=segments==2?64:32;'
NEW=(' const std::int64_t rows=q.dim()==3?q.size(0):0;\n'
     ' TORCH_CHECK((segments==2 && (rows==64 || rows==63 || rows==61 || rows==60)) ||\n'
     '             (segments==4 && (rows==32 || rows==31 || rows==30 || rows==29)),"exact accepted or mixed rows/S only");')

def sha(b):return hashlib.sha256(b).hexdigest()
def original(role):return subprocess.check_output(['git','-C',str(ROOT),'show',BASE+':tools/kernels/'+FILES[role]])
def path(role):return ROOT/'tools/kernels'/('biv150_native_k8_mixed_qk.cpp' if role=='qk.cpp' else FILES[role])
def generate():
    b=original('qk.cpp');assert b.count(OLD.encode())==1
    return b.replace(OLD.encode(),NEW.encode()).replace(OLD_NAMESPACE.encode(),NAMESPACE.encode())
def verify():
    records={}
    for role in FILES:
        actual=path(role).read_bytes();old=original(role)
        if role=='qk.cpp':
            assert actual==generate(),'host changes beyond exact row admission/namespace'
            assert actual.replace(NEW.encode(),OLD.encode()).replace(NAMESPACE.encode(),OLD_NAMESPACE.encode())==old
        else:assert actual==old,'CUDA or codec changed'
        records[role]={'path':str(path(role).relative_to(ROOT)),'sha256':sha(actual),'parent_sha256':sha(old),'unchanged':actual==old}
    import json
    edits=json.loads((ROOT/'tools/biv150_native_k8_mixed_host_edits.json').read_bytes())
    restored={}
    for row in reversed(edits):
        b=restored.setdefault(row['path'],(ROOT/row['path']).read_bytes())
        new,old=row['new'].encode(),row['old'].encode()
        assert b.count(new)==1,'missing exact host edit: '+row['path']
        restored[row['path']]=b.replace(new,old)
    for name,b in restored.items():
        assert b==subprocess.check_output(['git','-C',str(ROOT),'show',BASE+':'+name]),'unreviewed host change: '+name
    return {'base':BASE,'namespace':NAMESPACE,'host_reverse_byte_exact':sorted(restored),'files':records,'new_rows_segments':[[29,4],[30,4],[31,4],[60,2],[61,2],[63,2]],
            'existing_rows_segments':[[32,4],[64,2]],'CUDA_and_codec_bytes_unchanged':True,
            'no_new_valid_rows_or_padding':True,'physical_page_bytes':16384,
            'actual_SO_verified':False,'numeric_verified':False,'performance_verified':False,'service_verified':False}

if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');a=p.parse_args()
    if a.write:path('qk.cpp').write_bytes(generate())
    print(json.dumps(verify(),indent=2,sort_keys=True))
