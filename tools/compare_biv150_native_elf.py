"""CPU-only host/native-device ELF section comparison; preserves raw libraries."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def sections(data,start=0):
    data=data[start:]
    assert data[:4]==b'\x7fELF' and data[5]==1,'Expected little-endian ELF'
    if data[4]==2:
        offset=struct.unpack_from('<Q',data,40)[0];size,count,names=struct.unpack_from('<HHH',data,58);fmt='<IIQQQQIIQQ'
    elif data[4]==1:
        offset=struct.unpack_from('<I',data,32)[0];size,count,names=struct.unpack_from('<HHH',data,46);fmt='<IIIIIIIIII'
    else:raise ValueError('Unknown ELF class')
    assert size>=struct.calcsize(fmt) and offset+count*size<=len(data)
    rows=[struct.unpack_from(fmt,data,offset+i*size) for i in range(count)];assert names<count
    table=rows[names];strings=data[table[4]:table[4]+table[5]];result={}
    for row in rows:
        name=strings[row[0]:].split(b'\0',1)[0].decode()
        payload=data[row[4]:row[4]+row[5]] if row[1]!=8 else None
        if payload is not None:assert len(payload)==row[5]
        result[name]={'flags':row[2],'bytes':row[5],'sha256':hashlib.sha256(payload).hexdigest() if payload is not None else None,'offset':row[4]}
    return result


def inspect(path):
    raw=path.read_bytes();host=sections(raw);fat=host['.nv_fatbin'];payload=raw[fat['offset']:fat['offset']+fat['bytes']]
    start=payload.find(b'\x7fELF');assert start>=0
    return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'host':host,'embedded_device':sections(payload,start)}


def main(args):
    left,right=inspect(args.left),inspect(args.right)
    result={'left':left,'right':right,'limitation':'Compares ELF sections only; bytes outside the first embedded device ELF and runtime/JIT behavior require separate evidence.'}
    for scope in ['host','embedded_device']:
        result[scope+'_differences']=[name for name in left[scope].keys()|right[scope].keys()
            if {k:v for k,v in left[scope].get(name,{}).items() if k!='offset'}!={k:v for k,v in right[scope].get(name,{}).items() if k!='offset'}]
    args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k.endswith('_differences')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--left',type=Path,required=True);p.add_argument('--right',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
