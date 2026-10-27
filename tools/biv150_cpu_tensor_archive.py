"""Restricted CPU tensor-storage reader for saved evidence; never imports Torch.

Only this evidence's CPU storage tags and tensor rebuild are admitted. No
arbitrary pickle globals, CUDA storages or callable deserialization are allowed.
"""
from collections import OrderedDict
import gzip
import io
import pickle
import re
import zipfile

DTYPES={'BFloat16Storage':('u2',2),'ByteStorage':('u1',1),'CharStorage':('i1',1),
        'DoubleStorage':('f8',8),'FloatStorage':('f4',4),'IntStorage':('i4',4),'LongStorage':('i8',8)}


class Storage:
    def __init__(self,name,data,count): self.name,self.data,self.count=name,data,count


class Tensor:
    def __init__(self,storage,offset,shape,stride):
        if type(offset) is not int or offset<0 or type(shape) is not tuple or type(stride) is not tuple or len(shape)!=len(stride):
            raise ValueError('Unknown CPU tensor descriptor')
        if any(type(n) is not int or n<0 for n in (*shape,*stride)): raise ValueError('Unknown CPU shape/stride')
        end=offset+sum((n-1)*s for n,s in zip(shape,stride))+1
        if end>storage.count: raise ValueError('CPU tensor exceeds stored bytes')
        self.storage,self.offset,self.shape,self.stride=storage,offset,shape,stride
    def array(self,np):
        dtype,itemsize=DTYPES[self.storage.name]
        return np.ndarray(self.shape,dtype='<'+dtype,buffer=self.storage.data,offset=self.offset*itemsize,
                          strides=tuple(s*itemsize for s in self.stride))
    def values(self,np):
        array=self.array(np)
        if self.storage.name=='BFloat16Storage': return (array.astype(np.uint32)<<16).view(np.float32).astype(np.float64)
        return array.astype(np.float64)


def rebuild(storage,offset,shape,stride,requires_grad,hooks,*extra):
    if type(storage) is not Storage or requires_grad is not False or hooks or extra:
        raise ValueError('Unknown CPU tensor rebuild')
    return Tensor(storage,offset,shape,stride)


def read_tensor_archive(blob):
    raw=gzip.decompress(blob)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        keys=archive.namelist()
        if len(keys)!=len(set(keys)) or any(k.startswith('/') or '..' in k.split('/') for k in keys):
            raise ValueError('Unsafe CPU tensor ZIP')
        pickle_keys=[k for k in keys if k.endswith('/data.pkl')]
        if len(pickle_keys)!=1: raise ValueError('Unknown tensor archive prefix')
        prefix=pickle_keys[0][:-len('data.pkl')]
        if archive.read(prefix+'byteorder')!=b'little': raise ValueError('Only witnessed little-endian CPU evidence')
        stores={}
        class Reader(pickle.Unpickler):
            def find_class(self,module,name):
                if module=='torch' and name in DTYPES: return name
                if (module,name)==('torch._utils','_rebuild_tensor_v2'): return rebuild
                if (module,name)==('collections','OrderedDict'): return OrderedDict
                raise ValueError('Unapproved pickle global: '+module+'.'+name)
            def persistent_load(self,value):
                if type(value) is not tuple or len(value)!=5: raise ValueError('Unknown persistent tensor storage')
                tag,name,key,location,count=value
                if tag!='storage' or name not in DTYPES or location!='cpu' or type(count) is not int or count<0 or not re.fullmatch(r'[0-9]+',key):
                    raise ValueError('Only known original CPU storage is admitted')
                if key not in stores:
                    data=archive.read(prefix+'data/'+key)
                    if len(data)!=count*DTYPES[name][1]: raise ValueError('Storage byte count differs')
                    stores[key]=Storage(name,data,count)
                result=stores[key]
                if result.name!=name or result.count!=count: raise ValueError('Storage identity changed')
                return result
        return Reader(io.BytesIO(archive.read(pickle_keys[0]))).load()
