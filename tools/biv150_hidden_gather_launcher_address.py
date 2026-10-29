"""Readonly PyCFunction address ownership; module.__file__ is diagnostic only."""
import base64,ctypes,hashlib,importlib,inspect,json,types
from pathlib import Path
DRIVER_SHA='5023c317de1c2f3370d254320849a2013f68b353fd35884c60d857823c424f50'
BUILD_SHA='d6d0c78e98a1fba75923337af6e2b45df17033c90295303941b2a47d4cfca00c'
C_SOURCE_SHA='16b5948ce5a3d75dd077147571f2c544cc6d99e6c6890c988652ef26bb2c6f0f'
HOST_SHA='512414b6fbfae2135b62153619ef1f5919bd2fd07018058cd45035de1d83e361'
HOST_DIRECTORY='H7POXA2W3X5GQEOXOQQOM2VCB5S6HLHVZLJF3SAXLAJPDJFQOIOA'

def sha(b):return hashlib.sha256(b).hexdigest()
def need(v,why):
 if not v:raise ValueError('Gather launcher address: '+why)

def builtin_binding(function):
 """Do not call the builtin. Ask CPython for its C pointer, then the loader."""
 need(type(function) is types.BuiltinFunctionType,'exact CPython builtin required')
 get=ctypes.pythonapi.PyCFunction_GetFunction;get.argtypes=[ctypes.py_object];get.restype=ctypes.c_void_p
 flags=ctypes.pythonapi.PyCFunction_GetFlags;flags.argtypes=[ctypes.py_object];flags.restype=ctypes.c_int
 address=get(function);need(address is not None,'missing builtin C pointer')
 class Info(ctypes.Structure):
  _fields_=[('fname',ctypes.c_char_p),('base',ctypes.c_void_p),('sname',ctypes.c_char_p),('saddr',ctypes.c_void_p)]
 dl=ctypes.CDLL(None).dladdr;dl.argtypes=[ctypes.c_void_p,ctypes.POINTER(Info)];dl.restype=ctypes.c_int
 info=Info();need(dl(address,ctypes.byref(info))==1 and info.fname and info.base,'dladdr failed')
 path=Path(info.fname.decode()).resolve();raw=path.read_bytes();need(info.base<=address,'invalid DSO address range')
 owner=getattr(function,'__self__',None);legacy=getattr(owner,'__file__',None);legacy_sha=None
 if isinstance(legacy,str):
  try:legacy_sha=sha(Path(legacy).read_bytes())
  except OSError:pass
 return dict(method_name=function.__name__,method_flags=flags(function),C_address=address,DSO_base=info.base,DSO_offset=address-info.base,DSO_path=str(path),DSO_sha256=sha(raw),DSO_symbol=None if info.sname is None else info.sname.decode(),legacy_module_path=legacy,legacy_module_sha256=legacy_sha,legacy_module_path_is_not_function_ownership=True)

def verify(kernel,record,group,folder):
 """Record first; enforce original ABI-generated cache and actual C-address SHA."""
 launch=getattr(kernel._run,'launch',None);binding=builtin_binding(launch)
 folder.mkdir(parents=True,exist_ok=True);(folder/'launcher-binding.json').write_text(json.dumps(binding,sort_keys=True,indent=2)+'\n')
 # Keep actual bytes even if the following scope/identity check rejects them.
 binary=Path(binding['DSO_path']).read_bytes();(folder/'actual-launcher.so').write_bytes(binary)
 need(binding['method_name']=='launch' and binding['method_flags']==1,'exact METH_VARARGS launch')
 need(binary[:4]==b'\x7fELF' and binding['DSO_sha256']==HOST_SHA,'actual C function must belong to reviewed5124 ELF')
 need(Path(binding['DSO_path'])==Path(group['host'][0]['path']).resolve(),'actual C function must belong to private reviewed host file')
 driver=importlib.import_module('triton.backends.iluvatar.driver');build=importlib.import_module('triton.runtime.build')
 need(sha(Path(driver.__file__).read_bytes())==DRIVER_SHA and sha(Path(build.__file__).read_bytes())==BUILD_SHA,'actual SDK generator/build source')
 need(type(kernel._run) is driver.CorexLauncher and inspect.getfile(type(kernel._run))==driver.__file__,'actual SDK launcher type')
 src=kernel.src;idx=lambda k:(src.fn.arg_names.index(k),) if isinstance(k,str) else k
 constants={idx(k):v for k,v in src.constants.items()};signature=dict(src.signature)
 c_source=driver.make_launcher(constants,signature,getattr(kernel.metadata,'tensordesc_meta',None))
 platform=build.platform_key();key=sha((c_source+platform).encode());directory=base64.b32encode(bytes.fromhex(key)).decode().rstrip('=')
 binding.update(generator_driver_sha256=DRIVER_SHA,generator_build_sha256=BUILD_SHA,generated_C_sha256=sha(c_source.encode()),actual_platform_key=platform,generated_cache_directory=directory)
 (folder/'launcher-binding.json').write_text(json.dumps(binding,sort_keys=True,indent=2)+'\n');(folder/'actual-launcher-generated.c').write_text(c_source)
 need(binding['generated_C_sha256']==C_SOURCE_SHA and directory==HOST_DIRECTORY,'actual ABI must generate original H7/5124 launcher')
 need(record['hash']==str(kernel.hash),'already checked GPU object identity')
 return binding
