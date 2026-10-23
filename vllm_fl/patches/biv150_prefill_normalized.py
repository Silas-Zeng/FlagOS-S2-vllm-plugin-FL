"""Audited in-memory prefill normalization clone; serving registration is separate."""
import functools
import hashlib
import importlib
import inspect
import linecache
from pathlib import Path
import types

_AUDITED={
 'vllm.v1.attention.ops.triton_unified_attention':'54c495dcea404582a1526a96ae7e472965cbd03c8fa0931c71ef72223bf04e1f',
 'vllm.v1.attention.ops.triton_attention_helpers':'59d9d34bbda988436593cfd9bcb05015d17db8ee85c50f6697e45f987acbf0b1',
}


def make_normalized(attention):
    audited={}
    for name,expected in _AUDITED.items():
        module=importlib.import_module(name)
        actual=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        if actual!=expected:raise RuntimeError('Unaudited attention source: '+name)
        audited[name]=actual
    kernel=attention.kernel_unified_attention
    if not hasattr(kernel,'fn'):raise RuntimeError('Unrecognized prefill kernel wrapper')
    source=inspect.getsource(kernel.fn)
    old='L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)'
    if source.count(old)!=1:raise RuntimeError('Unexpected cumulative softmax initialization')
    source=source.replace(old,'L = tl.zeros([BLOCK_M], dtype=tl.float32)')
    source=source.replace('def kernel_unified_attention(','def kernel_biv150_prefill_normalized(',1)
    digest=hashlib.sha256(source.encode()).hexdigest()
    filename='<biv150-prefill-normalized-'+digest+'.py>'
    # Standard inspect/linecache support lets Triton read the exact pinned clone
    # without writing generated source into the installed package or checkout.
    linecache.cache[filename]=(len(source),None,source.splitlines(keepends=True),filename)
    namespace=dict(vars(attention))
    namespace['__name__']=__name__
    exec(compile(source,filename,'exec'),namespace)
    normalized_kernel=namespace['kernel_biv150_prefill_normalized']
    raw=inspect.unwrap(attention.unified_attention)
    if raw.__name__!='unified_attention':raise RuntimeError('Unexpected attention wrapper')
    globals_copy=dict(raw.__globals__)
    globals_copy['kernel_unified_attention']=normalized_kernel
    function=types.FunctionType(raw.__code__,globals_copy,raw.__name__,raw.__defaults__,raw.__closure__)
    function.__kwdefaults__=raw.__kwdefaults__
    functools.update_wrapper(function,raw)
    return function,dict(audited=audited,kernel_sha256=digest,initial_mass=0)
