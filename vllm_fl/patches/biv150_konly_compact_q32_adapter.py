"""Compact-only lifecycle admission around the unchanged actual-JIT hoist adapter.

No framework import/device work. The proof concerns immutable QCode, not cached
K contents. Every new call retains the current compact typed-role/owner epoch.
"""
from .biv150_konly_encoded_q32_adapter import Adapter as EncodedAdapter
import hashlib
from pathlib import Path
import sys
import types
import warnings

FORMAT = 'cjj-konly-compact12544-rn32-rne-v1'


def need(ok, message):
    if not ok: raise ValueError('Compact Q32 hoist: ' + message)


class Adapter(EncodedAdapter):
    def __init__(self, *args, source_module, source_sha256, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_binding = guard_source(self.fn, source_module, source_sha256)
        need(self.proposal.FORMAT == FORMAT, 'wrong proposal cache format')
        self.compact_epochs = {}

    def validate_call(self, runtime, entry, launch):
        # Invalid cache addresses cannot even retain unhoisted execution: this
        # check is OUTSIDE prepare's prelaunch compiler/proposal fallback catch.
        module, kernel_fn, kernel_code, helper, helper_fn, helper_code, codec, codec_fn, codec_code = self.source_binding
        need(module.kernel_cjj_konly_q32 is self.fn and self.fn.fn is kernel_fn
             and kernel_fn.__code__ is kernel_code and kernel_fn.__globals__ is vars(module)
             and module._cjj_konly_scale is helper and helper.fn is helper_fn
             and helper_fn.__code__ is helper_code and helper_fn.__globals__ is vars(module)
             and module.quantize_bf16_rne is codec and codec.fn is codec_fn
             and codec_fn.__code__ is codec_code and codec_fn.__globals__ is vars(module),
             'private kernel/scale helper binding changed')
        registry, broker = runtime.registry, runtime.registry.broker
        need(registry.armed, 'unarmed encoded lifetime')
        need(type(entry.generation) is int and 0 < entry.generation <= registry.generation,
             'unknown entry generation')
        current = registry.observe(entry.owner)
        need(current == entry.observation, 'cache owner changed/reallocated')
        identity = (current.storage_id, current.storage_offset, current.pointer,
                    current.shape, current.strides)
        need(registry.entries.get(identity) is entry and current.storage_id in registry.owned_storages,
             'entry not owned by this current registry epoch')
        roles = entry.roles['prefill80']
        supplied = {}
        for key, value in roles.items():
            need(key in launch, 'missing typed role ' + key)
            actual = launch[key]
            if hasattr(value, 'untyped_storage'):
                need(actual is value, 'role Tensor identity ' + key)
                supplied[key] = registry.observe(actual)
            else:
                need(type(actual) is type(value) and actual == value, 'role scalar ' + key)
                supplied[key] = actual
        broker.validate_binding(current, 'prefill80', supplied, format_id=FORMAT)
        token = (id(registry), id(entry.owner), entry.generation, current,
                 tuple((k,id(v)) for k,v in roles.items() if hasattr(v,'untyped_storage')))
        previous = self.compact_epochs.get(id(entry))
        need(previous is None or previous == token, 'entry lifecycle/role binding changed')
        self.compact_epochs[id(entry)] = token

    def stats(self):
        return dict(super().stats(), cache_format=FORMAT,
                    compact_owner_epochs=len(self.compact_epochs),
                    cache_contents_or_epoch_cached_as_compiler_proof=False)


def code_at(code, *path):
    for name in path:
        found = [c for c in code.co_consts if isinstance(c, types.CodeType) and c.co_name == name]
        need(len(found) == 1, 'unique full-module code object ' + name)
        code = found[0]
    return code


def code_field(code, name):
    # Compare the legacy field too; suppress only its attribute-access warning.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        return getattr(code, name)


def same_code(actual, expected):
    fields = [name for name in dir(expected) if name.startswith('co_')
              and not callable(code_field(expected, name))]
    for name in fields:
        left, right = code_field(actual, name), code_field(expected, name)
        if name == 'co_consts':
            need(len(left) == len(right), 'code constant count')
            for a, b in zip(left, right):
                if isinstance(b, types.CodeType):
                    need(isinstance(a, types.CodeType), 'nested code kind')
                    same_code(a, b)
                else: need(type(a) is type(b) and a == b, 'code constant')
        else: need(type(left) is type(right) and left == right, 'code field ' + name)


def guard_source(fn, module, source_sha256):
    path = Path(module.__file__)
    raw = path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == source_sha256,
         'compact Q32 whole-module source mismatch')
    expected = compile(raw, str(path), 'exec', dont_inherit=True, optimize=sys.flags.optimize)
    helper = module._cjj_konly_scale
    codec = module.quantize_bf16_rne
    for name, jit in [('kernel_cjj_konly_q32', fn), ('_cjj_konly_scale', helper),
                      ('quantize_bf16_rne', codec)]:
        actual = jit.fn
        need(isinstance(actual, types.FunctionType) and actual.__globals__ is vars(module)
             and not actual.__closure__, 'private source globals/closure ' + name)
        same_code(actual.__code__, code_at(expected, name))
    return module, fn.fn, fn.fn.__code__, helper, helper.fn, helper.fn.__code__, codec, codec.fn, codec.fn.__code__
