"""Opt-in prototype: cache newly created SQL models under their lookup key.

Does not import FlagGems/SQLAlchemy, connect to a DB, or install automatically.
The original source and loaded method must match before the single AST edit.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import types

SOURCE_SHA = '6dc19cb581d70154f14b95ed9112d682c6b5919b6cf3f8dea01a78a75ec0efd7'
_INSTALLED = {}


def sha(value):
    return hashlib.sha256(value).hexdigest()


def need(value, message):
    if not value:
        raise RuntimeError('SQL model pool prototype: ' + message)


def code_facts(code):
    """All CodeType fields, without marshal's reference/interning encoding."""
    def constant(value):
        if isinstance(value, types.CodeType):
            return ['code', code_facts(value)]
        if value is None or value is Ellipsis:
            return ['singleton', repr(value)]
        if type(value) in (bool, int, str):
            return [type(value).__name__, value]
        if type(value) is bytes:
            return ['bytes', value.hex()]
        if type(value) is float:
            return ['float64', struct.pack('>d', value).hex()]
        if type(value) is complex:
            return ['complex128', struct.pack('>dd', value.real, value.imag).hex()]
        if type(value) is tuple:
            return ['tuple', [constant(v) for v in value]]
        if type(value) is frozenset:
            return ['frozenset', sorted([constant(v) for v in value], key=repr)]
        raise RuntimeError('unsupported code constant: ' + type(value).__name__)
    fields = ('co_argcount', 'co_posonlyargcount', 'co_kwonlyargcount',
              'co_nlocals', 'co_stacksize', 'co_flags', 'co_code', 'co_consts',
              'co_names', 'co_varnames', 'co_filename', 'co_name', 'co_qualname',
              'co_firstlineno', 'co_linetable', 'co_exceptiontable',
              'co_freevars', 'co_cellvars')
    return {name: constant(getattr(code, name)) for name in fields}


def code_sha(code):
    return sha(json.dumps(code_facts(code), sort_keys=True,
                          separators=(',', ':')).encode())


def method_code(module_code):
    cls = [v for v in module_code.co_consts
           if isinstance(v, types.CodeType) and v.co_name == 'SQLPersistantModel']
    need(len(cls) == 1, 'one original model class')
    methods = [v for v in cls[0].co_consts
               if isinstance(v, types.CodeType) and v.co_name == 'get_sql_model']
    need(len(methods) == 1, 'one original method')
    return methods[0]


def prepare(source, filename):
    need(sha(source) == SOURCE_SHA, 'reviewed original source SHA')
    before = ast.parse(source, filename=filename)
    after = copy.deepcopy(before)
    cls = next(n for n in after.body
               if isinstance(n, ast.ClassDef) and n.name == 'SQLPersistantModel')
    fn = next(n for n in cls.body
              if isinstance(n, ast.FunctionDef) and n.name == 'get_sql_model')
    edits = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Attribute)
                and isinstance(target.value.value, ast.Name)
                and target.value.value.id == 'self'
                and target.value.attr == 'sql_model_pool'
                and isinstance(target.slice, ast.Name)
                and target.slice.id == 'name'):
            need(isinstance(node.value, ast.Name) and node.value.id == 'ModelCls',
                 'exact original insertion')
            edits.append(target)
    need(len(edits) == 1, 'exactly one fresh-model insertion')
    edits[0].slice.id = 'short_name'
    inverse = copy.deepcopy(after)
    invcls = next(n for n in inverse.body
                  if isinstance(n, ast.ClassDef) and n.name == 'SQLPersistantModel')
    invfn = next(n for n in invcls.body
                 if isinstance(n, ast.FunctionDef) and n.name == 'get_sql_model')
    # Match the same source location rather than changing other short_name uses.
    original_location = (edits[0].lineno, edits[0].col_offset)
    invtargets = [n for n in ast.walk(invfn) if isinstance(n, ast.Subscript)
                  and (n.lineno, n.col_offset) == original_location]
    need(len(invtargets) == 1, 'one inverse target')
    invtargets[0].slice.id = 'name'
    need(ast.dump(inverse, include_attributes=True)
         == ast.dump(before, include_attributes=True), 'whole-source AST inverse')
    original = method_code(compile(before, filename, 'exec', dont_inherit=True))
    patched = method_code(compile(after, filename, 'exec', dont_inherit=True))
    need(original.co_freevars == patched.co_freevars
         and original.co_argcount == patched.co_argcount
         and original.co_kwonlyargcount == patched.co_kwonlyargcount,
         'method call/closure ABI')
    return original, patched


def install(module=None, *, enabled=False):
    if not enabled:
        return dict(enabled=False, installed=False, database_access=False)
    need(isinstance(module, types.ModuleType), 'explicit already-loaded module')
    need(module.__name__ == 'flag_gems.utils.models.sql', 'actual module name')
    cls = module.SQLPersistantModel
    state = _INSTALLED.get(cls)
    if state is not None:
        need(vars(cls).get('get_sql_model') is state['patched'],
             'installed method binding changed')
        return dict(state['receipt'], reused=True)
    original = vars(cls).get('get_sql_model')
    need(isinstance(original, types.FunctionType), 'plain original method')
    path = Path(module.__file__).resolve()
    source = path.read_bytes()
    code, fixed = prepare(source, str(path))
    need(code_facts(original.__code__) == code_facts(code),
         'loaded complete method CodeType')
    need(original.__globals__ is vars(module)
         and original.__globals__.get('SQLPersistantModel') is cls,
         'original globals/class binding')
    patched = types.FunctionType(fixed, original.__globals__, original.__name__,
                                 original.__defaults__, original.__closure__)
    patched.__kwdefaults__ = copy.copy(original.__kwdefaults__)
    patched.__annotations__ = dict(original.__annotations__)
    patched.__dict__.update(original.__dict__)
    patched.__module__ = original.__module__
    patched.__qualname__ = original.__qualname__
    patched.__doc__ = original.__doc__
    receipt = dict(enabled=True, installed=True, source_sha256=SOURCE_SHA,
                   original_method_sha256=code_sha(code),
                   patched_method_sha256=code_sha(fixed),
                   python=sys.version.split()[0], database_access=False,
                   change='fresh model pool[name] -> pool[short_name]',
                   source_file_modified=False, kernel_modified=False,
                   performance_verified=False, service_verified=False,
                   precision_gate=False)
    cls.get_sql_model = patched
    _INSTALLED[cls] = dict(original=original, patched=patched, receipt=receipt)
    return dict(receipt, reused=False)


def uninstall(module):
    cls = module.SQLPersistantModel
    state = _INSTALLED.get(cls)
    if state is None:
        return dict(restored=False)
    need(vars(cls).get('get_sql_model') is state['patched'],
         'refuse to overwrite another method change')
    cls.get_sql_model = state['original']
    del _INSTALLED[cls]
    return dict(restored=True, database_access=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect-source', type=Path)
    args = parser.parse_args()
    result = dict(CPU_only=True, default_off=True, installed=False,
                  automatic_worker_hook=False, source_sha256=SOURCE_SHA,
                  database_access=False, GPU_access=False,
                  performance_verified=False, service_verified=False)
    if args.inspect_source:
        original, patched = prepare(args.inspect_source.read_bytes(),
                                    str(args.inspect_source.resolve()))
        result.update(source_inverse_verified=True,
                      original_method_sha256=code_sha(original),
                      patched_method_sha256=code_sha(patched))
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
