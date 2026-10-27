"""Execute the actual pinned get_sql_model body against an opaque DB adapter."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import types

import flaggems_sql_model_pool_fix as fix


class Engine:
    def __init__(self, url='sqlite:///private.db'):
        self.url = url
        self.tables = {}
        self.reflections = 0
        self.creates = 0
        self.enters = self.exits = 0
        self.error = None

    def begin(self):
        return self

    def __enter__(self):
        self.enters += 1
        return self

    def __exit__(self, *args):
        self.exits += 1

    def execute(self, value):
        table, options = value
        assert options == {'if_not_exists': True}
        self.creates += 1
        if self.error:
            raise self.error
        self.tables[table.name] = table.model


def fixture(source, filename, url='sqlite:///private.db'):
    original, _ = fix.prepare(source, filename)
    module = types.ModuleType('flag_gems.utils.models.sql')
    module.__file__ = filename
    cls = type('SQLPersistantModel', (), {'__module__': module.__name__})
    module.SQLPersistantModel = cls
    module.md5 = hashlib.md5
    module.sqlalchemy = types.SimpleNamespace(schema=types.SimpleNamespace(
        CreateTable=lambda table, **kw: (table, kw)))

    def reflect(name, engine):
        engine.reflections += 1
        return engine.tables.get(name)

    def build(name, keys, values):
        assert keys and values
        model = type('OpaqueSQLClass', (), {})
        model.__table__ = types.SimpleNamespace(name=name, model=model)
        return model

    cls.build_sql_model_by_db = staticmethod(reflect)
    cls.build_sql_model_by_py = staticmethod(build)
    method = types.FunctionType(original, vars(module), 'get_sql_model', ({}, {}))
    method.__qualname__ = 'SQLPersistantModel.get_sql_model'
    method.__module__ = module.__name__
    method.__dict__['__override__'] = True
    cls.get_sql_model = method
    instance = cls()
    instance.engine = Engine(url)
    instance.lock = threading.Lock()
    instance.sql_model_pool = {}
    return module, instance, method


def refused(fn):
    try:
        fn()
    except RuntimeError:
        return True
    raise AssertionError('Expected rejection')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_bytes()
    filename = str(args.source.resolve())
    assert fix.install() == dict(enabled=False, installed=False, database_access=False)
    reductions = []
    for first_get in (False, True):
        for url in ('sqlite:///private.db', 'postgresql://opaque/readonly-test'):
            seen = []
            for enabled in (False, True):
                module, obj, original = fixture(source, filename, url)
                receipt = fix.install(module, enabled=enabled)
                name = 'kernel_' + 'x' * 110
                keys, values = {'key_0': 1, 'key_1': 'bf16'}, {'BLOCK_N': 128}
                if first_get:
                    assert obj.get_sql_model(name, keys) is None
                created = obj.get_sql_model(name, keys, values)
                assert created is obj.get_sql_model(name, keys)
                assert obj.engine.creates == obj.engine.enters == obj.engine.exits == 1
                assert obj.get_sql_model(name, keys) is created
                if enabled:
                    assert len(obj.sql_model_pool) == 1
                    assert list(obj.sql_model_pool) == [created.__table__.name]
                    assert fix.install(module, enabled=True)['reused']
                    assert module.SQLPersistantModel.get_sql_model.__dict__ == original.__dict__
                    assert module.SQLPersistantModel.get_sql_model.__defaults__ is original.__defaults__
                    assert fix.uninstall(module)['restored']
                    assert module.SQLPersistantModel.get_sql_model is original
                seen.append(obj.engine.reflections)
            assert seen == ([3, 2] if first_get else [2, 1]), seen
            reductions.append(dict(first_get=first_get, backend=url.split(':')[0],
                                   original_reflections=seen[0], fixed_reflections=seen[1]))

    for enabled in (False, True):
        module, obj, original = fixture(source, filename)
        if enabled:
            fix.install(module, enabled=True)
        a = obj.get_sql_model('same', {'a': 1}, {'BLOCK_N': 1})
        b = obj.get_sql_model('same', {'b': 1}, {'BLOCK_N': 1})
        assert a is not b
        assert obj.get_sql_model('same', {'a': 1}) is a
        assert obj.get_sql_model('same', {'b': 1}) is b
        # A failed DDL must retain the same exception and leave no cached model.
        err = ValueError('real adapter failure sentinel')
        obj.engine.error = err
        before = dict(obj.sql_model_pool)
        try:
            obj.get_sql_model('failed', {'key': 1}, {'BLOCK_N': 1})
        except ValueError as caught:
            assert caught is err and obj.sql_model_pool == before
        else:
            raise AssertionError('DDL failure swallowed')
        assert obj.engine.enters == obj.engine.exits
        if enabled:
            fix.uninstall(module)

    module, obj, original = fixture(source, filename)
    fix.install(module, enabled=True)
    results = []
    def call():
        results.append(obj.get_sql_model('threads', {'key': 1}, {'BLOCK_N': 1}))
    workers = [threading.Thread(target=call) for _ in range(12)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert len(results) == 12 and all(v is results[0] for v in results)
    assert obj.engine.reflections == obj.engine.creates == 1
    fix.uninstall(module)

    refused(lambda: fix.prepare(source + b'\n', filename))
    module, obj, original = fixture(source, filename)
    module.SQLPersistantModel.get_sql_model = lambda *a: None
    refused(lambda: fix.install(module, enabled=True))
    module, obj, original = fixture(source, filename)
    fix.install(module, enabled=True)
    sentinel = lambda *a: None
    module.SQLPersistantModel.get_sql_model = sentinel
    refused(lambda: fix.uninstall(module))
    assert module.SQLPersistantModel.get_sql_model is sentinel
    # Restore our own test binding only; never overwrite foreign runtime state.
    module.SQLPersistantModel.get_sql_model = fix._INSTALLED[module.SQLPersistantModel]['patched']
    fix.uninstall(module)
    assert not fix._INSTALLED
    assert not any(n in __import__('sys').modules for n in ('torch', 'triton', 'flag_gems', 'sqlalchemy'))
    print(json.dumps(dict(passed=True, CPU_only=True,
          actual_source_method_executed=True, source_sha256=fix.SOURCE_SHA,
          opaque_adapter_not_real_SQLAlchemy=True, source_database_modified=False,
          reductions=reductions, two_schemas=True, DDL_error_identity=True,
          defaults_and_override_preserved=True, threads=12,
          existing_other_method_change_not_overwritten=True,
          framework_imported=False, production_installed=False,
          target_runtime_verified=False, performance_verified=False,
          kernel_modified=False, precision_gate=False), sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
