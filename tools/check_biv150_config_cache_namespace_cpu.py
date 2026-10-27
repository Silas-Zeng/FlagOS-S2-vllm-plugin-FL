#!/usr/bin/env python3
"""Actual full-module CodeTypes; fake runtime/model only, no framework imports."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types
sys.path.insert(0, str(Path(__file__).resolve().parent / 'diagnostics/flaggems_tuner'))
from cjj_flaggems_tuner_diag import code_at

LIB_SHA = '414e8a3f74e2522e40659e7fccd84e301f4517220837a87af0311f050bbf3025'
SQL_SHA = '6dc19cb581d70154f14b95ed9112d682c6b5919b6cf3f8dea01a78a75ec0efd7'


def check(root):
    def load(relative, sha):
        p = root / relative
        raw = p.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == sha
        return compile(raw, str(p), 'exec', dont_inherit=True, optimize=sys.flags.optimize)
    lc = load('flag_gems/utils/libentry.py', LIB_SHA)
    sc = load('flag_gems/utils/models/sql.py', SQL_SHA)
    make_name = types.FunctionType(code_at(lc, 'LibTuner', '_make_config_table_name'),
                                 dict(hashlib=hashlib))
    outputs = []
    # These are explicit test protocols, not evidence of a live target protocol.
    for protocol in [('triton_do_bench', 25, 100), ('custom_do_bench', -1, -1),
                     ('triton_cuda_graph_replay_v1', 25, 100, 10, 10.0)]:
        tuner = types.SimpleNamespace(__name__='namespace_probe', kernel_hash='kernel_identity',
                                      _benchmark_protocol=protocol)
        name = make_name(tuner)
        calls = []
        get = types.FunctionType(code_at(lc, 'ConfigCache', 'get'), {})
        put = types.FunctionType(code_at(lc, 'ConfigCache', 'set'), {})
        model = types.SimpleNamespace(
            get_config=lambda *a: calls.append(('get_config', a)) or 'known-config',
            put_config=lambda *a: calls.append(('put_config', a)))
        cache = types.SimpleNamespace(table_name=name, model=model)
        assert get(cache, ('shape',)) == 'known-config'
        put(cache, ('shape',), 'config-value')
        assert calls == [('get_config', (name, ('shape',))),
                         ('put_config', (name, ('shape',), 'config-value'))]
        # Execute actual SQL get_sql_model naming path, stop at read-only fake
        # reflection; no SQLAlchemy/database/framework execution occurs.
        import threading
        class SQL:
            def __init__(self):
                self.lock = threading.Lock()
                self.engine = types.SimpleNamespace(url='sqlite:///fake')
                self.sql_model_pool = {}
                self.reflected = []
            def build_sql_model_by_db(self, physical, engine):
                self.reflected.append(physical)
                assert engine is self.engine
                return None
        fn = types.FunctionType(code_at(sc, 'SQLPersistantModel', 'get_sql_model'),
                                dict(md5=hashlib.md5))
        sql = SQL()
        fn.__globals__['SQLPersistantModel'] = sql
        assert fn(sql, name, {'key_0': 'shape'}, {}) is None
        physical = name + '-' + hashlib.md5(b'key_0').hexdigest()
        assert sql.reflected == [physical]
        outputs.append(dict(test_protocol=list(protocol), config_logical_name=name,
                            actual_ConfigCache_get_put_names=[x[1][0] for x in calls],
                            actual_SQL_physical_table_name=physical,
                            old_benchmark_substring_filter_would_exclude='benchmark' in physical))
    assert not outputs[0]['old_benchmark_substring_filter_would_exclude']
    assert all(x['old_benchmark_substring_filter_would_exclude'] for x in outputs[1:])
    assert not {'torch', 'triton', 'sqlalchemy', 'flag_gems'}.intersection(sys.modules)
    return dict(libentry_sha256=LIB_SHA, sql_sha256=SQL_SHA, full_module_code_extraction=True,
                explicit_fake_protocols=outputs, current_v6_Config_with_benchmark_name_found=False,
                live_after_score_database_available=False, target_resolved_protocol_proven=False,
                claim='constructible source path only; not an observed v6 or after-score Config mutation',
                no_framework_imports=True, formal_score_eligible=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(check(a.source_root), sort_keys=True, indent=2))
