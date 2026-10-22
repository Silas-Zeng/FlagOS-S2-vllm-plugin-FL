"""Reflect one FlagGems SQLite configuration table per requested model.

No database/config writes, persistent or negative cache, tuning changes, or
device work. Install on the isolated Iluvatar plugin source before inference.
Foreign-key dependency reflection keeps SQLAlchemy's original default.
"""

import functools
import hashlib
import importlib
from pathlib import Path

import sqlalchemy.ext.automap


EXPECTED_SQL_SHA256 = "6dc19cb581d70154f14b95ed9112d682c6b5919b6cf3f8dea01a78a75ec0efd7"
_MARKER = "_biv150_sql_targeted_reflection"


def _targeted_builder(original):
    @functools.wraps(original)
    def build_sql_model_by_db(name, engine):
        if engine.dialect.name != "sqlite":
            return original(name, engine)
        # A fresh base preserves the original per-call database/schema
        # visibility and cannot mix models between engine instances.
        base = sqlalchemy.ext.automap.automap_base()
        base.prepare(
            engine,
            reflection_options={"only": lambda table_name, metadata: table_name == name},
        )
        # A predicate filters available tables. Unlike only=[name], a missing
        # name remains None, allowing the original put_config creation path.
        return base.classes.get(name)

    setattr(build_sql_model_by_db, _MARKER, True)
    return build_sql_model_by_db


def install(*, module=None, vendor_name=None):
    """Replace only the audited static reflection implementation, fail closed.

    The optional dependency arguments make the real SQLAlchemy CPU gate
    independent of importing FlagGems/vLLM and initializing their platforms.
    Runtime callers use install() without overrides.
    """
    if vendor_name is None:
        vendor_name = importlib.import_module("vllm.platforms").current_platform.vendor_name
    if vendor_name != "iluvatar":
        return {"installed": False, "reason": "non-iluvatar platform uses original reflection"}
    if module is None:
        module = importlib.import_module("flag_gems.utils.models.sql")
    source = Path(module.__file__)
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != EXPECTED_SQL_SHA256:
        raise RuntimeError(f"unaudited FlagGems SQL source: {source}: {actual}")
    cls = module.SQLPersistantModel
    original = cls.build_sql_model_by_db
    if not getattr(original, _MARKER, False):
        cls.build_sql_model_by_db = staticmethod(_targeted_builder(original))
    print("BIV150_SQL_REFLECTION_INSTALLED sqlite_targeted_only=1", flush=True)
    return {"installed": True, "scope": "FlagGems SQLite configuration table reflection",
            "sql_sha256": actual, "model_cache_changed": False,
            "missing_tables_cached": False, "database_rows_changed": False}
