"""Private opt-in bootstrap; no token/config values are printed."""
import os
if os.environ.get('CJJ_GATHER_SNAPSHOT') == '1':
    try:
        import importlib.util
        import importlib.machinery
        from pathlib import Path
        import sys
        own=Path(__file__).resolve().parent
        others=[x for x in sys.path if Path(x or os.getcwd()).resolve()!=own]
        if importlib.machinery.PathFinder.find_spec('sitecustomize',others) is not None:
            raise RuntimeError('Existing sitecustomize requires explicit composition review')
        path=Path(__file__).resolve().parents[1]/'biv150_hidden_gather_snapshot.py'
        spec=importlib.util.spec_from_file_location('cjj_hidden_gather_snapshot_private',path)
        module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module
        spec.loader.exec_module(module)
        module.install()
    except BaseException:
        raise SystemExit('Private runtime snapshot bootstrap refused configuration/source') from None
