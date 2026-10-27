"""Only opt-in worker diagnostics; never import a framework at interpreter start."""
import os
if os.environ.get('CJJ_FLAGGEMS_TUNER_DIAGNOSTIC')=='1':
    import importlib.machinery
    from pathlib import Path
    import sys
    here=Path(__file__).resolve().parent
    paths=[p for p in sys.path if Path(p or '.').resolve()!=here]
    other=importlib.machinery.PathFinder.find_spec('sitecustomize',paths)
    if other is not None:
        raise SystemExit('Diagnostic refused to shadow existing sitecustomize: '+str(other.origin))
    from cjj_flaggems_tuner_diag import activate
    activate(os.environ['CJJ_FLAGGEMS_TUNER_DIAGNOSTIC_OUT'])
