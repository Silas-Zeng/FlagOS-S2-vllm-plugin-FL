#!/usr/bin/env python3
"""Print stdlib-only source-pinned temporary diagnostic plan."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'diagnostics/flaggems_tuner'))
from cjj_flaggems_tuner_diag import plan
assert not {'torch','triton','flag_gems','sqlalchemy'}.intersection(sys.modules)
print(json.dumps(plan(),sort_keys=True,indent=2))
