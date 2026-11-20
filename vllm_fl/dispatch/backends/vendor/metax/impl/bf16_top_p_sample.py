# SPDX-License-Identifier: Apache-2.0

"""Compatibility path for the FlagGems MetaX sampler and its diagnostics."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module(
    "flag_gems.runtime.backend._metax.ops.bf16_top_p_sample"
)
