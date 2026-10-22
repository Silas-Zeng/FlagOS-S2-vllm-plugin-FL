# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from vllm_fl.patches import biv150_async_token_update as async_update
from vllm_fl.patches import biv150_sql_reflection as sql_reflection


def test_async_token_update_supports_metax():
    assert async_update._platform_supported("iluvatar")
    assert async_update._platform_supported("metax")
    assert not async_update._platform_supported("cuda")


def test_sql_reflection_supports_metax():
    assert sql_reflection._platform_supported("iluvatar")
    assert sql_reflection._platform_supported("metax")
    assert not sql_reflection._platform_supported("cuda")


def test_sql_reflection_installs_with_audited_flag_gems_source():
    result = sql_reflection.install(vendor_name="metax")
    assert result["installed"] is True
    assert result["sql_sha256"] == sql_reflection.EXPECTED_SQL_SHA256
