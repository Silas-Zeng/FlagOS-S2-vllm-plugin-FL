"""Default-off two-sequence identity; reuse the exact Q32 preparation machinery.

No GPU source, codec, thresholds or loaded-kernel qualification is replaced.
The separate instance owns its own actual compiled-object bindings and evidence.
"""
from . import biv150_konly_encoded_hoist_proposal as shared
from .biv150_konly_encoded_q32_adapter import Adapter as Q32Adapter

ENV = 'VLLM_FL_BIV150_KONLY_SERVICE_MULTI2_HOIST'


class Proposal:
    SCHEMA, FORMAT, PINS = shared.MULTI2_SCHEMA, shared.FORMAT, shared.PINS

    @staticmethod
    def validate_meta(text, meta):
        return shared.validate_meta(text, meta, _multi2=True)

    @staticmethod
    def propose(text, meta):
        return shared.propose(text, meta, _multi2=True)


class Adapter(Q32Adapter):
    PROFILE = 'multi2'
    BINDING_SCHEMA = 'cjj-encoded-multi2-live-binding-v1'
    LOG_PREFIX = 'BIV150_KONLY_MULTI2_HOIST'
