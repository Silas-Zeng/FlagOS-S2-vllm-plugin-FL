from pathlib import Path
import inspect,os,json
import vllm_fl
from vllm_fl.patches import biv150_live_carry_gate as own
from vllm.model_executor.models.llama import LlamaDecoderLayer
original=LlamaDecoderLayer.forward
os.environ['VLLM_FL_BIV150_LIVE_CARRY_GATE']='0'
disabled=own.install()
assert not disabled['installed'] and LlamaDecoderLayer.forward is original
os.environ['VLLM_FL_BIV150_LIVE_CARRY_GATE']='1'
enabled=own.install()
assert enabled['installed'] and LlamaDecoderLayer.forward is not original
source=inspect.getsource(LlamaDecoderLayer.forward)
assert '_biv150_post_norm_mlp' in source and 'self.self_attn' in source
print(json.dumps(dict(sdk_install_only=True,gpu_math_not_tested=True,package=str(Path(vllm_fl.__file__).resolve()),disabled=disabled,enabled=enabled,generated_source=source)))
