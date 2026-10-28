"""将 BI-V150 MiniCPM 的插件实现身份纳入 vLLM 的官方编译缓存键。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

_KEY = "vllm_fl_compile_semantics"
_GEOMETRY = dict(hidden_size=2048, intermediate_size=6144, num_hidden_layers=42,
                 num_attention_heads=16, num_key_value_heads=2, vocab_size=130560)
_FEATURES = ("VLLM_FL_BIV150_NATIVE_W8A8", "VLLM_FL_BIV150_GQA_DECODE",
             "VLLM_FL_BIV150_PREFILL_NORMALIZE", "VLLM_FL_BIV150_Q_FRAGMENT",
             "VLLM_FL_BIV150_STABLE_SMALL_O",
             "VLLM_FL_BIV150_HEAD_TEMPERATURE")
# 仅记录会影响实际计算选择的配置；不收集凭据、实验目录或 profiler 开关。
_DISPATCH_ENV = ("VLLM_FL_PREFER", "VLLM_FL_STRICT", "VLLM_FL_DENY_VENDORS",
                 "VLLM_FL_ALLOW_VENDORS", "VLLM_FL_PER_OP", "VLLM_FL_OOT_ENABLED",
                 "VLLM_FL_OOT_WHITELIST", "VLLM_FL_OOT_BLACKLIST",
                 "VLLM_FL_FLAGOS_WHITELIST", "VLLM_FL_FLAGOS_BLACKLIST",
                 "VLLM_FL_FLAGOS_BLACKLIST_APPEND")


def native_model_matches(config) -> bool:
    return all(getattr(config, key, None) == value for key, value in _GEOMETRY.items())


def native_library_path(*, package_root=None, environ=None) -> Path:
    """指纹与实际 load_library 共用解析；路径本身不进入缓存身份。"""
    env = os.environ if environ is None else environ
    package = Path(__file__).resolve().parent if package_root is None else Path(package_root)
    return Path(env.get("BIV150_NATIVE_W8A8_LIBRARY",
                        str(package.parent / "build-native-w8a8/biv150_native_w8a8.so"))).resolve()


def native_gqa_library_path(*, package_root=None, environ=None) -> Path:
    """与GQA loader共用解析；只将实际库内容纳入身份，不使用部署路径。"""
    env = os.environ if environ is None else environ
    package = Path(__file__).resolve().parent if package_root is None else Path(package_root)
    return Path(env.get("BIV150_NATIVE_GQA_LIBRARY",
                        str(package.parent / "build-native-gqa/corex_native_gqa_s2_probe.so"))).resolve()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def _runtime_abi():
    # PlatformFL 已加载 torch；这里不 import torch，也不创建设备上下文。
    torch = sys.modules.get("torch")
    if torch is None:
        raise RuntimeError("缓存语义身份须在 PlatformFL 已加载 torch 后计算")
    return {"torch": str(torch.__version__),
            "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI)}


def build_fingerprint(vllm_config, vendor_name, *, package_root=None, environ=None, abi=None):
    """只读文件和配置；不导入 runner、模型、算子或加载 SO。"""
    model = getattr(getattr(vllm_config, "model_config", None), "hf_config", None)
    if vendor_name != "iluvatar" or not native_model_matches(model):
        return None
    package = Path(__file__).resolve().parent if package_root is None else Path(package_root)
    env = os.environ if environ is None else environ
    features = {key: env.get(key, "1") == "1" for key in _FEATURES}
    features["VLLM_FL_BIV150_PREFILL_Q32"] = (
        features["VLLM_FL_BIV150_Q_FRAGMENT"]
        and features["VLLM_FL_BIV150_PREFILL_NORMALIZE"]
        and env.get("VLLM_FL_BIV150_PREFILL_Q32", "1") == "1")
    features["VLLM_FL_BIV150_PREFILL_MULTI2"] = (
        features["VLLM_FL_BIV150_PREFILL_Q32"]
        and env.get("VLLM_FL_BIV150_PREFILL_MULTI2", "1") == "1")
    features["VLLM_FL_BIV150_NATIVE_GQA"] = (
        features["VLLM_FL_BIV150_GQA_DECODE"]
        and env.get("VLLM_FL_BIV150_NATIVE_GQA", "1") == "1")
    features["VLLM_FL_BIV150_KONLY_ENCODED_SERVICE"] = env.get("VLLM_FL_BIV150_KONLY_ENCODED_SERVICE", "0") == "1"
    paths = {package / name for name in ("__init__.py", "platform.py", "utils.py",
             "compile_semantics.py", "worker/model_runner.py", "worker/worker.py",
             "patches/biv150_native_w8a8.py", "dispatch/config/iluvatar.yaml")}
    for pattern in ("patches/biv150_*.py", "compilation/*.py", "dispatch/*.py", "dispatch/config/*.py",
                    "dispatch/backends/flaggems/impl/*.py",
                    "dispatch/backends/vendor/iluvatar/**/*.py"):
        paths.update(package.glob(pattern))
    source = {path.relative_to(package).as_posix(): _sha(path) for path in sorted(paths)}
    external_config = {}
    for key in ("VLLM_FL_CONFIG", "VLLM_FL_OP_CONFIG"):
        configured = env.get(key, "").strip()
        external_config[key] = _sha(Path(configured)) if configured else None
    dispatch = {key: env.get(key) for key in _DISPATCH_ENV}
    dispatch["VLLM_FL_PREFER_ENABLED"] = env.get("VLLM_FL_PREFER_ENABLED", "True").lower() in ("true", "1")
    dispatch["USE_FLAGGEMS"] = env.get("USE_FLAGGEMS", "True").lower() in ("true", "1")
    facts = {"schema": 1, "source": source, "features": features, "dispatch": dispatch,
             "external_config": external_config, "abi": _runtime_abi() if abi is None else abi,
             "gqa_target_programs": int(env.get("VLLM_FL_BIV150_GQA_TARGET_PROGRAMS", "256"))}
    facts["small_out_lowering"] = (features["VLLM_FL_BIV150_STABLE_SMALL_O"]
        and env.get("VLLM_FL_BIV150_SMALL_OUT_LOWERING","1") == "1")
    facts['large_linear'] = (facts['small_out_lowering']
        and env.get('VLLM_FL_BIV150_LARGE_LINEAR','1') == '1')
    if features["VLLM_FL_BIV150_NATIVE_W8A8"]:
        library = native_library_path(package_root=package, environ=env)
        # wheel 可以不携带构建源码；实际加载的 SO 始终必须存在并按内容哈希。
        native_sources = {}
        for name in ("native_w8a8.cpp", "native_w8a8.cu"):
            path = package.parent / "csrc/iluvatar" / name
            native_sources[name] = _sha(path) if path.is_file() else None
        facts["native"] = {"library_sha256": _sha(library), "source": native_sources}
    if features["VLLM_FL_BIV150_NATIVE_GQA"]:
        library = native_gqa_library_path(package_root=package, environ=env)
        sources = {}
        for name in ("native_gqa_s2.cpp", "native_gqa_s2.cu"):
            path = package.parent / "csrc/iluvatar" / name
            sources[name] = _sha(path) if path.is_file() else None
        facts["native_gqa"] = {"library_sha256": _sha(library), "source": sources}
    compact = None
    if env.get("VLLM_FL_BIV150_KONLY_COMPACT_SERVICE", "0") == "1":
        if features["VLLM_FL_BIV150_KONLY_ENCODED_SERVICE"] or env.get("VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST", "0") == "1":
            raise ValueError("compact format/old-hoist conflict")
        directory = package / "patches"
        compact_paths = sorted(directory.glob("biv150_konly_compact_*.py")) + [directory / "biv150_konly_compact_sources.json"]
        compact = {"format": "cjj-konly-compact12544-rn32-rne-v1", "page_bytes": 12544, "producer_boundary": "owned-compact-before-stock-native", "q32_hoist_enabled": False,
                   "implementation_sha256": _digest({path.name: _sha(path) for path in compact_paths})}
        facts["konly_compact"] = compact
    encoded = None
    if features["VLLM_FL_BIV150_KONLY_ENCODED_SERVICE"]:
        directory = package / "patches"
        encoded_paths = sorted(directory.glob("biv150_konly_encoded_*.py")) + [directory / "biv150_konly_encoded_sources.json"]
        encoded = {"format": "cjj-konly-rn32-rne-v2",
                   "producer_boundary": "owned-encoded-before-stock-native",
                   "q32_hoist_enabled": env.get("VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST", "0") == "1",
                   "implementation_sha256": _digest({path.name: _sha(path) for path in encoded_paths})}
        facts["konly_encoded"] = encoded
    result = {"schema": 1, "sha256": _digest(facts),
              "native_enabled": features["VLLM_FL_BIV150_NATIVE_W8A8"]}
    if encoded is not None:
        result["konly_encoded"] = encoded
    if compact is not None:
        result["konly_compact"] = compact
    return result


class _WithCompileSemantics:
    """保留非 dict additional_config 的 SupportsHash 对象及其动态 hash。"""
    def __init__(self, base, fingerprint):
        self._base = base
        self._fingerprint = fingerprint

    def compute_hash(self):
        return _digest({"base": self._base.compute_hash(), _KEY: self._fingerprint})

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_base"), name)


def merge_fingerprint(additional_config, fingerprint):
    if fingerprint is None:
        return additional_config
    if isinstance(additional_config, dict):
        # 不原地修改调用方持有的 dict，也不覆盖其它团队配置。
        return {**additional_config, _KEY: fingerprint}
    if isinstance(additional_config, _WithCompileSemantics):
        return _WithCompileSemantics(additional_config._base, fingerprint)
    if not callable(getattr(additional_config, "compute_hash", None)):
        raise TypeError("additional_config 必须为 dict 或 SupportsHash")
    return _WithCompileSemantics(additional_config, fingerprint)


def apply_compile_semantics(vllm_config, vendor_name):
    fingerprint = build_fingerprint(vllm_config, vendor_name)
    if fingerprint is not None:
        vllm_config.additional_config = merge_fingerprint(vllm_config.additional_config, fingerprint)
        # 仅报告缓存身份，不能作为模型逐请求执行或正确性证据。
        print("BIV150_COMPILE_SEMANTICS", json.dumps(fingerprint, sort_keys=True), flush=True)
    return fingerprint
