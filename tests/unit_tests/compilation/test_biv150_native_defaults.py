"""复用缓存身份的纯 CPU 检查方式，不导入 torch/vLLM 或加载 SO。"""
import ast
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "vllm_fl"
NATIVE = "VLLM_FL_BIV150_NATIVE_W8A8"
spec = importlib.util.spec_from_file_location("native_default_semantics", PACKAGE / "compile_semantics.py")
sem = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sem)


class NativeDefaultsTest(unittest.TestCase):
    def fingerprint(self, env):
        config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(**sem._GEOMETRY)))
        return sem.build_fingerprint(config, "iluvatar", package_root=PACKAGE, environ=env,
                                     abi={"torch": "CPU fixture", "cxx11_abi": False})

    def test_default_and_explicit_off_need_no_library(self):
        with tempfile.TemporaryDirectory() as folder:
            env = {"BIV150_NATIVE_W8A8_LIBRARY": str(Path(folder) / "missing.so")}
            default = self.fingerprint(env)
            self.assertIs(default["native_enabled"], False)
            self.assertEqual(default, self.fingerprint({**env, NATIVE: "0"}))

    def test_explicit_on_has_distinct_library_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            library = Path(folder) / "fixture.so"
            env = {"BIV150_NATIVE_W8A8_LIBRARY": str(library), NATIVE: "1"}
            with self.assertRaises(FileNotFoundError):
                self.fingerprint(env)
            library.write_bytes(b"CPU fixture, never loaded")
            enabled = self.fingerprint(env)
            self.assertIs(enabled["native_enabled"], True)
            self.assertNotEqual(enabled["sha256"], self.fingerprint({**env, NATIVE: "0"})["sha256"])
            library.write_bytes(b"different CPU fixture")
            self.assertNotEqual(enabled["sha256"], self.fingerprint(env)["sha256"])

    def test_other_features_keep_enabled_defaults(self):
        default = self.fingerprint({})
        other = {key: "1" for key in sem._FEATURES if key != NATIVE}
        self.assertEqual(default, self.fingerprint(other))
        for key in other:
            with self.subTest(feature=key):
                self.assertNotEqual(default["sha256"], self.fingerprint({key: "0"})["sha256"])

    def test_installer_default_off_precedes_model_and_library(self):
        # 复用既有 CPU 审核的实际函数 AST 提取，绕开模块顶层 torch import。
        path = PACKAGE / "patches/biv150_native_w8a8.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        install = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "install")

        class LibraryReached(Exception):
            pass

        def load_library():
            raise LibraryReached

        for env in ({}, {NATIVE: "0"}, {NATIVE: "1"}):
            calls = []
            namespace = {"os": SimpleNamespace(environ=env),
                         "_eligible_model": lambda: calls.append("model") or True,
                         "load_library": load_library}
            exec(compile(ast.Module(body=[install], type_ignores=[]), str(path), "exec"), namespace)
            with self.subTest(environment=env):
                if env.get(NATIVE) == "1":
                    with self.assertRaises(LibraryReached):
                        namespace["install"]()
                    self.assertEqual(calls, ["model"])
                else:
                    self.assertEqual(namespace["install"](), {"installed": False})
                    self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
