"""Opt-in diagnostic import hook; production source files remain unmodified."""

import os

if os.environ.get("BIV150_GQA_COVERAGE_DIR"):
    import functools
    import importlib.abc
    import importlib.machinery
    import sys

    class Loader(importlib.abc.Loader):
        def __init__(self, original):
            self.original = original

        def create_module(self, spec):
            return self.original.create_module(spec)

        def exec_module(self, module):
            self.original.exec_module(module)
            original = module.install

            @functools.wraps(original)
            def install():
                result = original()
                if result.get("installed") and not result.get("already"):
                    from biv150_gqa_coverage import install as observe

                    observe(os.environ["BIV150_GQA_COVERAGE_DIR"])
                return result

            module.install = install

        def __getattr__(self, name):
            return getattr(self.original, name)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "vllm_fl.patches.biv150_native_w8a8":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                raise ImportError("Native operator source missing")
            spec.loader = Loader(spec.loader)
            sys.meta_path.remove(self)
            return spec

    sys.meta_path.insert(0, Finder())
