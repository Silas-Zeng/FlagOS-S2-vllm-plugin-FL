#!/usr/bin/env python3
"""Default CPU plan; private Root-only Native V8 compilation, zero launches."""
import sys
import biv150_native_k8_packed_v8_factory as factory


if __name__ == '__main__':
    # Reuse the frozen native compiler; all finite differences are in the factory.
    source = factory.frontend_source()
    namespace = dict(__name__='cjj_native_v8_compile_frontend', __file__=__file__)
    exec(compile(source, __file__+'::<frozen-parent-adapter>', 'exec'), namespace)
    raise SystemExit(namespace['main']())
