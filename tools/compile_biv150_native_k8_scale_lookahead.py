#!/usr/bin/env python3
"""Default stdlib plan; explicit Root-only private compile and actual SO export."""
import biv150_native_k8_scale_lookahead_factory as factory

if __name__ == '__main__':
    source = factory.frontend_source()
    namespace = dict(__name__='cjj_native_scale_compile_frontend', __file__=__file__)
    exec(compile(source, __file__+'::<frozen-parent-adapter>', 'exec'), namespace)
    raise SystemExit(namespace['main']())
