#!/usr/bin/env python3
"""Default stdlib CPU plan; explicit Root-only one warm compile, zero launches."""
import sys
sys.dont_write_bytecode = True
import biv150_konly_q32_w8_compile_source as c

if __name__ == '__main__':
    c.driver().main()
