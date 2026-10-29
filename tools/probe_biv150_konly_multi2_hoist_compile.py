#!/usr/bin/env python3
"""Private multi2 compile-only adapter; default CPU plan, no service activation."""
import sys
sys.dont_write_bytecode = True
import biv150_konly_multi2_hoist_source as c

if __name__ == '__main__':
    c.driver().main()
