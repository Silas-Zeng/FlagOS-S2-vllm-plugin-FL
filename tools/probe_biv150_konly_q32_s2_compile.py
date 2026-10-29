#!/usr/bin/env python3
"""Default CPU plan; one explicit private raw stage2 warmup, zero kernel launches."""
import sys
sys.dont_write_bytecode = True
import biv150_konly_q32_s2_compile_source as c

if __name__ == '__main__':
    c.driver().main()
