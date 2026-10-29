#!/usr/bin/env python3
"""Default CPU plan; explicit Root-owned production prepare compile only."""
import sys
sys.dont_write_bytecode=True
import biv150_konly_multi2_service_compile_source as c
if __name__=='__main__':c.driver().main()
