#!/usr/bin/env python3
"""CPU plan by default; one explicit Root-owned fixed-w8 override warmup, no launches."""
import sys
sys.dont_write_bytecode=True
import biv150_konly_q32_w8_preheader_source as c

if __name__=='__main__':c.driver().main()
