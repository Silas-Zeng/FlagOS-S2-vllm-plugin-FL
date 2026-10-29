#!/usr/bin/env python3
"""Default CPU plan; explicit Root-only Q32 S2 preheader override warmup,0 launches."""
import sys
sys.dont_write_bytecode=True
import biv150_konly_q32_s2_preheader_source as c
if __name__=='__main__':c.driver().main()
