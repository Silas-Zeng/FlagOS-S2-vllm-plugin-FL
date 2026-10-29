#!/usr/bin/env python3
"""Default CPU plan; explicit Root-only numeric run of the actual 9b cache pair."""
import sys
sys.dont_write_bytecode=True
import biv150_konly_multi2_service_numeric_source as bridge
if __name__=='__main__':
    bridge.c.module_from_source('_cjj_production_pair_numeric_driver',bridge.driver_source()[0]).main()
