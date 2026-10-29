#!/usr/bin/env python3
"""Default CPU plan; explicit isolated multi2 numerical screening only."""
import biv150_konly_multi2_hoist_source as c
from biv150_konly_multi2_numeric_source import adapted_driver
if __name__=='__main__':
    c.module_from_source('cjj_private_multi2_numeric_driver',adapted_driver()[0]).main()
