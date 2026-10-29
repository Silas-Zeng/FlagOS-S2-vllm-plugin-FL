#!/usr/bin/env python3
"""Default CPU plan; optional Root-only complete multi2 reader timing."""
from biv150_konly_multi2_timing_source import adapted_driver
if __name__ == '__main__':
    source,_=adapted_driver()
    exec(compile(source,__file__,'exec'),dict(__name__='__main__',__file__=__file__))
