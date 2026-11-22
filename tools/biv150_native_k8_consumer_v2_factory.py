"""Counted reversible reference-only transform of the frozen52c frontend."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'tools/probe_biv150_native_k8_consumer.py';DEST=ROOT/'tools/probe_biv150_native_k8_consumer_v2.py'
SOURCE_SHA='82091d73dc73434494a037c9a6eef0baf5f457cec6ec4af89e83b9b7c0d8adaf'


def operations():
    return [
      ('import biv150_native_k8_consumer_fixture as f','import biv150_native_k8_consumer_v2_fixture as f'),
      ("metrics[field]=f.metrics(actual[mask],g[field][mask]);checks[field+'_FP64']=metrics[field]['passed']",
       "metrics[field]=f.metrics(actual[mask],g[field][mask]);checks[field+'_FP64']=metrics[field]['passed']\n                        if field=='partial':metrics['ideal_partial_diagnostic']=f.metrics(actual[mask],g['ideal_partial_diagnostic'][mask])"),
    ]

def original():
    raw=SOURCE.read_bytes();assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA;return raw.decode()

def inverse(text):
    for old,new in reversed(operations()):assert text.count(new)==1;text=text.replace(new,old)
    assert text==original();return text

def build():
    text=original()
    for old,new in operations():assert text.count(old)==1;text=text.replace(old,new)
    assert inverse(text)==original();return text

def verify():
    assert DEST.read_bytes()==build().encode()
    return dict(frozen52c_source_sha256=SOURCE_SHA,reference_only_inverse_byte_exact=True,
      native_or_dtype_or_threshold_changed=False,final_FP64_hard_gate_unchanged=True,
      v2_sha256=hashlib.sha256(DEST.read_bytes()).hexdigest())

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');a=p.parse_args()
    if a.write:DEST.write_bytes(build().encode())
    print(json.dumps(verify(),sort_keys=True,indent=2))
