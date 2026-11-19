import json,hashlib,argparse,math
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert not a.out.exists();r=a.raw_root;m=json.loads((r/'download-manifest.json').read_text());assert m['raw_files']==30 and m['original_owner']['same_start'] and m['original_owner']['state']=='Z' and not m['model_rerun']
 for f in m['files']:
  x=r/f['relative'];assert x.stat().st_size==f['bytes'] and sha(x)==f['sha256']
 j=json.loads((r/'out/manifest.json').read_text());s=json.loads((r/'out/state.json').read_text());assert s['completed'] and s['smoke_passed'] and s['cleanup_complete'] and not s['errors'];assert not s['model_precision_verified'] and not s['throughput_measured'] and not s['native_quality_covered']
 assert j['source_commit']=='a975e8715beb752d6a76b456f417ba2011ea917b' and j['controller_sha256']=='4d597cd4b8d632a8ea928bbdc6be607dc02ed02f03ba241bcc1d47ddd6059a63'
 for name,h in j['source_hashes'].items():assert sha(a.source_root/name)==h
 rt=j['actual_runtime'];assert rt['actual_compile_semantics'] and all(x==j['expected_compile_semantics'] for x in rt['actual_compile_semantics']);assert len(rt['preserved_parent_installations'])==9
 for values in rt['preserved_parent_installations'].values():assert all(x['installed'] for x in values)
 n=rt['native_decode'];assert n['enabled'] and n['actual_imports_verified'] and n['host_or_capture_only'] and not n['actual_request_frequency_proven'] and not n['Graph_replay_observed'] and not n['quality_covered'];assert len(n['installations'])==1;install=n['installations'][0]
 assert install['installed'] and install['enabled'] and install['loaded_library_confirmed'] and not install['runtime_compile'] and install['physical_page_bytes']==16384;assert install['module_sha256']==sha(a.source_root/'vllm_fl/patches/biv150_konly_encoded_native.py')
 pairs={};events=n['events'];assert len(events)==168
 for x in events:
  v=x['event'];size,segments=v['shape'],v['segments'];assert (size,segments) in ((64,2),(32,4));assert v['physical_page_bytes']==16384 and v['cache_pages']==34276 and v['native_launch_returned'];assert v['library']['sha256']=='22e4a5e2a0b9d9a36ea390a11126417e1bf58e3c27f6b96ba0913b27690f09dd';assert v['format']=='cjj-konly-rn32-rne-v2';q,o,h,t=[v[k] for k in ('query','partial','maximum','table')];assert q['shape']==[size,16,128] and q['strides'] in ([2048,128,1],[2560,128,1]);assert o['shape']==[64,16,segments,128] and o['strides']==[32768,2048,128,1];assert h['shape']==[64,16,segments] and h['strides']==[256,16,1];assert t['shape']==[size,8192] and t['strides']==[8192,1]
  pairs.setdefault((x['marker'],size,segments),set()).add(v['entry_generation'])
 assert len(pairs)==4 and all(len(v)==42 for v in pairs.values())
 b=json.loads((r/'out/native-burst64-requests.json').read_text());assert b['completed']==b['concurrency_requested']==len(b['records'])==64 and not b['actual_native_bucket_proven'] and not b['quality_covered'];assert {x['index'] for x in b['records']}==set(range(64))
 for v in b['records']:assert v['response']['usage']['completion_tokens']==16 and v['request']['ignore_eos'] is True
 assert len(json.loads((r/'out/smoke-requests.json').read_text()))==3
 report=dict(passed=True,CPU_only=True,source_commit=j['source_commit'],raw_files_verified=30,raw_bytes_verified=m['raw_bytes'],source_hashes_verified=len(j['source_hashes']),actual_C3_verified=True,parent_installers_verified=9,actual_native_library_install_verified=True,native_host_capture_events=len(events),each_shape_each_event_has_42_entry_generations=True,observed_query_stride_and_width16_verified=True,observed_scratch_backing_batch64=True,HTTP64_completed=True,actual_graph_replays_counted=False,native_request_quality_covered=False,model_precision_verified=False,throughput_measured=False,capacity_reclaimed=False,errors=[],source_root=str(a.source_root),raw_root=str(r),manifest_sha256=sha(r/'out/manifest.json'))
 a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(dict(passed=True,sha256=sha(a.out),out=str(a.out))))
if __name__=='__main__':main()
