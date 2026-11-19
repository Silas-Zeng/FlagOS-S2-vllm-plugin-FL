"""CPU replay of actual new and historical consumer arrays via unchanged adapters."""
import hashlib
import json
from pathlib import Path
import sys
import time

SOURCE=Path('/private/tmp/flagos-cjj-native-k8-v-overlap-20261028')
PARENT=Path('/private/tmp/flagos-cjj-native-k8-fullstep-frozen-3eb-20261028')
NEW=Path('/private/tmp/cjj-native-k8-v-overlap-consumer-v1-raw-20261028')
OLD=Path('/private/tmp/cjj-native-k8-consumer-v2-raw-20261028')
COMPILE=Path('/private/tmp/cjj-native-k8-v-overlap-compile-v1-raw-20261028/out')
REVIEW=Path('/private/tmp/cjj-native-k8-v-overlap-root-SO-review-v1-20261028.json')
OUT=Path('/private/tmp/cjj-native-k8-v-overlap-consumer-actual-replay-20261028.json')
NEW_SHA='4474281381486c0b813be4d2eee0b045a43858e70a498a8846d9436804fb066a'
OLD_SHA='c2e13e3766172e9b74342bbcf48b1b4729d1b9604bc249b6974740f7f90cc128'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0,str(SOURCE/'tools'))
import probe_biv150_native_k8_v_overlap_fullstep as probe


def archive(root,files,size):
    manifest=json.loads((root/'download-manifest.json').read_text())
    assert manifest['raw_files']==files and manifest['raw_bytes']==size
    assert manifest['original_owner']['same_start'] and manifest['original_owner']['state']=='Z'
    assert manifest['model_rerun'] is False
    for row in manifest['files']:
        path=root/row['relative'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    return dict(manifest_sha256=sha(root/'download-manifest.json'),files=files,bytes=size,owner=manifest['original_owner'])


start=time.monotonic()
archives=dict(new=archive(NEW,158,49147045),old=archive(OLD,158,49128520))
new_path=NEW/'out/consumer-summary.json';old_path=OLD/'out/consumer-summary.json'
assert sha(new_path)==NEW_SHA and sha(old_path)==OLD_SHA
summary=json.loads(new_path.read_text());old_summary=json.loads(old_path.read_text())
assert sha(REVIEW)=='0f0562664c9077db5c5fa0f0fd27eae91703a398860d087e6d6f424e4a848425'
compile_summary=COMPILE/'compile-summary.json';compiled=json.loads(compile_summary.read_text())
assert compiled['completed'] and compiled['compile_passed'] and not compiled['errors']
assert compiled['torch']['cuda_initialized_before'] is False and compiled['torch']['cuda_initialized_after'] is False
assert all(compiled[k]==0 for k in ('attention_launches','encoder_launches','codec_launches','reducer_launches','input_initializations'))
for row in compiled['artifact_manifest']:
    path=COMPILE/row['relative'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
proof=probe.consumer.factory.verify()
assert compiled['source_transform']==proof and compiled['explicit_cuda_flags']==probe.consumer.CUDA_FLAGS
for role in probe.consumer.factory.FILES:
    path=probe.consumer.factory.path(role);assert (COMPILE/path.name).read_bytes()==path.read_bytes()
library=COMPILE/'build/cjj_native_k8_v_overlap_probe.so'
assert sha(library)==compiled['library']['sha256']=='8b86cb688ad72dcf9859994ed2092620312cdd5ae96d194904b2bde7926aacc5'
review=json.loads(REVIEW.read_text());probe.consumer.review_fields(review)
for kind in ('IR','SASS'):
    assert sha(COMPILE/'actual-so-inspection'/(kind+'.stdout.txt'))==review['actual_SO_disassembly'][kind]['sha256']
base=dict(compile_summary_sha256=sha(compile_summary),library_sha256=sha(library),namespace=probe.consumer.factory.NAMESPACE,source_proof=proof)
for key,value in base.items():assert review[key]==value
binding=dict(**base,assembly_review_sha256=sha(REVIEW),review=review,
             external_exclusion_authority='Root serial scheduling; no namespace-based idle inference')
parent=probe.parent_files(PARENT);adapter=probe.consumer.adapted_run(parent)[1]
probe.variant_provenance(summary,binding,adapter)
try:probe.variant_provenance(old_summary,binding,adapter)
except (AssertionError,KeyError):pass
else:raise AssertionError('Historical consumer incorrectly qualified the variant')
source,receipt_adapter=probe.adapted_source(parent,'receipt')
sys.path.insert(0,str(parent/'tools'))
module=probe.load_module(parent,'biv150_native_k8_consumer_receipt','cjj_actual_v_overlap_consumer_replay')
namespace=dict(vars(module),variant_sha256=base['library_sha256'])
exec(compile(source,'<actual-V-overlap-original-receipt-identity-adapter>','exec'),namespace)
assert not {'torch','triton'}.intersection(sys.modules)
print('Replaying new 4474 consumer: four cases, original complete saved-array gates',flush=True)
t=time.monotonic();new_replay=namespace['require_consumer'](new_path,NEW_SHA);new_seconds=time.monotonic()-t
print('New consumer passed; replaying historical c2e consumer independently',flush=True)
t=time.monotonic();old_replay=module.require_consumer(old_path,OLD_SHA);old_seconds=time.monotonic()-t
assert not {'torch','triton'}.intersection(sys.modules)
report=dict(CPU_only=True,script_sha256=sha(__file__),archives=archives,
            actual_variant_compile_review_binding_verified=True,assembly_review_sha256=sha(REVIEW),
            compile_summary_sha256=sha(compile_summary),candidate_library_sha256=sha(library),
            local_archive_mapping='Original device absolute paths retained unchanged; artifacts explicitly mapped to downloaded out/ for CPU verification only. Operational fullstep admission was not invoked.',
            parent_commit=probe.consumer.PARENT,consumer_adapter=adapter,receipt_adapter=receipt_adapter,
            new_consumer=new_replay,old_consumer=old_replay,historical_consumer_rejected_as_variant=True,
            new_CPU_replay_seconds=new_seconds,old_CPU_replay_seconds=old_seconds,total_CPU_seconds=time.monotonic()-start,
            fresh_four_cases_full_arrays_replayed=True,old_four_cases_full_arrays_replayed=True,
            GPU_executed_by_reviewer=False,Torch_or_Triton_imported=False,fullstep_executed=False,
            model_verified=False,precision_gate=False,performance_verified=False)
OUT.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(report=str(OUT),sha256=sha(OUT),new=new_replay,old=old_replay,total_CPU_seconds=report['total_CPU_seconds']),indent=2))
