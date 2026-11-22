"""Finite adapter of the accepted3eb/117 consumer; no new numerical framework."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import biv150_native_k8_packed_v8_factory as factory
import biv150_native117_pair_contract as previous
import biv150_native127_v8_fixture as vf

ROOT=Path(__file__).resolve().parents[1]
SOURCE='fcf11f11f1d3d241784d292b7970c3d2fd862594'
DEVELOPMENT_BASE='5f0086be50f1d6d115c1ebb25b9d3c7250470372'
COMPILE_SHA='6c443d0bc2c473138dd887974b1cadb1e14329a8f8a6c5376a0686061ddc7b17'
# Root independently replayed this static report; no numeric or slot permission.
STATIC_REVIEW_SHA='1c9bf67eaa92a25865c2ffdf62270f8bb91c40391e959f541ec86560c7e0f352'
MANIFEST_SHA='6a34c8a5e337722fe26cf803d564afd4beed8c37753b2376a10c09b6f582769d'
ROLES={'control':(factory.REFERENCE_SO,'cjj_native_k8_klookahead_probe'),
       'packed_v8':('138cac22acc33d5a632d7156d0d3366af84697900d0a6c0593874bb71bf34e43',factory.NAMESPACE)}
STAGES=previous.STAGES
sha=previous.sha


def load(parent,name,alias):
    if alias in sys.modules:
        module=sys.modules[alias];assert Path(module.__file__).resolve()==(parent/'tools'/(name+'.py')).resolve();return module
    return previous.load(parent,name,alias)


def plan(role=None):
    return dict(component='isolated packed V8 N64/S2 consumer; no writer or timing',
      actual_compile_source=SOURCE,development_base=DEVELOPMENT_BASE,parent=previous.old_consumer.PARENT,
      source_inverse=factory.verify(),roles={r:dict(library_sha256=d,namespace=n) for r,(d,n) in ROLES.items()},
      static_Root_review_sha256=STATIC_REVIEW_SHA,static_review_is_numeric_admission=False,
      role=role,library_sha256=ROLES[role][0] if role else None,cases=vf.CASES,cache_pages=vf.PAGES,table_columns=vf.COLS,lengths=list(vf.SHORT)+[6143,6144],
      page_bytes=16384,V_codes_byte_offset=8192,V_scales_byte_offset=12288,V_scale_FP32_page_stride=4096,
      control_V='BF16_RNE(F32(signed_I8_code*F32scale))',K_Q_codec='unchanged rational RN32/RNE',
      scratch='original3eb consumer compact S2; not service width16 fullstep',
      partial_reference='original3eb page16 BF16-P; extent-only decoder adapter',
      final_reference='original ideal quantized-QK FP64 over dequantized BF16 V',
      original_BF16_proxy='unquantized BF16 Q/K/V; separate diagnostic, unchanged .005 gates',
      cross_arm='full guarded partial/maximum/summation/output bitwise; input caches intentionally have different typed roles',
      phases=['eagerA','captureA','changedB+poison Graph before freshB','sideC'],
      default_stdlib_only=True,Root_exclusive_scheduling_required=True,numeric_verified=False,
      performance_verified=False,precision_gate=False,model_verified=False,service_verified=False,capacity_recovered=False)


def adapted_run(parent):
    original=previous.function(parent,'probe_biv150_native_k8_consumer_v2','run')
    return previous.edit(original,[
      ('f.sha(args.library.read_bytes())==f.LIBRARY_SHA','f.sha(args.library.read_bytes())==args.role_sha256',2),
      ('torch.ops.cjj_native_k8_qk_probe.produce','getattr(torch.ops,args.namespace).produce',1),
      ('record=dict(case=[n,s,pitch],snapshots={},checks=[],golden={});result[\'records\'].append(record)',
       'record=dict(case=[n,s,pitch],snapshots={},checks=[],golden={});result[\'records\'].append(record)\n            record[\'CPU_oracles\']=f.cpu_oracles(cases,blob)',1),
      ("value=inputs['cache'].view(torch.bfloat16).as_strided((f.PAGES,16,2,128),(8192,256,128,1),storage_offset=(f.LAYER_OFFSET+8192)//2)",
       "value=(inputs['cache'].view(torch.bfloat16).as_strided((f.PAGES,16,2,128),(8192,256,128,1),storage_offset=(f.LAYER_OFFSET+8192)//2) if args.role=='control' else inputs['cache'].view(torch.int8).as_strided((f.PAGES,16,2,128),(16384,256,128,1),storage_offset=f.LAYER_OFFSET+8192))",1),
      ("            assert key.data_ptr()==scale.data_ptr()==inputs['cache'].data_ptr()+f.LAYER_OFFSET and value.data_ptr()==key.data_ptr()+8192",
       "            assert key.data_ptr()==scale.data_ptr()==inputs['cache'].data_ptr()+f.LAYER_OFFSET and value.data_ptr()==key.data_ptr()+8192\n            record['binding'].update(V_dtype=str(value.dtype),V_shape=list(value.shape),V_stride=list(value.stride()),V_offset=value.storage_offset(),scale_shape=list(scale.shape),scale_stride=list(scale.stride()),scale_offset=scale.storage_offset())",1)])


def admission(a):
    assert a.root_owned_exclusive,'Root serial slot confirmation required; not inferred from namespace PID table'
    assert re.fullmatch('[0-9a-f]{64}',STATIC_REVIEW_SHA or ''),'actual Root static review pending'
    assert a.static_review_sha256==STATIC_REVIEW_SHA and sha(a.static_review)==STATIC_REVIEW_SHA
    parent=previous.parent(a.parent_source)
    for revision in (SOURCE,DEVELOPMENT_BASE):
        subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',revision,'HEAD'],check=True)
    report=json.loads(a.static_review.read_text());summary=json.loads(a.compile_summary.read_text())
    assert sha(a.compile_summary)==COMPILE_SHA==report['compile_summary_sha256']
    assert report['completed'] is True and report['errors']==[] and summary['completed'] is True and summary['errors']==[]
    assert summary['compile_passed'] is True and report['source_commit']==summary['source_commit']==SOURCE
    assert summary['source_transform']==report['source_inverse']==factory.verify()
    assert report['SO_sha256']==summary['library']['sha256']==ROLES['packed_v8'][0]
    assert report['reference_SO_sha256']==ROLES['control'][0] and report['download_manifest_sha256']==MANIFEST_SHA
    assert report['ABI_layout_equal_5898'] is True and report['final_SO_images_equal_cuda_object'] is True
    assert report['IR']['original_QK_softmax_PV_math_and_flags_equal'] is True
    assert report['backend']['no_SME_or_G2S_in_entire_function'] is True and report['backend']['Knext']['preserved'] is True
    assert len(report['backend']['paths'])==2 and all(p['barrier_dominates_PV'] is True and p['V_publish_paths_reach_barrier'] is True for p in report['backend']['paths'])
    assert report['recommendation']['worth_small_isolated_numeric_screen'] is True
    assert all(report[k] is False for k in ('numeric_verified','performance_verified','service_verified','model_precision_verified','capacity_recovered'))
    assert summary['frontend_explicit_kernel_launches']==summary['input_initializations']==0
    for row in summary['artifact_manifest']:
        rel=Path(row['relative']);assert not rel.is_absolute() and '..' not in rel.parts
        p=a.compile_summary.parent/rel;assert p.resolve().is_relative_to(a.compile_summary.parent.resolve())
        assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
    for role in ('qk.cu','qk.cpp','codec.cuh'):
        assert (a.compile_summary.parent/factory.path(role).name).read_bytes()==factory.path(role).read_bytes()
    assert sha(a.library)==ROLES['packed_v8'][0] and sha(a.control_library)==ROLES['control'][0]
    assert a.codec_proof_sha256==previous.CODEC_SHA and sha(a.codec_summary)==previous.CODEC_SHA
    assert a.out.is_absolute() and not a.out.exists()
    assert not a.out.resolve().is_relative_to(parent) and not a.out.resolve().is_relative_to(ROOT)
    return parent,dict(compile_source=SOURCE,development_base=DEVELOPMENT_BASE,compile_summary_sha256=COMPILE_SHA,
      static_Root_review_sha256=STATIC_REVIEW_SHA,libraries={role:d for role,(d,_) in ROLES.items()},
      static_resources=report['resources'],actual_SO_sha256=report['SO_sha256'],
      Root_serial_slot_confirmed=True,numeric_permission='only this explicit Root invocation; no static automatic queue')


def compare(arms):
    assert set(arms)==set(ROLES)
    rows={role:{tuple(x['case']):x for x in report['records']} for role,report in arms.items()}
    assert all(set(r)==set(vf.CASES) for r in rows.values())
    result=[]
    for case in vf.CASES:
        a,b=(rows[role][case] for role in ROLES)
        assert set(a['snapshots'])==set(b['snapshots'])==set(STAGES)
        for stage in STAGES:
            for field in ('partial','maximum','summation','output'):
                x,y=(z['snapshots'][stage][field] for z in (a,b))
                assert (x['bytes'],x['sha256'])==(y['bytes'],y['sha256']),(case,stage,field)
        result.append(dict(case=case,full_guarded_OML_final_bitwise=True,stages=len(STAGES)))
    return result
