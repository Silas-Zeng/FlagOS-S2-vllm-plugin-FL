"""Fixed PR128 numerical-only admission; default never imports Torch."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MAIN = '5f0086be50f1d6d115c1ebb25b9d3c7250470372'
COMPILE_SOURCE = 'fbc6858bdccf7688936f9d0887b65e855762ab83'
LIBRARY = '8eefb1b8012b62e6849eeaf5fc43f9600746a39f19354532255ba899d1c9e592'
REFERENCE = '5cbb8853a81a4621efc936fa60b952391bed1b56408db1bad62d0031be573dbe'
COMPILE_SUMMARY = 'd7254573f698cb04fd66f389c4d3746c87752ba14c8f0357df85b41d302dc54c'
# Actual static review is independently pending. No --override or user hash bypass.
STATIC_REVIEW_SHA256 = None
ROOT_ADMISSION_SHA256 = None
SOURCE_PINS = {
 'tools/kernels/biv150_native_i8_gate2048.cpp':'68d458a4f20ffd7c7f82597249ca8cdb916efe7eb8546a7215ecd9c7670ab509',
 'tools/kernels/biv150_native_i8_gate2048.cu':'d47c27e721c2d54d73e23e4a936bcb517eaee81bc5882bda333bc4d4b532e624',
 'vllm_fl/patches/biv150_native_w8a8.py':'78a9af84ef73c295e85698bed215c60c7547a584f32635c31186dace5e0ac708',
 'csrc/iluvatar/native_w8a8.cpp':'b37c84064474fe904334d7c06f473712077230f0d319df53032372b8b8111bcb',
 'csrc/iluvatar/native_w8a8.cu':'d18191625fe45ec4aacba09d1f982bfc88a61ddd0efae32566f8a8b91f502297'}
STAGES = ('eagerA','graphA','graphB','eagerB','sideC')
GUARD = 64

def sha(raw):return hashlib.sha256(raw).hexdigest()
def need(ok,why):
 if not ok:raise ValueError(why)
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])
def source_proof():
 for name,h in SOURCE_PINS.items():
  actual=(ROOT/name).read_bytes();need(sha(actual)==h,'source identity '+name)
  base=COMPILE_SOURCE if name.startswith('tools/kernels/') else MAIN
  need(actual==git('show',base+':'+name),'exact frozen source '+name)
 need(not git('diff',MAIN,'--','vllm_fl','csrc'),'production remains latest main')
 text=(ROOT/'vllm_fl/patches/biv150_native_w8a8.py').read_text()
 node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='quantize_weight')
 return ast.get_source_segment(text,node)

def plan():
 quant=source_proof()
 return dict(scope=dict(M=2048,K=2048,N=12288,output=[2048,6144]),
  compile_source=COMPILE_SOURCE,merged_main=MAIN,candidate_library_sha256=LIBRARY,
  reference_library_sha256=REFERENCE,static_review_sha256=STATIC_REVIEW_SHA256,
  Root_admission_sha256=ROOT_ADMISSION_SHA256,static_review_pending=STATIC_REVIEW_SHA256 is None,
  quantize_weight_function_sha256=sha(quant.encode()),source_pins=SOURCE_PINS,
  reference='unchanged 5cbb gate_silu_cached_probe(x,wq,cols,compressed_marker,256)',
  candidate_inputs='unchanged 5cbb quantize_rows_probe(x,256); same fixed weight quantizer',
  stages=list(STAGES),fixed_wq_cols_marker_across_all_epochs=True,
  changed_Graph_B_before_fresh_B=True,complete_output_comparison='bitwise; original gate_auto exact route gate',
  synthetic_fixture_only=True,real_model_weight_coverage=False,numeric_verified=False,
  performance_verified=False,model_precision_verified=False,service_installed=False,
  timings=False,default_CPU_only=True)

def review_fields(r):
 need(r['schema']=='cjj_native128_actual_SO_independent_review_v1','actual reviewer schema')
 need(r['passed'] is True and r['static_ready_for_numeric'] is True,'actual static readiness')
 need(r['source_commit']==COMPILE_SOURCE and r['library_sha256']==LIBRARY and r['summary_sha256']==COMPILE_SUMMARY,'actual static source/SO/summary binding')
 need(r['all_collector_and_summary_hashes_verified'] is True and r['object_embedded_code_identical_to_final_SO'] is True,'actual final SO and complete artifacts')
 need(all(r[k] is False for k in ('numeric_verified','performance_verified','precision_gate','service_verified')),'no inherited device/model/service qualification')
 return True

def require_admissions(review_bytes,approval_bytes):
 need(STATIC_REVIEW_SHA256 is not None and ROOT_ADMISSION_SHA256 is not None,'actual static review and Root approval are pending')
 need(sha(review_bytes)==STATIC_REVIEW_SHA256 and sha(approval_bytes)==ROOT_ADMISSION_SHA256,'frozen review/approval hash')
 r=json.loads(review_bytes);a=json.loads(approval_bytes);review_fields(r)
 # The actual reviewer schema is checked above; exact approved file hashes
 # remain pending until Root's separate review and narrow receipt commit.
 need(a['approved_numeric_only'] is True and a['source_commit']==COMPILE_SOURCE,'Root numeric-only source admission')
 need(a['actual_static_review_sha256']==STATIC_REVIEW_SHA256 and a['library_sha256']==LIBRARY,'Root exact review/SO binding')
 need(a['numeric_verified'] is False and a['performance_verified'] is False and a['service_verified'] is False,'no inherited numeric/service qualification')
 return r,a

def preflight(a):
 source_proof()
 # Fail closed on unknown static review before any Torch import or device work.
 need(STATIC_REVIEW_SHA256 is not None and ROOT_ADMISSION_SHA256 is not None,'actual static review and Root approval are pending')
 need(a.root_owned_exclusive is True,'Root exclusive schedule required')
 need(git('rev-parse','HEAD').decode().strip()==a.expected_source_commit,'exact numeric frontend commit')
 need(not git('status','--porcelain'),'clean numeric frontend')
 for ref in (COMPILE_SOURCE,MAIN):subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',ref,'HEAD'],check=True)
 need(a.out.is_absolute() and not a.out.exists() and not a.out.resolve().is_relative_to(ROOT),'new output outside source tree')
 for path,h in ((a.library,LIBRARY),(a.reference_library,REFERENCE),(a.compile_summary,COMPILE_SUMMARY)):
  need(path.is_absolute() and path.is_file() and sha(path.read_bytes())==h,'exact library/compile input')
 s=json.loads(a.compile_summary.read_text());need(s['completed'] is True and s['compile_passed'] is True and s['errors']==[],'actual compile completed')
 need(s['source_commit']==COMPILE_SOURCE and s['library']['sha256']==LIBRARY,'compile actual identity')
 for row in s['artifacts']:
  rel=Path(row['path']);need(not rel.is_absolute() and '..' not in rel.parts,'safe compile artifact')
  b=(a.compile_summary.parent/rel).read_bytes();need((len(b),sha(b))==(row['bytes'],row['sha256']),'full actual compile artifacts')
 r,approval=require_admissions(a.static_review.read_bytes(),a.root_admission.read_bytes())
 return dict(frontend_source=a.expected_source_commit,compile_source=COMPILE_SOURCE,
  actual_compile_summary_sha256=COMPILE_SUMMARY,actual_library_sha256=LIBRARY,
  reference_library_sha256=REFERENCE,static_review_sha256=STATIC_REVIEW_SHA256,
  Root_admission_sha256=ROOT_ADMISSION_SHA256,review=r,Root_admission=approval)
