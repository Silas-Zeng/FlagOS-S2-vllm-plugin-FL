#!/usr/bin/env python3
"""Small stdlib checks for the compile frontend; no fake worker/receipt or framework."""
import ast,json,sys
from pathlib import Path
from types import SimpleNamespace as NS
import compile_biv150_dynamic_hidden_gather as c

def main():
 p=c.plan();assert len(c.CASES)==84 and p['expected_warmups']==168
 assert {m for m,n in c.CASES}==set(c.INPUT_ROWS) and {n for m,n in c.CASES}==set(c.SELECTED_ROWS)
 source=Path(c.__file__).read_text();tree=ast.parse(source)
 execute=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='execute')
 calls=[n for n in ast.walk(execute) if isinstance(n,ast.Call)]
 assert sum(isinstance(n.func,ast.Attribute) and n.func.attr=='warmup' for n in calls)==1
 assert not any(isinstance(n.func,ast.Subscript) for n in calls)
 assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('copy_','fill_','zeros','full','rand','randn','synchronize') for n in calls)
 signature={(i,):v for i,v in enumerate(('*bf16','*i32','*bf16','i32','i32'))};constants={(5,):8,(6,):128}
 good=NS(src=NS(signature=signature,constants=constants,attrs={}))
 assert c.runtime_ABI(good)['verified']
 for kind in ('folded','wrong_scalar','unknown'):
  s=dict(signature);k=dict(constants)
  if kind=='folded':k[(3,)]=2048
  if kind=='wrong_scalar':s[(4,)]='constexpr'
  if kind=='unknown':s={'unrecognized':'i32'}
  assert not c.runtime_ABI(NS(src=NS(signature=s,constants=k,attrs={ })))['verified']
 # A compact SSA-only parser fixture. It does not represent compiled hardware IR.
 fixture='''module {
 tt.func public @gather(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<bf16>, %arg3: i32, %arg4: i32) attributes {} {
 %r = tt.make_range {start = 0, end = 8} : tensor<8xi32>
 %n = tt.splat %arg4 : i32 -> tensor<8xi32>
 %rm = arith.cmpi slt, %r, %n : tensor<8xi32>
 %ip = tt.addptr %arg1, %r : tensor<8x!tt.ptr<i32>>
 %idx = tt.load %ip, %rm : tensor<8xi32>
 %idx64 = arith.extsi %idx : tensor<8xi32> to tensor<8xi64>
 %zero = arith.constant dense<0> : tensor<8xi64>
 %lower = arith.cmpi sge, %idx64, %zero : tensor<8xi64>
 %m0 = arith.extsi %arg3 : i32 to i64
 %m = tt.splat %m0 : i64 -> tensor<8xi64>
 %upper = arith.cmpi slt, %idx64, %m : tensor<8xi64>
 %cols = tt.make_range {start = 0, end = 128} : tensor<128xi32>
 %limit = arith.constant dense<2048> : tensor<128xi32>
 %cm = arith.cmpi slt, %cols, %limit : tensor<128xi32>
 %a = arith.andi %rm, %lower : tensor<8xi1>
 %b = arith.andi %a, %upper : tensor<8xi1>
 %mask = arith.andi %b, %cm : tensor<8x128xi1>
 %p = tt.addptr %arg0, %idx64, %cols : tensor<8x128x!tt.ptr<i16>>
 %payload = tt.load %p, %mask : tensor<8x128xi16>
 %op = tt.addptr %arg2, %r, %cols : tensor<8x128x!tt.ptr<i16>>
 tt.store %op, %payload, %mask : tensor<8x128x!tt.ptr<i16>>
 }
}'''
 assert c.mask_audit(fixture)['verified']
 for text in (fixture.replace('tt.load %ip, %rm','tt.load %ip'),fixture.replace('slt, %idx64, %m','sge, %idx64, %m'),fixture.replace('tt.store %op, %payload, %mask','tt.store %op, %payload, %rm'),fixture.replace('dense<2048>','dense<4096>'),fixture.replace('%arg3: i32','%arg3: i64')):
  assert not c.mask_audit(text)['verified']
 assert not c.mask_audit('unsupported backend format')['verified']
 assert not {'torch','triton','numpy'}.intersection(sys.modules)
 return dict(passed=True,CPU_only=True,cases=84,warmups_planned=168,production_kernel_source_frozen=True,
  frontend_has_only_warmup_and_handle_load_no_launch_or_initialization=True,
  runtime_ABI_parser_refuses_folded_wrong_unknown=True,SSA_mask_parser_negative_controls=True,
  synthetic_SSA_fixture_is_not_actual_compiled_IR=True,actual_SDK_compile_or_device_verified=False)

if __name__=='__main__':print(json.dumps(main(),sort_keys=True,indent=2))
