"""Independent byte-count/alias/fail-closed checks for the static buffer audit."""
import ast
import json
from pathlib import Path
import tempfile

from audit_biv150_codegen_buffers import Audit, inspect_module


SOURCE = '''raise RuntimeError("Generated modules must never execute")
class Runner:
    def call(self,args):
        arg0_1,arg1_1,arg2_1=args
        args.clear()
        s0=arg1_1
        a=empty_strided_cuda((s0,8),(8,1),torch.bfloat16)
        b=reinterpret_tensor(a,(s0,4,2),(8,2,1),0)
        del a
        c=empty_strided_cuda((s0,8),(8,1),torch.bfloat16)
        del c
        d=torch.ops.biv150_stable_small_o.linear.default(reinterpret_tensor(b,(s0,8),(8,1),0),arg2_1)
        del b
        return (d,)
def benchmark_compiled_module():
    arg0_1=rand_strided((4,8),(8,1),device="cuda:0",dtype=torch.bfloat16)
    arg1_1=4
    arg2_1=rand_strided((8,8),(8,1),device="cuda:0",dtype=torch.bfloat16)
'''


def main():
    with tempfile.TemporaryDirectory() as directory:
        p=Path(directory)/'module.py';p.write_text(SOURCE)
        for rows in (1,4):
            result=inspect_module(p,rows)
            # Two distinct live roots, never three when b aliases a. Each
            # 8-column BF16 allocation is rows*16 bytes; 3 are created in total.
            assert result['visible_created_peak_bytes']==rows*32
            assert result['created_allocation_bytes']==rows*48
        p.write_text(SOURCE.replace('c=empty_strided_cuda','c=torch.zeros'))
        try:inspect_module(p,4)
        except ValueError:pass
        else:raise AssertionError('Unknown allocator was silently ignored')
    audit=Audit(4);audit.env['x']=audit.allocate((4,8),(8,1),'torch.bfloat16',1)
    for text in ('reinterpret_tensor(x,(4,8),(8,1),1)',
                 'reinterpret_tensor(x,(4,8),(-8,1),0)'):
        try:audit.tensor(ast.parse(text,mode='eval').body)
        except AssertionError:pass
        else:raise AssertionError('Invalid view extent accepted')
    print(json.dumps(dict(passed=True,gpu_executed=False,generated_code_never_executed=True,
                         alias_roots_counted_once=True,independent_byte_peaks=True,
                         symbolic_row_binding=True,unknown_allocator_and_invalid_views_rejected=True)))


if __name__=='__main__':main()
