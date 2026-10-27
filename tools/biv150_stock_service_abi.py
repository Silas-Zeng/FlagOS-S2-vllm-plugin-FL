"""Narrow typed entry ABI check using actual target TTIR/LLVM public definitions."""
import re

EXPECTED_TTIR = [
    ('segm_output_ptr', '!tt.ptr<f32>'), ('segm_max_ptr', '!tt.ptr<f32>'),
    ('segm_expsum_ptr', '!tt.ptr<f32>'), ('query_ptr', '!tt.ptr<bf16>'),
    ('key_cache_ptr', '!tt.ptr<bf16>'), ('value_cache_ptr', '!tt.ptr<bf16>'),
    ('seq_lens_ptr', '!tt.ptr<i32>'), ('block_table_ptr', '!tt.ptr<i32>'),
    ('softmax_scale_log2e', 'f32'), ('block_table_stride', 'i64'),
    ('query_stride_token', 'i64'), ('k_stride_block', 'i64'), ('k_stride_slot', 'i64'),
    ('k_stride_head', 'i64'), ('v_stride_block', 'i64'), ('v_stride_slot', 'i64'),
]
EXPECTED_LLVM = ['ptr addrspace(1)'] * 8 + ['float'] + ['i64'] * 7 + ['ptr addrspace(1)'] * 2


def entry_abi(artifacts):
    ttir = [line for line in artifacts['ttir'].splitlines() if 'tt.func public @' in line]
    llvm = [line for line in artifacts['llir'].splitlines() if line.startswith('define iluvatar_kernel void @')]
    if len(ttir) != 1 or len(llvm) != 1:
        raise ValueError('One actual public TTIR and iluvatar LLVM kernel entry required')
    typed = re.findall(r'%([\w_]+):\s*(!tt.ptr<[^>]+>|f32|i64|i32)(?=\s|,|\))', ttir[0])
    actual = re.findall(r'(ptr addrspace\(1\)|float|i64|i32)\s+(?:[^%,]*?\s+)?%(\d+)', llvm[0])
    if typed != EXPECTED_TTIR or [t for t, _ in actual] != EXPECTED_LLVM or [int(n) for _, n in actual] != list(range(18)):
        raise ValueError('Actual stock BF16 producer typed ABI differs from audited baseline')
    return dict(ttir_parameters=typed, llvm_parameters=actual, typed_abi_match=True,
                opaque_native_mma_component_mapping_assumed=False)


def compare(baseline, candidate):
    old, new = entry_abi(baseline), entry_abi(candidate)
    if old != new:
        raise ValueError('Actual baseline/candidate typed entries differ')
    return new
