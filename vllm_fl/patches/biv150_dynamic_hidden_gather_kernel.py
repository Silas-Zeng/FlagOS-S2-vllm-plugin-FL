"""Fixed-tile BF16-bit gather; row extents never specialize the JIT source."""
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['input_rows', 'selected_rows'])
def gather_hidden_bits(hidden, indices, output, input_rows: tl.int32,
                       selected_rows: tl.int32, BLOCK_R: tl.constexpr,
                       BLOCK_D: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    columns = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    index = tl.load(indices + rows, mask=rows < selected_rows, other=0).to(tl.int64)
    # All real indices are nonnegative/in-range by the CPU producer receipt.
    # Keep explicit address/tail masks; never clamp or wrap an unknown index.
    valid = (rows < selected_rows) & (index >= 0) & (index < input_rows)
    mask = valid[:, None] & (columns[None, :] < 2048)
    bits = tl.load(hidden.to(tl.pointer_type(tl.uint16))
                   + index[:, None] * 2048 + columns[None, :], mask=mask, other=0)
    tl.store(output.to(tl.pointer_type(tl.uint16))
             + rows[:, None] * 2048 + columns[None, :], bits, mask=mask)
