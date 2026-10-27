"""CPU proposal for a compact K8/BF16 page. No runtime registration or GPU imports."""
from dataclasses import dataclass


FORMAT = 'cjj-k8-bf16-page12544-proposal-v1'
INSTALLED_SPEC_SHA = '7e3c220998267190a622e216b2b125306807cf8be3ec1e9a8e84fe6e7a293003'


@dataclass(frozen=True)
class CompactPage:
    tokens: int = 16
    heads: int = 2
    dim: int = 128
    alignment: int = 256

    def __post_init__(self):
        if (self.tokens, self.heads, self.dim, self.alignment) != (16, 2, 128, 256):
            raise ValueError('Only the explicit CPU proposal is admitted')

    @property
    def code_bytes(self):
        return self.tokens * self.heads * self.dim

    @property
    def scale_offset(self):
        return self.code_bytes

    @property
    def scale_bytes(self):
        return self.tokens * self.heads * 4

    @property
    def value_offset(self):
        end = self.scale_offset + self.scale_bytes
        return ((end + self.alignment - 1) // self.alignment) * self.alignment

    @property
    def value_bytes(self):
        return self.code_bytes * 2

    @property
    def page_bytes(self):
        return self.value_offset + self.value_bytes

    def address(self, role, page, slot, head, dim=0):
        if role not in ('code', 'scale', 'value'):
            raise ValueError('Unknown role')
        if any(type(n) is not int for n in (page, slot, head, dim)):
            raise ValueError('Integer coordinates are required')
        if min(page, slot, head, dim) < 0 or slot >= 16 or head >= 2 or dim >= 128:
            raise ValueError('Address outside the admitted page geometry')
        base = page * self.page_bytes
        if role == 'scale':
            if dim != 0:
                raise ValueError('One scale per token/head')
            return base + self.scale_offset + (slot * 2 + head) * 4
        element = (slot * 2 + head) * 128 + dim
        return base + (element if role == 'code' else self.value_offset + element * 2)

    def typed_views(self):
        """Relative byte geometry, with element units explicit. Not Tensor views."""
        return dict(
            code=dict(dtype='int8', page_stride_elements=self.page_bytes,
                      storage_offset_elements=0, slot_stride_elements=256),
            scale_full=dict(dtype='float32', page_stride_elements=self.page_bytes // 4,
                            storage_offset_elements=0, internal_offset_elements=self.scale_offset // 4),
            value=dict(dtype='bfloat16', page_stride_elements=self.page_bytes // 2,
                       storage_offset_elements=self.value_offset // 2, slot_stride_elements=256))

    def capacity(self, budget_bytes):
        if not isinstance(budget_bytes, int) or budget_bytes < 0:
            raise ValueError('A non-negative integer per-layer byte budget is required')
        old = budget_bytes // 16384
        new = budget_bytes // self.page_bytes
        return dict(per_layer_budget_bytes=budget_bytes, legacy_pages=old, proposed_pages=new,
                    proposed_tokens=new * 16, legacy_tokens=old * 16,
                    theoretical_page_saving_fraction=1 - self.page_bytes / 16384,
                    theoretical_page_capacity_ratio=16384 / self.page_bytes,
                    measured_capacity=False, throughput_prediction_allowed=False)


def repack_cpu_pages(legacy):
    """Bit-copy an already encoded V1 fixture; no new quantization is performed."""
    layout = CompactPage()
    if len(legacy) % 16384:
        raise ValueError('Whole original 16KiB pages are required')
    count = len(legacy) // 16384
    result = bytearray(count * layout.page_bytes)
    for page in range(count):
        src, dst = page * 16384, page * layout.page_bytes
        result[dst:dst + 4096] = legacy[src:src + 4096]
        result[dst + 4096:dst + 4224] = legacy[src + 4096:src + 4224]
        result[dst + layout.value_offset:dst + layout.page_bytes] = legacy[src + 8192:src + 16384]
    return bytes(result)
