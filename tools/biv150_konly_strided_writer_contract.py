"""Stdlib component contracts; these do not certify live GPU-derived slots."""
from dataclasses import dataclass
import struct

from biv150_konly_decode_quant_reference import quantize_v2

PAGE = 16384
FORMAT = 'cjj-konly-rn32-rne-v2'
WIDTHS = {'bfloat16': 2, 'int64': 8, 'uint8': 1}


def need(ok, message):
    if not ok:
        raise ValueError(message)


@dataclass(frozen=True)
class View:
    dtype: str
    shape: tuple
    strides: tuple
    storage_base: int
    storage_bytes: int
    byte_offset: int
    device: str = 'cuda:0'

    def validate(self):
        need(type(self.dtype) is str and self.dtype in WIDTHS, 'dtype')
        need(type(self.shape) is tuple and type(self.strides) is tuple
             and len(self.shape) == len(self.strides), 'view rank')
        need(all(type(n) is int and 0 <= n < 2**63 for n in self.shape), 'view shape')
        need(all(type(n) is int and 0 < n < 2**63 for n in self.strides), 'positive strides')
        need(all(type(n) is int and n >= 0 for n in
                 (self.storage_base, self.storage_bytes, self.byte_offset)), 'storage metadata')
        width = WIDTHS[self.dtype]
        need(self.byte_offset % width == 0 and self.byte_offset <= self.storage_bytes, 'storage offset')
        elements = 0 if 0 in self.shape else 1+sum((n-1)*s for n, s in zip(self.shape, self.strides))
        need(self.byte_offset+elements*width <= self.storage_bytes, 'storage extent')
        need(self.storage_base+self.storage_bytes < 2**63, 'address range')
        need(type(self.device) is str and self.device == 'cuda:0', 'single component device')


@dataclass(frozen=True)
class SlotSnapshot:
    values: tuple
    provenance: str
    epoch: int


def overlap_storage(a, b):
    # Conservative full-backing rejection, even when strided active spans differ.
    return max(a.storage_base, b.storage_base) < min(
        a.storage_base+a.storage_bytes, b.storage_base+b.storage_bytes)


def qualify(key, value, slots, cache, pages, snapshot):
    """Qualify explicit CPU descriptors/snapshot for this standalone fixture.

    The device entry owns/uploads/checks the slot tensor. A caller cannot pass an
    unknown GPU-only mapping here and claim that its contents were inspected.
    K/V may share read-only QKV backing, but no input may alias writable cache.
    """
    need(type(pages) is int and 0 < pages <= 2**31-1, 'cache pages')
    need(type(snapshot) is SlotSnapshot and type(snapshot.provenance) is str
         and snapshot.provenance == 'component-owned-cpu-preupload'
         and type(snapshot.epoch) is int and snapshot.epoch >= 0, 'slot provenance/freshness')
    for view in (key, value, slots, cache):
        need(type(view) is View, 'explicit view')
        view.validate()
    for view in (key, value):
        need(view.dtype == 'bfloat16' and len(view.shape) == 3 and view.shape[1:] == (2, 128), 'BF16 source shape')
        need(view.strides[1:] == (128, 1) and view.strides[0] >= 256, 'source inner/row stride')
    need(slots.dtype == 'int64' and len(slots.shape) == 1 and slots.strides == (1,), 'slot view')
    count = slots.shape[0]
    need(count <= key.shape[0] and count <= value.shape[0], 'slot count exceeds K/V rows')
    need(cache.dtype == 'uint8' and cache.shape == (pages*PAGE,) and cache.strides == (1,), 'raw full cache')
    need((cache.storage_base+cache.byte_offset) % 4 == 0, 'full-base FP32 alignment')
    need(type(snapshot.values) is tuple and len(snapshot.values) == count, 'slot snapshot extent')
    need(all(type(s) is int and -2**63 <= s < 2**63 for s in snapshot.values), 'slot int64 contents')
    positive = [s for s in snapshot.values if s >= 0]
    need(all(s < pages*16 for s in positive), 'positive slot outside cache')
    need(len(positive) == len(set(positive)), 'duplicate writable slot')
    need(not any(overlap_storage(cache, src) for src in (key, value, slots)), 'source/cache alias')
    need(not any(overlap_storage(slots, src) for src in (key, value)), 'slot/source alias')
    return dict(grid=(count, 2), launch=count > 0, key_row_stride=key.strides[0],
                value_row_stride=value.strides[0], key_rows=key.shape[0], value_rows=value.shape[0],
                slot_count=count, cache_pages=pages, format=FORMAT, slot_epoch=snapshot.epoch,
                requires_gpu_slots_equal_snapshot=True, requires_finite_valid_K_external=True,
                live_gpu_mapping_certified=False)


def bf16_value(bits):
    return struct.unpack('<f', struct.pack('<I', bits << 16))[0]


def oracle(before, slot_values, logical_keys, logical_values):
    """Independent field-row reconstruction; V is opaque BF16 source bits."""
    need(len(before) % PAGE == 0, 'reference page bytes')
    pages = []
    for p in range(len(before)//PAGE):
        b = before[p*PAGE:(p+1)*PAGE]
        pages.append([bytearray(b[:4096]), bytearray(b[4096:4224]), b[4224:8192], bytearray(b[8192:])])
    for token, physical in enumerate(slot_values):
        if physical < 0:
            continue
        page, slot = divmod(physical, 16)
        for head in range(2):
            code, scale = quantize_v2([bf16_value(b) for b in logical_keys[token][head]])
            row = slot*2+head
            pages[page][0][row*128:(row+1)*128] = bytes(c & 255 for c in code)
            pages[page][1][row*4:(row+1)*4] = struct.pack('<f', scale)
            pages[page][3][row*256:(row+1)*256] = struct.pack('<128H', *logical_values[token][head])
    return b''.join(bytes(code)+bytes(scale)+reserved+bytes(value) for code, scale, reserved, value in pages)


def scatter_model(case):
    """Emulate address expressions against real pitched CPU source bytes."""
    binding = qualify(case['key'], case['value'], case['slots'], case['cache'], case['pages'], case['snapshot'])
    stores = []
    for token, physical in enumerate(case['snapshot'].values):
        if physical < 0:
            continue
        for head in range(2):
            vectors = []
            for name in ('key', 'value'):
                desc, storage = case[name], case['buffers'][name]
                offset = desc.byte_offset+2*(token*desc.strides[0]+head*128)
                vectors.append(struct.unpack_from('<128H', storage, offset))
            code, scale = quantize_v2([bf16_value(b) for b in vectors[0]])
            # Validate/quantize every valid source before mutating any destination.
            page, slot = divmod(physical, 16)
            stores += [(page*PAGE+slot*256+head*128, bytes(c & 255 for c in code)),
                       (page*PAGE+4096+slot*8+head*4, struct.pack('<f', scale)),
                       (page*PAGE+8192+slot*512+head*256, struct.pack('<128H', *vectors[1]))]
    base = case['cache'].byte_offset
    output = bytearray(b'\x5a'*case['cache'].storage_bytes)
    output[base:base+len(case['before'])] = case['before']
    for offset, data in stores:
        output[base+offset:base+offset+len(data)] = data
    need(output[:base] == b'\x5a'*base
         and output[base+len(case['before']):] == b'\x5a'*(len(output)-base-len(case['before'])), 'CPU cache canaries')
    return bytes(output[base:base+len(case['before'])]), binding
