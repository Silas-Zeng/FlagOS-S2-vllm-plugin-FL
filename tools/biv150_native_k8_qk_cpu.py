"""CPU logical/byte model for b3 native page16 plus signed-I8 QK; no framework."""
from biv150_konly_decode_quant_reference import quantize_v2

BASE='b3b87d2a1c13448bd8c6bc24e8045c0458b8b9dd'
FORMAT='cjj-konly-rn32-rne-v2'
SDK_HPP='ddb5cd4a1d53aa2cacad501a1b2d138715c045bbf837a162d6c20a144ca45094'


def signed32(bits):
    if type(bits) is not int or not 0<=bits<2**32:raise ValueError('I32 bit pattern required')
    return bits-2**32 if bits>=2**31 else bits


def pack4(values):return sum((x&255)<<(8*i) for i,x in enumerate(values))
def unpack4(word):return tuple(((word>>(8*i))&255)-256 if (word>>(8*i))&128 else (word>>(8*i))&255 for i in range(4))


def key_byte(dim,slot):
    """Own decomposition of the observed I8 col image into64-D panels/banks."""
    if not (0<=dim<128 and 0<=slot<16):raise ValueError('page16/D128')
    panel,within=divmod(dim,64);quartet,byte=divmod(within,4);bank,subslot=divmod(slot,4)
    return panel*1024+bank*256+(quartet%4)*64+((quartet//4)^bank)*16+subslot*4+byte


def query_byte(row,dim):
    if not (0<=row<16 and 0<=dim<128):raise ValueError('M16/D128 padded query image')
    panel,column=divmod(dim,16);group,byte=divmod(row,4)
    return panel*256+((group+panel)%4)*64+column*4+byte


def a_word_index(lane,k,component):
    g,col=divmod(lane,16)
    return k*512+component*256+((g+2*k+component)%4)*64+col*4


def b_word_index(lane,k,component):
    _,col=divmod(lane,16);panel,half=divmod(k,2);em=half*2+component
    return panel*1024+(col//4)*256+(lane^(em*4))*4


def fragment_a(qcodes,lane,k,component):
    """Reuse original2-head/lane Q reads; two pair shuffles form4 head bytes."""
    g,col=divmod(lane,16);dimension=32*k+16*component+col
    groups=((2*g)%4,(2*g+1)%4)
    pairs=[(qcodes[2*s][dimension],qcodes[2*s+1][dimension]) for s in groups]
    return pack4(pairs[0]+pairs[1])


def score_coordinate(lane,component):return lane//16+4*component,lane%16


def score_scale(qscales,lane,component):
    row,_=score_coordinate(lane,component)
    # Original input pairs own(2*g,2*g+1); C owns(g,g+4,g+8,g+12).
    head=row%8;source_group,which=divmod(head,2)
    return (qscales[2*source_group],qscales[2*source_group+1])[which]


def key_address(page,slot,head,dim,pages,layer_base=0):
    if not (0<=page<pages and 0<=slot<16 and head in (0,1) and 0<=dim<128):raise ValueError('unknown/invalid address')
    return layer_base+page*16384+slot*256+head*128+dim


def scale_address(page,slot,head,pages,layer_base=0):
    if not (0<=page<pages and 0<=slot<16 and head in (0,1)):raise ValueError('unknown/invalid address')
    # FULL layer-page0 base, not a view already offset4096B.
    return layer_base+page*16384+4096+slot*8+head*4


def admit(description):
    expected=dict(format=FORMAT,key_dtype='i8',scale_dtype='f32',value_dtype='bf16',query_dtype='bf16',
        page_bytes=16384,k_strides=(16384,256,128,1),v_strides=(8192,256,128,1),
        scale_full_base=True,scale_byte_offset=4096,head_dim=128,queries_per_kv=8,page_size=16)
    def same_typed(actual, reference):
        if type(actual) is not type(reference):return False
        if isinstance(reference,tuple):
            return len(actual)==len(reference) and all(same_typed(a,b) for a,b in zip(actual,reference))
        return actual==reference
    if any(not same_typed(description.get(k),v) for k,v in expected.items()):return False
    return (any(same_typed(description.get('shape'),v) for v in ((64,2),(32,4)))
        and any(same_typed(description.get('query_stride'),v) for v in ((2048,128,1),(2560,128,1))))


def plan():
    return dict(CPU_only=True,base=BASE,format=FORMAT,profiles=[[64,2],[32,4]],
        page_bytes=16384,compact12544_supported=False,
        I8_shape=[16,16,32],block_shape=[16,16,128],logical_Q_heads=8,QK_I8_MMA_per_page=4,
        BF16_PV_MMA_per_page=8,shared_allocation_bytes=8192,K_image_bytes=2048,V_offset_bytes=4096,
        K_load_plan='ordinary coalesced global32-bit words plus proven shared scatter; no assumed SME8 builtin',
        Q_codec='same RN32/RNE v2; once per CTA, repeated across2/4 segments',
        codec_values_per_sequence_S2=4096,codec_values_per_sequence_S4=8192,
        ideal_KV_read_bytes_per_head_page=dict(BF16=8192,K8=6208),
        append_output_bytes_per_token=dict(BF16=1024,K8=776),append_common_source_read_bytes=1024,
        native_CUDA_source_implemented=False,device_compile=False,target_layout_verified=False,
        loaded_resources_verified=False,numerical_device_verified=False,performance_measured=False,
        precision_gate=False,model_verified=False,service_installed=False)
