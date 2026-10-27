"""Compile-only component geometry and Tensor descriptors; no framework imports."""
from biv150_konly_prefill_factory import FORMAT, PROFILES
from biv150_konly_prefill_reference import grid, validate_binding

PROFILES_SUPPORTED = ('q32', 'raw3d', 'multi2')


def plan(profile='q32'):
    if profile not in PROFILES_SUPPORTED:
        raise ValueError('Only explicit normalized-Q32/raw3D/M2 compile profiles')
    qlens = [4096] if profile == 'q32' else ([1]*64 if profile == 'raw3d' else [2048,2048])
    klens = [4096]*len(qlens)
    pages = 2*sum((n+15)//16 for n in klens)+5
    is3d, bm, bq, tile, warps, stages, _ = PROFILES[profile]
    segments = 16 if is3d else 1
    allocation=dict(two_caches=2*pages*16384,query_and_output=2*sum(qlens)*16*128*2,
                    table=len(qlens)*8192*4,lengths_and_prefix=(2*len(qlens)+1)*4,descales=8,
                    scratch_and_scalars=(sum(qlens)*16*segments*(128+2)*4 if is3d else 0))
    return dict(profile=profile, qlens=qlens, klens=klens, total=sum(qlens), sequences=len(qlens),
                pages=pages, segments=segments, grid=grid(profile,qlens,segments),
                config=dict(BLOCK_M=bm,BLOCK_Q=bq,TILE_SIZE=tile,IS_3D=is3d),
                requested_options={} if is3d else dict(num_warps=warps,num_stages=stages),
                raw3d_options='SDK defaults; actual options recorded after compile' if is3d else None,
                table_shape=(len(qlens),8192), table_pitch=8192,
                fixture='main56f prefill component: full padded table, 2*sum(page_counts)+5 pages, scalar k/v descales',
                packed_allocation_bytes=pages*16384, baseline_allocation_bytes=pages*16384,
                q_and_out_allocation_bytes=2*sum(qlens)*16*128*2,
                empty_allocation_bytes=allocation,total_empty_allocation_bytes=sum(allocation.values()),
                metadata_values_initialized=False, kernel_launches=0, capacity_reclaimed=False)


def allocate(torch, p):
    """Only empty/view/unbind/as_strided. No RNG, copy, fill, encoder or launch."""
    pages, total, sequences, segments = (p[x] for x in ('pages','total','sequences','segments'))
    def empty(shape,dtype):
        return torch.empty(shape,dtype=dtype,device='cuda')
    q=empty((total,16,128),torch.bfloat16);out=empty((total,16,128),torch.bfloat16)
    packed=empty((pages,2,16,2,128),torch.bfloat16)
    key,value=packed.unbind(1)
    raw=empty((pages*16384,),torch.uint8)
    code=torch.as_strided(raw.view(torch.int8),(pages,16,2,128),(16384,256,128,1),0)
    packed_bf16_v=torch.as_strided(raw.view(torch.bfloat16),(pages,16,2,128),(8192,256,128,1),4096)
    scale=raw.view(torch.float32)
    result=dict(q=q,out=out,baseline_k=key,baseline_v=value,k=code,v=packed_bf16_v,kscale=scale,
                table=empty((sequences,8192),torch.int32),lengths=empty((sequences,),torch.int32),
                cu_q=empty((sequences+1,),torch.int32),k_descale=empty((),torch.float32),v_descale=empty((),torch.float32))
    if p['profile']=='raw3d':
        result.update(scratch=empty((total,16,segments,128),torch.float32),
                      maxima=empty((total,16,segments),torch.float32),sums=empty((total,16,segments),torch.float32))
    # Retain allocation owners explicitly; no view can outlive the compile job.
    return result, (packed,raw)


def describe(tensor, storages):
    storage=tensor.untyped_storage();pointer=storage.data_ptr()
    identity=storages.setdefault(pointer,'allocation'+str(len(storages)))
    return dict(dtype=str(tensor.dtype).removeprefix('torch.'),shape=tuple(tensor.shape),stride=tuple(tensor.stride()),
                storage_offset=tensor.storage_offset(),storage_bytes=storage.nbytes(),storage=identity,
                device=str(tensor.device),is_contiguous=tensor.is_contiguous())


def descriptors(tensors):
    storages={}
    return {name:describe(tensor,storages) for name,tensor in tensors.items()}


def check_descriptors(p, d):
    roles=('q','out','k','kscale','v')+(('scratch','maxima','sums') if p['profile']=='raw3d' else ())
    result=validate_binding(p['profile'],format_id=FORMAT,query_lengths=p['qlens'],key_lengths=p['klens'],
                            pages=p['pages'],segments=p['segments'],tensors={r:d[r] for r in roles})
    for role in ('baseline_k','baseline_v'):
        value=d[role]
        if (value['dtype'],tuple(value['shape']),tuple(value['stride']),value['storage_offset']) != (
                'bfloat16',(p['pages'],16,2,128),(8192,256,128,1),0 if role=='baseline_k' else 4096):
            raise ValueError('Actual BF16 control cache descriptor changed')
    if d['baseline_k']['storage'] != d['baseline_v']['storage'] or d['baseline_k']['storage']==d['k']['storage']:
        raise ValueError('Independent original BF16 and encoded packed allocations required')
    expected={'table':('int32',(p['sequences'],8192),(8192,1)),
              'lengths':('int32',(p['sequences'],),(1,)), 'cu_q':('int32',(p['sequences']+1,),(1,)),
              'k_descale':('float32',(),()),'v_descale':('float32',(),())}
    for role,(dtype,shape,stride) in expected.items():
        value=d[role]
        if (value['dtype'],tuple(value['shape']),tuple(value['stride']),value['storage_offset']) != (dtype,shape,stride,0):
            raise ValueError('Actual SDK scalar/metadata descriptor changed: '+role)
    if any(not str(value['device']).startswith('cuda') for value in d.values()):
        raise ValueError('Compile opt-in must use target device Tensor metadata')
    return result


def kwargs(p,t,candidate):
    key,value=(t['k'],t['v']) if candidate else (t['baseline_k'],t['baseline_v'])
    is3d=p['profile']=='raw3d'
    result=dict(output_ptr=t['out'],query_ptr=t['q'],key_cache_ptr=key,value_cache_ptr=value,
        sink_ptr=None,block_tables_ptr=t['table'],seq_lens_ptr=t['lengths'],alibi_slopes_ptr=None,qq_bias_ptr=None,
        scale=128**-.5,q_scale=None,k_scale=t['k_descale'],v_scale=t['v_descale'],out_scale=1.0,softcap=0,
        num_query_heads=16,num_queries_per_kv=8,block_table_stride=t['table'].stride(0),
        query_stride_0=t['q'].stride(0),query_stride_1=t['q'].stride(1),output_stride_0=t['out'].stride(0),
        output_stride_1=t['out'].stride(1),qq_bias_stride_0=0,BLOCK_SIZE=16,HEAD_SIZE=128,HEAD_SIZE_PADDED=128,
        USE_ALIBI_SLOPES=False,USE_ALIBI_SQRT=False,USE_QQ_BIAS=False,USE_SOFTCAP=False,USE_SINKS=False,
        SLIDING_WINDOW=0,USE_CAUSAL=True,USE_PER_SEQ_CAUSAL=False,per_seq_causal_ptr=None,
        USE_MM_PREFIX=False,MAX_MM_RANGES=0,mm_prefix_range_ptr=None,query_start_len_ptr=t['cu_q'],
        num_seqs=p['sequences'],NUM_SEGMENTS_PER_SEQ=p['segments'],USE_FP8=False,
        segm_output_ptr=t['scratch'] if is3d else t['out'],segm_max_ptr=t['maxima'] if is3d else t['out'],
        segm_expsum_ptr=t['sums'] if is3d else t['out'],k_scale_cache_ptr=key,v_scale_cache_ptr=value,
        stride_ks_blk=0,stride_ks_slot=0,stride_ks_head=0,stride_vs_blk=0,stride_vs_slot=0,stride_vs_head=0,
        KV_QUANT_MODE=0,Q_IS_FP8=False,CHUNK_LOOKBACK=-1,CHUNK_SIZE=-1,USE_TD=False,USE_TD_QO=False,
        **p['config'])
    for prefix,tensor in (('k',key),('v',value)):
        result.update({f'stride_{prefix}_cache_{i}':tensor.stride(i) for i in range(4)})
    if candidate:
        result['key_code_scale_ptr']=t['kscale']
    return result


def reducer_kwargs(p,t):
    if p['profile']!='raw3d':
        raise ValueError('2D has no segmented reducer')
    return dict(output_ptr=t['out'],segm_output_ptr=t['scratch'],segm_max_ptr=t['maxima'],segm_expsum_ptr=t['sums'],
        seq_lens_ptr=t['lengths'],num_seqs=p['sequences'],num_query_heads=16,out_scale_inv=1.0,
        output_stride_0=t['out'].stride(0),output_stride_1=t['out'].stride(1),block_table_stride=8192,
        TILE_SIZE=16,HEAD_SIZE=128,HEAD_SIZE_PADDED=128,query_start_len_ptr=t['cu_q'],BLOCK_Q=2,
        NUM_SEGMENTS_PER_SEQ=p['segments'],USE_FP8=False)
