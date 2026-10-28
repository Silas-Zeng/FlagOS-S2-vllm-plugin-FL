"""Unchanged Q32 ABI mapper (5497877b); private compile screen uses candidate=True."""
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
