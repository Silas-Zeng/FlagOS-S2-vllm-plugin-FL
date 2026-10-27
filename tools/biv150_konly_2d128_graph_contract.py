"""Independent BM128 dynamic-length CPU reference/admission; no framework imports."""
import math
import biv150_konly_q32_numeric_contract as old

FROZEN='f04aa70773a714169c439d6771c54fe8ee72f12f'


def need(ok,message):
    if not ok:raise ValueError(message)


def changed_metadata(table,length):
    need(type(length) is int and length==6144 and len(table)==8192,'exact capacity fixture')
    need(all(type(v) is int and 0<=v<773 for v in table) and len(set(table[:384]))==384,'physical table bounds/uniqueness')
    return table[383:384]+table[:383]+table[384:],6143


def control_fingerprints(actual,cpu):
    names={'query_ptr':'q','block_tables_ptr':'table','seq_lens_ptr':'lengths'}
    for gpu_name,cpu_name in names.items():
        need(actual.get(gpu_name)==cpu[cpu_name],'actual device control differs from CPU '+cpu_name)
    return {name:actual[gpu_name] for gpu_name,name in names.items()}


def eager_admission(summary):
    need(summary['commit']==FROZEN and summary['completed'] is True and summary['errors']==[], 'completed frozen f04 eager required')
    need(summary['reader_launches']==4 and summary['encoder_launches']==0
         and summary['actual_binding_verified'] is True and summary['loaded_resources_verified'] is True,'eager execution/binding/resources')
    need(summary['precision_gate'] is False and summary['performance_measured'] is False,'component scope')
    need([r['pattern'] for r in summary['records']]==[0,1],'both eager patterns')
    for r in summary['records']:
        names={'finite','baseline_reference','CPU_codec_cache_bytes','output_canaries','read_only_inputs','actual_IR','quantized_implementation'}
        need(set(r['hard_checks'])==names and all(v is True for v in r['hard_checks'].values()),'eager hard gate failed')
        need(r['numeric_checks']['quantized_implementation_passed'] is True,'eager quantized implementation gate')
    return dict(frozen_eager_hard_gates_verified=True,original_BF16_quality_passed=summary['eager_numeric_passed'],
                no_model_or_performance_claim=True)


def references(cpu,code,scale,torch,np):
    """Same original/quantized FP64 algebra, explicitly runtime-length aware.

    Q rows/head counts remain fixed; only CPU-owned Q/table/KV-length data vary.
    No GPU tensor/scalar supplies an independent scale or reference.
    """
    rows=[0,1,15,16,31,32,1024,2047];length=int(cpu['lengths'][0]);qrows=len(cpu['q'])
    need(qrows==2048 and length in (6144,6143),'bounded BM128 changed-length reference')
    ids=cpu['table'][0,:(length+15)//16].long()
    keys=cpu['packed'][ids,0].reshape(-1,2,128)[:length].double()
    values=cpu['packed'][ids,1].reshape(-1,2,128)[:length].double()
    kc=code[ids].reshape(-1,2,128)[:length].long();ks=scale[ids].reshape(-1,2)[:length].double()
    selected=cpu['q'][rows];qc,qs,_=old.quantize_cpu(selected,torch,np)
    q=selected.double().reshape(len(rows),2,8,128).permute(1,2,0,3)
    original_score=torch.einsum('hgrd,thd->hgrt',q,keys)*128**-.5
    integer=np.einsum('hgrd,thd->hgrt',qc.long().reshape(len(rows),2,8,128).permute(1,2,0,3).numpy(),kc.numpy(),dtype=np.int64,optimize=True)
    need(int(np.max(np.abs(integer)))<=128*127**2,'INT64 QK bound')
    quant_score=torch.from_numpy(integer).double()*qs.double().reshape(len(rows),2,8).permute(1,2,0)[:,:,:,None]*ks.T[:,None,None,:]*128**-.5
    mask=torch.arange(length)[None,:]<(length-qrows)+torch.tensor(rows)[:,None]+1
    def attention(score):
        probability=score.masked_fill(~mask[None,None],-math.inf).softmax(-1)
        return torch.einsum('hgrt,thd->rhgd',probability,values).reshape(len(rows),16,128)
    return dict(rows=rows,original=attention(original_score),quantized=attention(quant_score),runtime_KV_length=length,
                quantized_reference='INT64 QK, FP64 dequantized scales/global softmax/BF16 V',
                probability_rounding_modeled=False,online_softmax_rounding_modeled=False)
