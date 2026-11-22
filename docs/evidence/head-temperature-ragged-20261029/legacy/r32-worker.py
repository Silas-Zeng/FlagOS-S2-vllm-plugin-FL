"""M32真实head→完整原Sampler：唯一温度epilogue组件。"""
from pathlib import Path
from types import SimpleNamespace
import hashlib,importlib,inspect,json,statistics,traceback


def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def worker(run,model):
    run,model=Path(run),Path(model)
    result=dict(status='started',cases=[],timings=[],resources=[],kind='lm_head_temperature_epilogue',
        source_main='56f1838192abc940b3fb9accef490022ac125000',rows=32,vocab=130560,
        new_persistent_weight_bytes=0,precision_qualified=False,official_qualified=False,
        limitations=['单层真实head W与合成hidden；不是模型/官方收益。','完整eager head+Sampler含实际分配/原RNG；Graph仅确定性prefix。','实际原F.linear配置首次捕获；未变原stats，失败不救旧stats融合。'])
    module=original=None
    try:
        import torch,triton,flag_gems,factory,candidate
        from safetensors import safe_open
        from reused_helpers import timed_batch
        torch.set_num_threads(4)
        prop=torch.cuda.get_device_properties(0);target=triton.runtime.driver.active.get_current_target()
        assert 'BI-V150' in prop.name and prop.multi_processor_count==16 and target.warp_size==64
        from vllm_fl.utils import get_flag_gems_whitelist_blacklist
        whitelist,blacklist=get_flag_gems_whitelist_blacklist();assert not whitelist and not blacklist
        flag_gems.enable(record=True,once=True,path=str(run/'flaggems-oplist.txt'))
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import patch_sampler_compile_for_iluvatar
        patch_sampler_compile_for_iluvatar()
        from vllm_fl.patches import biv150_runtime
        biv150_runtime._patch_top_p_sampler()
        smod=importlib.import_module('vllm.v1.sample.sampler');top=importlib.import_module('vllm.v1.sample.ops.topk_topp_sampler')
        hist=importlib.import_module('vllm_fl.patches.biv150_top_p_triton')
        from vllm.v1.sample.metadata import SamplingMetadata
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
        from vllm.platforms import current_platform
        assert current_platform.vendor_name=='iluvatar'
        receipt=json.loads((Path(__file__).parent/'installed-manifest.json').read_bytes())
        for relative,info in receipt['files'].items():assert hashlib.sha256(Path(info['source']).read_bytes()).hexdigest()==info['sha256'],relative
        assert hashlib.sha256(Path(inspect.getsourcefile(hist)).read_bytes()).hexdigest()=='da77c411a88ad4ead072c99cdc4edb52b5bef5b8e5990b9cc8e04db08ed31ca9'
        assert getattr(top.apply_top_k_top_p_triton,'_biv150_top_p_dispatch',False)
        for op in ('_to_copy','div_.Tensor','linear'):
            table=torch._C._dispatch_dump_table('aten::'+op)
            (run/('dispatch-'+op.replace('.','-')+'.txt')).write_text(table)
            assert any(line.startswith('CUDA:') and 'flag_gems/' in line.replace('\\','/') for line in table.splitlines())
        module=factory.actual_linear_module(flag_gems,run);original=module.linear_kernel
        cfg=json.loads((model/'config.json').read_bytes())
        assert cfg['vocab_size']==130560 and cfg['hidden_size']==2048 and cfg.get('logit_scale',1.0)==1.0
        index=json.loads((model/'model.safetensors.index.json').read_bytes());name='lm_head.weight'
        with safe_open(str(model/index['weight_map'][name]),framework='pt',device='cpu') as f:wc=f.get_tensor(name).contiguous()
        assert tuple(wc.shape)==(130560,2048) and wc.dtype==torch.bfloat16
        w=wc.cuda();w_before=w.clone()
        generator=torch.Generator().manual_seed(20261027)
        xc=(torch.randn((32,2048),generator=generator)*.125).bfloat16();x=xc.cuda();x_before=x.clone()
        layer=SimpleNamespace(weight=w,prefix='lm_head')
        head_method=UnquantizedLinearMethod()
        def head():return head_method.apply(layer,x,bias=None)
        capture=factory.Capture(original);module.linear_kernel=capture
        try:initial=head()
        finally:module.linear_kernel=original
        torch.cuda.synchronize();assert len(capture.records)==1 and initial.dtype==torch.bfloat16
        record=capture.records[0]
        result['identity']=dict(constants={k:str(v) for k,v in record['constants'].items()},kwargs=record['kwargs'],grid=record['grid'],linear_source_sha256=factory.LINEAR_SHA,
            head_source='actual UnquantizedLinearMethod.apply/F.linear default dispatch',weight_name=name,weight_sha256=hashlib.sha256(wc.view(torch.uint8).numpy().tobytes()).hexdigest(),output_dtype=str(initial.dtype),output_stride=list(initial.stride()))
        def preserve(label,kernel):
            result['resources'].append(dict(arm=label,hash=getattr(kernel.metadata,'hash',None),shared=getattr(kernel.metadata,'shared',None),n_regs=getattr(kernel,'n_regs',None),n_spills=getattr(kernel,'n_spills',None)))
            for kind,data in kernel.asm.items():
                if isinstance(data,(str,bytes)):(run/(label+'.'+kind)).write_bytes(data.encode() if isinstance(data,str) else data)
        preserve('original-head',record['kernel']);save(run/'result.json',result)
        temp=torch.ones(32,dtype=torch.float32,device='cuda');pp=torch.full((32,),.95,device='cuda');zeros=torch.zeros_like(temp)
        sm=SamplingMetadata(temperature=temp,all_greedy=False,all_random=True,top_p=pp,top_k=None,generators={},max_num_logprobs=None,no_penalties=True,prompt_token_ids=None,
            frequency_penalties=zeros,presence_penalties=zeros,repetition_penalties=torch.ones_like(zeros),output_token_ids=[[] for _ in range(32)],allowed_token_ids_mask=None,bad_words_token_ids={},
            logitsprocs=SimpleNamespace(argmax_invariant=[],non_argmax_invariant=[]))
        a_sampler=smod.Sampler();b_sampler=factory.PreScaledSampler(smod.Sampler(),torch)
        def bits(a,b):return a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
        def rng():return dict(cpu=torch.get_rng_state().clone(),default=torch.cuda.get_rng_state().clone(),request={i:g.get_state().clone() for i,g in sm.generators.items()})
        def reset(s):
            torch.set_rng_state(s['cpu']);torch.cuda.set_rng_state(s['default'])
            for i,v in s['request'].items():sm.generators[i].set_state(v)
        def equal_rng(a,b):return bits(a['cpu'],b['cpu']) and bits(a['default'],b['default']) and a['request'].keys()==b['request'].keys() and all(bits(a['request'][i],b['request'][i]) for i in a['request'])
        latest={}
        def scaled_b():
            out=torch.empty((32,130560),dtype=torch.float32,device=x.device)
            ck=factory.launch(candidate,record,x,w,temp,out);latest['kernel']=ck
            return out
        def complete_a():return a_sampler(head(),sm)
        def complete_b():return b_sampler(scaled_b(),sm)
        def observe(callback,sampler,out):
            old_temp=sampler.apply_temperature;old_filter=top.apply_top_k_top_p
            def filter_hook(value,k,p):
                value=old_filter(value,k,p);out['filtered']=value.clone();return value
            def temp_hook(*args):
                value=old_temp(*args);out['scaled']=value.clone();return value
            sampler.apply_temperature=temp_hook;top.apply_top_k_top_p=filter_hook
            try:return callback()
            finally:
                sampler.apply_temperature=old_temp;top.apply_top_k_top_p=old_filter
        def check(label):
            before=rng();aa={};bb={};tb=temp.clone();pb=pp.clone()
            a=observe(complete_a,a_sampler,aa);after_a=rng();reset(before)
            b=observe(complete_b,b_sampler.sampler,bb);after_b=rng();torch.cuda.synchronize()
            case=dict(label=label,scaled_bitwise=bits(aa['scaled'],bb['scaled']),filtered_bitwise=bits(aa['filtered'],bb['filtered']),ids_bitwise=bits(a.sampled_token_ids,b.sampled_token_ids),rng_equal=equal_rng(after_a,after_b),
                finite=bool(torch.isfinite(aa['scaled']).all() and torch.isfinite(bb['scaled']).all()),input_unchanged=bits(x,x_before) and bits(w,w_before) and bits(temp,tb) and bits(pp,pb))
            result['cases'].append(case);save(run/'result.json',result)
            assert all(v for k,v in case.items() if k!='label'),'逐位/RNG/输入门失败，停止不放宽'
        check('normal_real_head');preserve('candidate-head',latest['kernel'])
        # FP64独立样本只核点积→原BF16 round；跨臂全scaled/全filtered另为零容差。
        rows=[0,1,15,31];cols=[0,1,127,65535,130559]
        reference=(xc[rows].double()@wc[cols].double().T).bfloat16().double()
        measured=initial[rows][:,cols].cpu().double()
        oracle=dict(label='independent_fp64',passed=bool(torch.allclose(measured,reference,atol=.001,rtol=.01)),max_abs=float((measured-reference).abs().max()))
        result['cases'].append(oracle);assert oracle['passed'],'独立FP64门失败'
        for label,source,t in (('changed',-xc.flip(0),.8),('zero',torch.zeros_like(xc),1.),('extreme',xc*64,2.)):
            x.copy_(source.cuda());x_before.copy_(x);temp.fill_(t);check(label)
        x.copy_(xc.cuda());x_before.copy_(x);temp.fill_(1.)
        sm.generators={0:torch.Generator(device='cuda').manual_seed(19),31:torch.Generator(device='cuda').manual_seed(37)}
        check('per_request_generators');sm.generators={}
        side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):check('side_stream')
        torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize()
        # Graph只覆盖新增确定性head/cast/temperature，不捕随机Sampler。
        def prefix_a():return a_sampler.apply_temperature(head().float(),temp,True)
        graphs={};outputs={}
        for arm,callback in (('A',prefix_a),('B',scaled_b)):
            callback();torch.cuda.synchronize();graphs[arm]=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[arm]):outputs[arm]=callback()
        for ver,source,t in ((0,-xc.flip(0),.5),(1,xc,1.)):
            x.copy_(source.cuda());x_before.copy_(x);temp.fill_(t)
            for arm in 'AB':outputs[arm].fill_(float('nan'));graphs[arm].replay()
            torch.cuda.synchronize();case=dict(label='changed_graph_'+str(ver),bitwise=bits(outputs['A'],outputs['B']) and bits(prefix_a(),outputs['B']),finite=bool(torch.isfinite(outputs['B']).all()),input_unchanged=bits(x,x_before) and bits(w,w_before))
            result['cases'].append(case);assert all(v for k,v in case.items() if k!='label'),'Graph逐位门失败'
        del graphs,outputs
        x.copy_(xc.cuda());x_before.copy_(x);temp.fill_(1.)
        for _ in range(3):complete_a();complete_b()
        torch.cuda.synchronize();pair=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
        for arm in 'ABBA':result['timings'].append(dict(arm=arm,**timed_batch(torch,complete_a if arm=='A' else complete_b,pair,8)))
        check('after_timing')
        result['median']={arm:{metric:statistics.median(v[metric] for v in result['timings'] if v['arm']==arm) for metric in ('event_per_call_ms','wall_per_call_ms')} for arm in 'AB'}
        result['status']='component_one_abba_positive' if all(result['median']['B'][k]<result['median']['A'][k] for k in ('event_per_call_ms','wall_per_call_ms')) else 'nonpositive_stop'
        result['cost_includes']='完整eager原head与Sampler、实际输出分配、原统计/采样/RNG及Python提交；不含W加载，不以prefix Graph计性能'
        save(run/'result.json',result);return 0
    except BaseException:
        result.update(status='failed_stop',error=traceback.format_exc());save(run/'result.json',result);traceback.print_exc();return 1
    finally:
        if module is not None and original is not None:module.linear_kernel=original
