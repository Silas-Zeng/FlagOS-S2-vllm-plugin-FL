"""M32真实head→完整原Sampler：唯一温度epilogue组件。"""
from pathlib import Path
from types import SimpleNamespace
import hashlib,importlib,inspect,json,statistics,traceback


def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def worker(run,model,row_count):
    assert type(row_count) is int and row_count in (31,41,42,43)
    run,model=Path(run),Path(model)
    result=dict(status='started',cases=[],timings=[],resources=[],kind='lm_head_temperature_epilogue',
        gpu_executed=False,source_main='a455a3de364051fbea80226870c239de3a48e716',rows=row_count,vocab=130560,
        new_persistent_weight_bytes=0,precision_qualified=False,official_qualified=False,
        limitations=['单层真实head W与合成hidden；不是模型/官方收益。','完整eager head+Sampler含实际分配/原RNG；Graph仅确定性prefix。','实际原F.linear配置首次捕获；未变原stats，失败不救旧stats融合。'])
    module=original=None
    try:
        import torch,triton,flag_gems
        import biv150_head_temperature_ragged_factory as factory
        from vllm_fl.patches import biv150_lm_head_temperature_kernel as candidate
        factory.configure(row_count)
        import vllm_fl
        package=Path(__file__).resolve().parents[1]/'vllm_fl'
        assert Path(vllm_fl.__file__).resolve()==package/'__init__.py','package import shadowed'
        assert Path(candidate.__file__).resolve()==package/'patches/biv150_lm_head_temperature_kernel.py','candidate import shadowed'
        assert candidate.head_temperature_kernel.fn.__globals__ is vars(candidate)
        assert Path(inspect.getsourcefile(candidate.head_temperature_kernel.fn)).resolve()==Path(candidate.__file__).resolve()
        result['actual_package']=dict(path=str(vllm_fl.__file__),candidate=str(candidate.__file__),candidate_sha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest())
        from safetensors import safe_open
        from biv150_head_temperature_ragged_timing import timed_batch
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
        receipt=json.loads((Path(__file__).parents[1]/'docs/evidence/head-temperature-ragged-20261029/legacy/r32-installed-manifest.json').read_bytes())
        result['actual_imports']={}
        for relative,info in receipt['files'].items():
            name=relative[:-3].replace('/','.').removesuffix('.__init__')
            imported=importlib.import_module(name);path=Path(imported.__file__).resolve()
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest==info['sha256'],relative
            result['actual_imports'][name]=dict(path=str(path),sha256=digest)
        for name,pin in [('flag_gems.utils.libentry','414e8a3f74e2522e40659e7fccd84e301f4517220837a87af0311f050bbf3025'),('triton.runtime.jit','66a168ae959e719186771e66815e807f1e449e66c5283ef596ec3eb0e35cb775')]:
            imported=importlib.import_module(name);path=Path(imported.__file__).resolve();digest=hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest==pin,name
            result['actual_imports'][name]=dict(path=str(path),sha256=digest)
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
        guarded={}
        guard_templates={}
        def buffer(name,shape,dtype):
            row_bytes=(2 if dtype==torch.bfloat16 else 4)
            for extent in shape[1:]:row_bytes*=extent
            padding=max(1,16//row_bytes)
            storage=torch.full((shape[0]+2*padding,*shape[1:]),71,dtype=dtype,device='cuda')
            view=storage[padding:-padding]
            if name not in guard_templates:
                guard_templates[name]=torch.full((padding,*shape[1:]),71,dtype=dtype,device='cpu').contiguous().view(torch.uint8).reshape(padding,-1)
            expected=guard_templates[name]
            guarded[name]=dict(storage=storage,padding=padding,prefix=expected,suffix=expected)
            return view
        def guard_check():
            record={}
            phase=len(result.setdefault('guard_records',[]))
            for name,item in guarded.items():
                authority=item['storage'].cpu().contiguous().view(torch.uint8).reshape(item['storage'].shape[0],-1)
                padding=item['padding']
                ok=torch.equal(authority[:padding],item['prefix']) and torch.equal(authority[-padding:],item['suffix'])
                record[name]=bool(ok)
                digest=hashlib.sha256(authority.numpy().tobytes()).hexdigest()
                if name=='weight':
                    expected=item.setdefault('readonly_full_sha256',digest)
                    assert digest==expected,'whole weight storage readonly失败'
                torch.save(dict(shape=list(item['storage'].shape),full_sha256=digest,prefix=authority[:padding].clone(),suffix=authority[-padding:].clone(),whole_payload_saved=False),run/(str(phase)+'-'+name+'-full-storage-authority.pt'))
            result['guard_records'].append(record);save(run/'result.json',result)
            assert all(record.values()),'CPU权威完整storage canary失败'
            return record
        w=buffer('weight',tuple(wc.shape),torch.bfloat16);w.copy_(wc.cuda());w_before=w.clone()
        result.update(gpu_executed=True,CUDA_context_initialized=bool(torch.cuda.is_initialized()),input_initialization_executed=True)
        generator=torch.Generator().manual_seed(20261027)
        xc=(torch.randn((row_count,2048),generator=generator)*.125).bfloat16();x=buffer('hidden',tuple(xc.shape),torch.bfloat16);x.copy_(xc.cuda());x_before=x.clone()
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
            assert {'ttir','ttgir','llir','asm','cubin'}<=set(kernel.asm),'实际完整IR/机器码缺失'
            for kind,data in kernel.asm.items():
                if isinstance(data,(str,bytes)):(run/(label+'.'+kind)).write_bytes(data.encode() if isinstance(data,str) else data)
            result.setdefault('typed_compilation',{})[label]=dict(signature=str(getattr(kernel.src,'signature',None)),constants=str(getattr(kernel.src,'constants',None)),metadata=str(kernel.metadata),post_launch=True)
        preserve('original-head',record['kernel']);save(run/'result.json',result)
        temp=buffer('temperature',(row_count,),torch.float32);temp.fill_(1.);pp=buffer('top_p',(row_count,),torch.float32);pp.fill_(.95);zeros=torch.zeros_like(temp)
        sm=SamplingMetadata(temperature=temp,all_greedy=False,all_random=True,top_p=pp,top_k=None,generators={},max_num_logprobs=None,no_penalties=True,prompt_token_ids=None,
            frequency_penalties=zeros,presence_penalties=zeros,repetition_penalties=torch.ones_like(zeros),output_token_ids=[[] for _ in range(row_count)],allowed_token_ids_mask=None,bad_words_token_ids={},
            logitsprocs=SimpleNamespace(argmax_invariant=[],non_argmax_invariant=[]))
        a_sampler=smod.Sampler();b_sampler=factory.PreScaledSampler(smod.Sampler(),torch)
        def bits(a,b):return a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
        def rng():return dict(cpu=torch.get_rng_state().clone(),default=torch.cuda.get_rng_state().clone(),request={i:g.get_state().clone() for i,g in sm.generators.items()})
        def reset(s):
            torch.set_rng_state(s['cpu']);torch.cuda.set_rng_state(s['default'])
            for i,v in s['request'].items():sm.generators[i].set_state(v)
        def equal_rng(a,b):return bits(a['cpu'],b['cpu']) and bits(a['default'],b['default']) and a['request'].keys()==b['request'].keys() and all(bits(a['request'][i],b['request'][i]) for i in a['request'])
        latest={}
        timing_mode=False
        def scaled_b():
            out=(torch.empty((row_count,130560),dtype=torch.float32,device=x.device) if timing_mode else buffer('candidate_output',(row_count,130560),torch.float32))
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
            case['CPU_full_storage_canaries']=all(guard_check().values())
            result['cases'].append(case);save(run/'result.json',result)
            assert all(v for k,v in case.items() if k!='label'),'逐位/RNG/输入门失败，停止不放宽'
        initial_weight=w.cpu().contiguous().view(torch.uint8)
        assert hashlib.sha256(initial_weight.numpy().tobytes()).hexdigest()==result['identity']['weight_sha256'],'GPU weight bytes differ from original CPU W'
        check('normal_real_head');preserve('candidate-head',latest['kernel'])
        # 保留旧FP64阈值，将原4行扩至全部真实行×原5列；跨臂全scaled/filtered仍零容差。
        rows=list(range(row_count));cols=[0,1,127,65535,130559]
        reference=(xc[rows].double()@wc[cols].double().T).bfloat16().double()
        measured=initial[rows][:,cols].cpu().double()
        oracle=dict(label='independent_fp64',passed=bool(torch.allclose(measured,reference,atol=.001,rtol=.01)),max_abs=float((measured-reference).abs().max()))
        result['cases'].append(oracle);assert oracle['passed'],'独立FP64门失败'
        for label,source,t in (('changed',-xc.flip(0),.8),('zero',torch.zeros_like(xc),1.),('extreme',xc*64,2.)):
            x.copy_(source.cuda());x_before.copy_(x);temp.fill_(t);check(label)
        x.copy_(xc.cuda());x_before.copy_(x);temp.fill_(1.)
        temp.copy_(torch.linspace(.5,1.5,row_count,dtype=torch.float32).cuda());check('row_varying_temperature');temp.fill_(1.)
        sm.generators={0:torch.Generator(device='cuda').manual_seed(19),row_count-1:torch.Generator(device='cuda').manual_seed(37)}
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
            case['CPU_full_storage_canaries']=all(guard_check().values())
            result['cases'].append(case);assert all(v for k,v in case.items() if k!='label'),'Graph逐位门失败'
        del graphs,outputs
        x.copy_(xc.cuda());x_before.copy_(x);temp.fill_(1.)
        timing_mode=True
        for _ in range(3):complete_a();complete_b()
        torch.cuda.synchronize();pair=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
        for arm in 'ABBA':result['timings'].append(dict(arm=arm,**timed_batch(torch,complete_a if arm=='A' else complete_b,pair,8)))
        timing_mode=False
        check('after_timing')
        result['median']={arm:{metric:statistics.median(v[metric] for v in result['timings'] if v['arm']==arm) for metric in ('event_per_call_ms','wall_per_call_ms')} for arm in 'AB'}
        result['requested_component_gates_passed']=True
        result['fresh_IR_saved']=True
        result['status']='component_one_abba_positive' if all(result['median']['B'][k]<result['median']['A'][k] for k in ('event_per_call_ms','wall_per_call_ms')) else 'nonpositive_stop'
        result['cost_includes']='完整eager原head与Sampler、实际输出分配、原统计/采样/RNG及Python提交；不含W加载，不以prefix Graph计性能'
        save(run/'result.json',result);return 0
    except BaseException:
        result.update(status='failed_stop',error=traceback.format_exc());save(run/'result.json',result);traceback.print_exc();return 1
    finally:
        if module is not None and original is not None:module.linear_kernel=original
