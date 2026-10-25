"""M2048 相邻 vec2 访存；逐位、变值 Graph、原完整 gate 和尾处理 ABBA。"""
import hashlib
import json
import os
from pathlib import Path
import statistics
import traceback
from reused_helpers import timed_batch

HERE=Path(__file__).resolve().parent

def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def require(ok,msg):
    if not ok: raise RuntimeError(msg)

def worker(run,model):
    result=dict(status='started',scope='随机独立M2048组件，无模型/官方资格',shape=[2048,12288,2048],cases=[],seed=20261025,
        limitations=['生产native SO未加载/未替换；两臂使用同编译器重编当前源的私有测试库。',
                    '仅M2048/N12288；逐位通过不代表其它几何或模型准确率已测。'])
    saved={}
    try:
        import torch
        from torch.utils.cpp_extension import load
        config=json.loads((model/'config.json').read_text())
        require((config['hidden_size'],config['intermediate_size'],config['num_hidden_layers'])==(2048,6144,42),'模型几何改变')
        require('BI-V150' in torch.cuda.get_device_name(),'设备不符')
        os.environ.setdefault('MAX_JOBS','2')
        os.environ.setdefault('TORCH_CUDA_ARCH_LIST','7.1')
        build=run/'build';build.mkdir()
        library=load(name='biv150_vec2_component',sources=[str(HERE/'native_w8a8.cpp'),str(HERE/'native_w8a8.cu')],
            build_directory=str(build),extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-x','ivcore'],
            is_python_module=False,verbose=True)
        result['library']=dict(path=str(library),sha256=hashlib.sha256(Path(library).read_bytes()).hexdigest(),
            torch=torch.__version__,flags=['-O3','-x','ivcore'],cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI)
        op=torch.ops.biv150_vec2_component
        gen=torch.Generator(device='cpu').manual_seed(result['seed'])
        # Explicit tensors; no framework/native cache/quantization configuration change.
        x0=torch.randn((2048,2048),generator=gen,dtype=torch.bfloat16)
        x=x0.cuda();wq=torch.randint(-127,128,(2048,12288),generator=gen,dtype=torch.int8).cuda()
        scale_cpu=torch.rand(12288,generator=gen)*0.0002+0.0001
        scales=scale_cpu.cuda()
        marker=torch.zeros((),device='cuda',dtype=torch.bfloat16).expand(12288,2048)
        ac=torch.randint(-8000000,8000001,(2048,12288),generator=gen,dtype=torch.int32)
        rc=torch.rand(2048,generator=gen)*0.01+0.0001
        cc=torch.rand(12288,generator=gen)*0.001+0.00001
        ac[0,:8]=torch.tensor([0,1,-1,127,-127,8000000,-8000000,32768],dtype=torch.int32)
        acc,rows,cols=ac.cuda(),rc.cuda(),cc.cuda()
        bufs=[torch.full((2048*6144+64,),123,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
        a,b=[buf[32:-32].view(2048,6144) for buf in bufs]
        tensors=[x,wq,scales,marker,acc,rows,cols,*bufs]
        spans=[(t.untyped_storage().data_ptr(),t.untyped_storage().nbytes()) for t in tensors]
        require(all(p+n<=q or q+m<=p for i,(p,n) in enumerate(spans) for q,m in spans[i+1:]),'组件输入/输出storage重叠')
        result['pointer_alignment']=dict(acc=acc.data_ptr()%8,cols=cols.data_ptr()%8,output=a.data_ptr()%4)
        iso={'A':lambda:op.epilogue_into(acc,rows,cols,a,False),'B':lambda:op.epilogue_into(acc,rows,cols,b,True)}
        full={'A':lambda:op.gate_silu_auto(x,wq,scales,marker,17),'B':lambda:op.gate_vec2(x,wq,scales,marker)}
        torch.cuda.synchronize()
        cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
        def equal(label,old,new,extra=None):
            torch.cuda.synchronize()
            good=torch.equal(old.view(torch.int16),new.view(torch.int16))
            finite=bool(torch.isfinite(old).all() and torch.isfinite(new).all())
            guard=all(bool((buf[:32]==123).all() and (buf[-32:]==123).all()) for buf in bufs)
            result['cases'].append(dict(name=label,bit_exact=good,finite=finite,guard=guard,**(extra or {})))
            write(run/'result.json',result)
            if not good or not finite or not guard:
                torch.save(dict(acc=acc.cpu(),rows=rows.cpu(),cols=cols.cpu(),x=x.cpu(),wq=wq.cpu(),scales=scales.cpu(),baseline=old.cpu(),candidate=new.cpu()),run/'failure.pt')
                raise RuntimeError(label+'逐位/finite/guard失败')
        iso['A']();iso['B']();equal('isolated-original',a,b)
        rsel=torch.tensor([0,1,17,1023,2047]);csel=torch.tensor([0,1,2,3,127,1023,6142,6143]);both=torch.cat((csel,csel+6144))
        linear=ac[rsel[:,None],both[None,:]].float()*(rc[rsel,None]*cc[both])
        gate,up=linear.bfloat16().float().chunk(2,-1)
        reference=(torch.nn.functional.silu(gate)*up).bfloat16()
        observed=a[rsel.cuda()[:,None],csel.cuda()[None,:]].cpu()
        torch.testing.assert_close(observed,reference,atol=0.005,rtol=0.005)
        result['independent_cpu_reference']=dict(rows=rsel.tolist(),cols=csel.tolist(),passed=True,atol=.005,rtol=.005,
            note='复用既有PR30 CPU尾处理容差；候选对原核仍要求完整bitexact')
        saved['cpu_reference']=dict(reference=reference,observed=observed)
        for pattern in ('zero','tiny','normal','large','outlier'):
            inp=x0.clone()
            if pattern=='zero':inp.zero_()
            elif pattern=='tiny':inp.mul_(1e-9)
            elif pattern=='large':inp.mul_(100)
            elif pattern=='outlier':inp[:,0]=32
            x.copy_(inp)
            old,new=full['A'](),full['B']()
            equal('full-gate-'+pattern,old,new)
        x.copy_(x0)
        # New pointer/content call is separate from stable-pointer Graph validation.
        x2=x0.flip(0).contiguous().cuda();wq2=wq.clone();scales2=scales*0.5
        equal('full-gate-new-pointers',op.gate_silu_auto(x2,wq2,scales2,marker,17),op.gate_vec2(x2,wq2,scales2,marker))
        graph_sets={}
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for label,calls in (('isolated',iso),('full-gate',full)):
                captured={}
                for arm,callback in calls.items():
                    for _ in range(3):callback()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):output=callback()
                    captured[arm]=(graph,output)
                graph_sets[label]=captured
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        # Same pointer, changed every input category used by the tested kernels.
        acc.neg_();rows.mul_(0.5);cols.mul_(2);x.mul_(0.5);scales.mul_(0.75);wq.neg_()
        for label,calls in (('isolated',iso),('full-gate',full)):
            cap=graph_sets[label]
            cap['A'][0].replay();cap['B'][0].replay()
            if label=='isolated':
                old,new=a.clone(),b.clone()
                calls['A']();calls['B']();equal('isolated-graph-vs-fresh-old',old,a);equal('isolated-graph-vs-fresh-new',new,b)
            else:
                old,new=cap['A'][1],cap['B'][1]
                equal('full-graph-vs-fresh-old',old,calls['A']());equal('full-graph-vs-fresh-new',new,calls['B']())
            equal(label+'-changed-input-graph',old,new)
        # Restore original measured values before warmup/timing.
        acc.copy_(ac);rows.copy_(rc);cols.copy_(cc);x.copy_(x0);scales.copy_(scale_cpu);wq.neg_()
        original_inputs=[t.clone() for t in (x,wq,scales,acc,rows,cols)]
        result['timings']={}
        for label,cap in graph_sets.items():
            for _ in range(5):cap['A'][0].replay();cap['B'][0].replay()
            pairs=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(4)]
            for pair in pairs:
                for event in pair:event.record()
            torch.cuda.synchronize()
            windows=[dict(arm=arm,**timed_batch(torch,cap[arm][0].replay,pair,32)) for arm,pair in zip('ABBA',pairs)]
            means={arm:statistics.mean(row['event_per_call_ms'] for row in windows if row['arm']==arm) for arm in 'AB'}
            walls={arm:statistics.mean(row['wall_per_call_ms'] for row in windows if row['arm']==arm) for arm in 'AB'}
            result['timings'][label]=dict(windows=windows,event_mean_ms=means,wall_mean_ms=walls,
                event_saved_percent=(1-means['B']/means['A'])*100,wall_saved_percent=(1-walls['B']/walls['A'])*100)
            equal(label+'-after-timing',a if label=='isolated' else cap['A'][1],b if label=='isolated' else cap['B'][1])
        result['inputs_unchanged']=all(torch.equal(old.view(torch.uint8),new.view(torch.uint8)) for old,new in zip(original_inputs,(x,wq,scales,acc,rows,cols)))
        result['rng_unchanged']=torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
        require(result['inputs_unchanged'] and result['rng_unchanged'],'输入或RNG被修改')
        torch.save(saved,run/'outputs.pt')
        result['status']='completed'
        result['decision']='仅组件信号；整体gate也正向才交主线决定模型实测，未授模型资格'
    except BaseException:
        result['status']='failed';result['error']=traceback.format_exc();traceback.print_exc()
    finally:write(run/'result.json',result)
    return 0 if result['status']=='completed' else 1
