"""Prepared device metadata/Graph gate for the private causal-skew route.

Importing this file performs no Torch import or GPU work. The caller supplies
its already validated main Q32 policy, private managers, fixture and Torch.
"""
DYNAMIC_METADATA = (
    ('short_prefix',0,2048,4096),
    ('long_prefix',0,2048,16384),
    ('prefix6144',0,2048,8192),
    ('threshold_below',0,2048,10239),
    ('threshold_equal',0,2048,10240),
    ('threshold_above',0,2048,10241),
    ('prefix10240',0,2048,12288),
    ('actual1024',0,1024,5120),
    ('actual1023_padded',0,1023,5115),
    ('actual1025_partial',0,1025,5125),
    ('nonzero_start1',1,1025,5120),
    ('nonzero_start31',31,1055,5120),
    ('return_to_short',0,2048,4096),
)


def expected_reverse(start,stop,kvlen):
    query=stop-start
    return start==0 and query>=1024 and 0<=kvlen-query<=4*query


def run_dynamic_metadata(c):
    torch,fixture=c.torch,c.fixture
    qrows,maxkv=2048,16384
    grid=(qrows//32+1,2)
    host=fixture.cpu_inputs([qrows],[maxkv],20261027,1)
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        device={name:value.cuda() for name,value in host.items()}
        key,value=device['packed'].unbind(1)
        storage=torch.full((qrows+2,16,128),43.0,device='cuda',dtype=torch.bfloat16)
        output=storage[1:-1]
        ids=torch.full((2,grid[0]),-999,device='cuda',dtype=torch.int32)
    stream.synchronize()
    def call(label):
        c.parent.wrapper=c.taps[label]
        assert c.base.holder[c.base.kernel_key] is c.policy
        c.function(q=device['q'],k=key,v=value,out=output,
            cu_seqlens_q=device['cu_q'],max_seqlen_q=qrows,
            seqused_k=device['lengths'],max_seqlen_k=maxkv,
            softmax_scale=128**-.5,causal=True,window_size=(-1,-1),block_table=device['table'],
            softcap=0,q_descale=None,k_descale=device['k_scale'],v_descale=device['v_scale'])
        assert not c.policy.disabled
    probe=c.reverse.route_probe
    graphs={}
    probe_graph=None
    initial_descriptors=None
    record=dict(completed=False,fixed_grid=list(grid),fixed_pointer_addresses=True,
        captured_once_per_route=True,route_probe_exact_generated_predicate=True,
        metadata_cse_device_verified=False,cases=[])
    c.result['dynamic_metadata']=record;c.save()
    for name,start,stop,kvlen in DYNAMIC_METADATA:
        qlen=stop-start
        active=dict(host,cu_q=torch.tensor([start,stop],dtype=torch.int32),
                    lengths=torch.tensor([kvlen],dtype=torch.int32))
        with torch.cuda.stream(stream):
            device['cu_q'].copy_(active['cu_q']);device['lengths'].copy_(active['lengths'])
            output.fill_(43.0);call('baseline')
        stream.synchronize();baseline=output.cpu().clone()
        with torch.cuda.stream(stream):
            output.fill_(43.0);call('candidate')
        stream.synchronize();candidate=output.cpu().clone()
        before=c.descriptors(c.taps['baseline'].kwargs)
        assert c.taps['baseline'].grid==c.taps['candidate'].grid==grid
        assert before==c.descriptors(c.taps['candidate'].kwargs)
        assert c.fragment._abi(c.taps['baseline'].compiled.src)==c.fragment._abi(c.taps['candidate'].compiled.src)
        if initial_descriptors is None:initial_descriptors=before
        else:assert before==initial_descriptors,'Dynamic metadata changed launch descriptors/pointers'
        canonical=dict(active,q=host['q'][start:stop],cu_q=torch.tensor([0,qlen],dtype=torch.int32))
        positions,reference=fixture.reference(canonical,[qlen],[kvlen],False,False)
        check=dict(name=name,q_start=start,actual_query_rows=qlen,kv_rows=kvlen,
            prefix_rows=kvlen-qlen,expected_order='reverse' if expected_reverse(start,stop,kvlen) else 'forward',
            direct=fixture.compare(candidate,baseline,0,0),
            reference_guard=fixture.reference_guard(candidate[start+positions],baseline[start+positions],reference),
            graphs=[])
        record['cases'].append(check);c.save()
        assert check['direct']['passed'] and check['reference_guard']['passed'],'Dynamic metadata output/reference failed'
        inactive=torch.ones(qrows,dtype=torch.bool);inactive[start:stop]=False
        assert bool(torch.all(candidate[inactive]==43.0)),'Inactive padded/nonzero-start output modified'
        assert bool(torch.isfinite(candidate[start:stop]).all())
        for label,manager in c.managers.items():
            tap=c.taps[label]
            assert tap.compiled.hash in manager.optimized,'Dynamic route did not retain Q-hoist proof'
            if label not in graphs:
                with torch.cuda.stream(stream):
                    for _ in range(3):manager.candidate[tap.grid](**tap.kwargs)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):
                        manager.candidate[tap.grid](**tap.kwargs)
                stream.synchronize();graphs[label]=graph
            with torch.cuda.stream(stream):
                output.fill_(43.0);graphs[label].replay()
            stream.synchronize()
            comparison=fixture.compare(output.cpu(),baseline,0,0)
            check['graphs'].append(dict(route=label,changed_metadata=name!='short_prefix',**comparison))
            assert comparison['passed'],'Fixed-pointer Graph did not follow changed metadata'
        with torch.cuda.stream(stream):
            if probe_graph is None:
                compiled=probe[grid](device['cu_q'],device['lengths'],ids,num_warps=4,num_stages=1)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):
                    probe[grid](device['cu_q'],device['lengths'],ids,num_warps=4,num_stages=1)
                probe_graph=graph
                record['route_probe_resources']=c.resources(compiled,c.fragment,c.out,'route-probe')
            ids.fill_(-999);probe_graph.replay()
        stream.synchronize()
        wanted=list(reversed(range(grid[0]))) if expected_reverse(start,stop,kvlen) else list(range(grid[0]))
        actual_ids=ids.cpu().tolist()
        assert actual_ids==[wanted,wanted],'Device route probe failed full-grid permutation/uniformity'
        check['device_route_ids_exact']=check['both_heads_same_permutation']=True
        check['inactive_output_canaries_unchanged']=True
        assert bool(torch.all(storage[0].cpu()==43.0)) and bool(torch.all(storage[-1].cpu()==43.0))
        for field,expected in active.items():
            assert torch.equal(device[field].cpu().contiguous().view(-1).view(torch.uint8),
                               expected.contiguous().view(-1).view(torch.uint8)),'Input changed: '+field
        check['input_bytes_unchanged']=True;c.save()
    record['launch_descriptors']=initial_descriptors
    record['completed']=True;c.save()
    del graphs,probe_graph,device,key,value,storage,output,ids
    torch.cuda.empty_cache()
    return record
