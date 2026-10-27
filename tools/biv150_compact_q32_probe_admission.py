"""Pre-CUDA refusals only; exclusive authority is Root's external assertion.

Only parse_observation is reused from the existing stdlib95 admission helper.
No host PID mapping, invented own PID or context-candidate protocol is added.
"""
import re
PRE_CUDA_MAX_MIB=1024
NO_VISIBLE=('No running processes','No running processes found','No running processes found.')
SOURCE_HELPER_SHA256='aa340dd1850ebac723115f642011aa11aa09b7bed0a66934cd93b8578744e9ee'


def need(ok,message):
    if not ok:raise ValueError('compact probe admission: '+message)


def parse_observation(full_text,pid_text):
    memory=re.findall(r'(\d+)MiB\s*/\s*(\d+)MiB',full_text)
    need(len(memory)==1,'unknown or multi-device GPU memory observation')
    used,total=map(int,memory[0]);need(0<=used<=total and total>0,'invalid GPU memory observation')
    text=pid_text.strip()
    if not text or text in NO_VISIBLE:pids=[]
    else:
        lines=text.splitlines()
        need(all(s.strip().isdigit() and int(s.strip())>0 for s in lines),'unknown native PID query')
        pids=[int(s.strip()) for s in lines]
    full_pids=[]
    if 'Processes:' in full_text:
        full_pids=[int(x) for x in re.findall(r'(?m)^\|\s*0\s+([1-9][0-9]*)\s+',full_text.split('Processes:',1)[1])]
    all_pids=sorted(set(pids+full_pids))
    return dict(memory_used_mib=used,memory_total_mib=total,visible_native_pids=all_pids,
                native_query_pids=pids,full_table_native_pids=full_pids,
                native_pid_visibility='visible_unowned' if all_pids else 'unavailable',
                no_visible_pid_is_idle_proof=False)



def pre_cuda(root_owned_exclusive,full_text,pid_text,port_closed):
    need(root_owned_exclusive is True,'Root explicit serial ownership acknowledgement required')
    state=parse_observation(full_text,pid_text)
    need(port_closed is True,'service9031 is not closed')
    need(not state['visible_native_pids'],'visible pre-existing GPU PID is not authorized')
    need(state['memory_used_mib']<=PRE_CUDA_MAX_MIB,'GPU memory above conservative pre-CUDA cap')
    return dict(state,root_owned_exclusive_acknowledged=True,port9031_closed=True,
        pre_cuda_max_used_mib=PRE_CUDA_MAX_MIB,exclusive_by_ixsmi=False,
        exclusivity_authority='Root checked prior owned job terminal and schedules serially; external assertion',
        own_driver_PID_inferred=False)
