"""Read selected-config state during unscored warmups; no device or source writes.

The last snapshot completed before command.json is a conservative wider-interval
baseline: changes may include the final seconds of warmup, not just scoring.
No periodic DB reads are scheduled after the command is detected. A read begun
before command creation may overlap client startup; capture times are retained.
This sidecar does not change or restart the frozen acceptance process.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import time
import traceback


def live(pid,ticks):
    try:fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
    except FileNotFoundError:return False
    return int(fields[19])==ticks and fields[0]!='Z'


def main(a):
    assert not a.state.exists()
    spec=importlib.util.spec_from_file_location('_frozen_pair_noise',a.tools/'run_biv150_provider_noise.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert 'torch' not in sys.modules
    state=dict(diagnostic_only=True,read_only_database_access=True,device_imported=False,
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               upstream=dict(pid=a.pid,start_ticks=a.start),cases={})
    def save(phase):
        state.update(phase=phase,updated=datetime.datetime.now(datetime.timezone.utc).isoformat())
        temporary=a.state.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(a.state)
    def snapshot(folder):
        started=time.time_ns();fingerprint=module.tuning_config_fingerprint(folder/'flaggems-cache/config_cache')
        return dict(start_ns=started,end_ns=time.time_ns(),fingerprint=fingerprint)
    try:
        save('observing')
        while True:
            changed=False
            for label in ['baseline','candidate']:
                folder=a.pair/label
                for length in [4096,16384]:
                    key=label+'-'+str(length);warm=folder/f'full-warm-{length}';score=folder/f'score-{length}'
                    entry=state['cases'].setdefault(key,dict(warm_snapshots=[]))
                    command=score/'command.json'
                    if command.exists():
                        if 'command_ns' not in entry:
                            entry['command_ns']=command.stat().st_mtime_ns
                            valid=[x for x in entry['warm_snapshots'] if x['end_ns']<entry['command_ns']]
                            entry['before']=valid[-1] if valid else None
                            changed=True
                        results=[p for p in score.glob('*.json') if p.name!='command.json']
                        if results and 'after' not in entry:
                            assert len(results)==1
                            entry['after']=snapshot(folder)
                            entry['result_ns']=results[0].stat().st_mtime_ns
                            entry['selected_config_changed']=None if not entry['before'] else entry['before']['fingerprint']!=entry['after']['fingerprint']
                            entry['limitation']='Before snapshot may precede final warmup changes; selected-config changes are not isolated scoring costs.'
                            changed=True
                    elif warm.exists():
                        previous=entry['warm_snapshots'][-1]['end_ns'] if entry['warm_snapshots'] else 0
                        if time.time_ns()-previous>=5_000_000_000:
                            entry['warm_snapshots'].append(snapshot(folder));changed=True
            if changed:save('observing')
            upstream=json.loads((a.pair/'state.json').read_text())
            if upstream['phase'] in ('completed','failed'):
                save('upstream_'+upstream['phase']);break
            if not live(a.pid,a.start):
                save('upstream_not_live');break
            time.sleep(.5)
    except BaseException as error:
        state.update(error=repr(error),traceback=traceback.format_exc());save('failed');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['pair','state','tools']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--pid',type=int,required=True);parser.add_argument('--start',type=int,required=True)
    main(parser.parse_args())
