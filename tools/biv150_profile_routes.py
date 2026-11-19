"""Read existing CPU counters at profile boundaries; no imports or launch hooks.

An installation message alone does not prove the accepted attention policy was
used inside a trace. These snapshots neither inspect tensors nor query CUDA.
"""
import os
import sys

MODULES = {
    'qfragment': 'vllm_fl.patches.biv150_q_fragment_service',
    'q32': 'vllm_fl.patches.biv150_prefill_q32',
    'k128': 'vllm_fl.patches.biv150_prefill_k128',
}
COUNTERS = ('selected_calls', 'fallback_calls', 'original_3d_calls',
            'selected_compiles', 'fallback_compiles')


def data(value):
    try:
        return vars(value)
    except TypeError:
        return {}


def snapshot(worker=None, *, modules=None):
    modules = sys.modules if modules is None else modules
    managers = {key: data(modules.get(name)).get('_MANAGER')
                for key, name in MODULES.items()}
    base = data(managers['qfragment'])
    policy = base.get('holder', {}).get(base.get('kernel_key'))
    bindings = {}
    # Current policy -> previous policy -> original Q-fragment launcher.
    for _ in range(4):
        fields = data(policy)
        manager = fields.get('manager')
        if manager is None:
            break
        bindings[id(manager)] = fields.get('disabled', False)
        policy = fields.get('base')
    records = {}
    for key, manager in managers.items():
        fields = data(manager)
        records[key] = {
            'present': manager is not None,
            'identity': id(manager) if manager is not None else None,
            'active': fields.get('active') is True,
            'bound': manager is not None and id(manager) in bindings,
            'policy_disabled': bindings.get(id(manager)),
            'counters': {name: fields.get(name) for name in COUNTERS},
            'record_errors': list(fields.get('record_errors', [])),
            'selected_kernel_hashes': sorted(fields.get('optimized', {})),
        }
    runner = data(worker).get('model_runner')
    state = data(runner).get('_biv150_compute_order_buffers')
    c3 = dict(runner_identity=id(runner) if runner is not None else None,
              active=isinstance(state, dict), rejected=state is False,
              selected=state.get('selected') if isinstance(state, dict) else None)
    return dict(pid=os.getpid(), managers=records, compute_order=c3)


def validate(before, after, required_route):
    errors, changes = [], {}
    if required_route not in ('q32', 'k128'):
        raise ValueError('Explicit current production route required')
    if before.get('pid') != after.get('pid'):
        errors.append('Worker changed during profiling')
    for key in ('qfragment', 'q32', *(['k128'] if required_route == 'k128' else [])):
        left = before.get('managers', {}).get(key, {})
        right = after.get('managers', {}).get(key, {})
        if (not left.get('present') or not right.get('present')
                or left.get('identity') != right.get('identity')
                or left.get('active') is not True or right.get('active') is not True
                or left.get('bound') is not True or right.get('bound') is not True
                or left.get('policy_disabled') is not False
                or right.get('policy_disabled') is not False):
            errors.append(key + ': missing, replaced, unbound or disabled policy')
        if right.get('record_errors') != []:
            errors.append(key + ': compile evidence errors')
        changes[key] = {}
        for counter in COUNTERS:
            start = left.get('counters', {}).get(counter)
            end = right.get('counters', {}).get(counter)
            if type(start) is not int or type(end) is not int or start < 0 or end < start:
                errors.append(key + ': invalid or decreasing counter ' + counter)
            else:
                changes[key][counter] = end - start
    if changes[required_route].get('selected_calls', 0) <= 0:
        errors.append(required_route + ': no selected launch inside profile window')
    c3_before = before.get('compute_order', {})
    c3_after = after.get('compute_order', {})
    start, end = c3_before.get('selected'), c3_after.get('selected')
    if (c3_before.get('active') is not True or c3_after.get('active') is not True
            or c3_before.get('runner_identity') != c3_after.get('runner_identity')
            or type(start) is not int or type(end) is not int or start < 0 or end <= start):
        errors.append('Natural compute-order path not active inside profile window')
    return dict(passed=not errors, required_route=required_route,
                manager_counter_deltas=changes, errors=errors)
