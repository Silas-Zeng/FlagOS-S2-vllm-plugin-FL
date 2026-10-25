#!/usr/bin/env python3
"""Offline, standard-library-only analysis of BI-V150 worker diagnostic JSON.

No GPU or service interaction. Workload identity comes only from driver window
validation files. All node times are instrumented event intervals, never native
kernel latency or additive wall time. Run:
  python analyze_biv150_node_profiles.py --run-dir RUN --out NEW_DIRECTORY
The run directory can be a local mirror retaining observations/ and validation
files. Source paths in Triton IR are read only if the recorded files exist.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
import statistics


def read_json(path):
    return json.loads(Path(path).read_text())


def median(values):
    return statistics.median(values) if values else None


def distribution(values):
    return dict(count=len(values), median=median(values),
                min=min(values) if values else None, max=max(values) if values else None)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, label, *, positive=False):
    require(isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) and (value > 0 if positive else value >= 0),
            f"{label}: invalid numeric value {value!r}")
    return float(value)


def topology(metadata):
    """Validate original DAG and distinguish permitted from observed concurrency."""
    nodes = metadata['nodes']
    ids = [n['node_index'] for n in nodes]
    require(len(set(ids)) == len(ids) and bool(ids), 'Empty or duplicate graph node index')
    edges = [tuple(e) for e in metadata['original_edges']]
    require(len(set(edges)) == len(edges), 'Duplicate graph edge')
    incoming = {i: set() for i in ids}
    outgoing = {i: set() for i in ids}
    for a, b in edges:
        require(a in outgoing and b in incoming and a != b, 'Invalid graph edge')
        incoming[b].add(a)
        outgoing[a].add(b)
    for n in nodes:
        require(set(n['dependencies']) == incoming[n['node_index']], 'Node dependencies mismatch')
        require(set(n['successors']) == outgoing[n['node_index']], 'Node successors mismatch')
    degree = {i: len(v) for i, v in incoming.items()}
    ready = [i for i in ids if not degree[i]]
    heapq.heapify(ready)
    order, first_incomparable = [], None
    while ready:
        if len(ready) > 1 and first_incomparable is None:
            first_incomparable = sorted(ready)[:2]
        node = heapq.heappop(ready)
        order.append(node)
        for following in sorted(outgoing[node]):
            degree[following] -= 1
            if degree[following] == 0:
                heapq.heappush(ready, following)
    require(len(order) == len(ids), 'Original graph has a cycle')
    position = {i: j for j, i in enumerate(order)}
    reach = {}
    for node in reversed(order):
        mask = 0
        for following in outgoing[node]:
            mask |= (1 << position[following]) | reach[following]
        reach[node] = mask
    kernels = sorted((n for n in nodes if n['type'] == 0), key=lambda n: position[n['node_index']])
    kernel_incomparable = None
    for a, b in zip(kernels, kernels[1:]):
        if not reach[a['node_index']] & (1 << position[b['node_index']]):
            kernel_incomparable = [a['kernel_index'], b['kernel_index']]
            break
    return dict(node_count=len(nodes), edge_count=len(edges), kernel_count=len(kernels),
                original_dag_acyclic=True, all_nodes_totally_ordered=first_incomparable is None,
                simple_chain=(len(edges) == len(ids) - 1 and
                              all(len(v) <= 1 for v in incoming.values()) and
                              all(len(v) <= 1 for v in outgoing.values())),
                kernels_totally_ordered=kernel_incomparable is None,
                incomparable_node_witness=first_incomparable,
                incomparable_kernel_witness=kernel_incomparable,
                parallel_execution_permitted=first_incomparable is not None,
                parallel_execution_proven=False,
                node_type_counts=dict(collections.Counter(n['type_name'] for n in nodes)),
                undecomposed_child_graphs=sum(n['type'] == 4 for n in nodes))


def interval_geometry(rows):
    """Union/overlap of measured event windows; not kernel occupancy or wall time."""
    intervals = sorted((r['start_ms'], r['end_ms']) for r in rows)
    union, end = 0.0, None
    events = []
    for start, stop in intervals:
        if stop <= start:
            continue
        union += stop - max(start, end if end is not None else start) if end is None or stop > end else 0
        end = max(stop, end if end is not None else stop)
        events.extend(((start, 1), (stop, -1)))
    current = maximum = 0
    for _, change in sorted(events):  # touching endpoints do not overlap
        current += change
        maximum = max(maximum, current)
    return dict(instrumented_interval_sum_ms=sum(r['elapsed_ms'] for r in rows),
                instrumented_window_union_ms=union,
                instrumented_window_overlap_excess_ms=sum(b-a for a, b in intervals) - union,
                max_overlapping_event_windows=maximum,
                zero_duration_intervals=sum(r['elapsed_ms'] == 0 for r in rows))


def binding_index(bindings):
    result = collections.defaultdict(list)
    for binding in bindings:
        handle = int(binding['function_handle'])
        if binding not in result[handle]:
            result[handle].append(binding)
    return result


def binding_sources(candidates, read_sources):
    locations, files_read, unavailable = set(), [], []
    if read_sources:
        for binding in candidates:
            for path in binding.get('files', {}).values():
                source = Path(path)
                if source.suffix not in ('.ttir', '.ttgir', '.llir', '.source', '.json'):
                    continue
                if not source.is_file():
                    unavailable.append(path)
                    continue
                try:
                    with source.open('r', errors='replace') as stream:
                        content = stream.read(2 * 1024 * 1024)
                    files_read.append(path)
                    for filename, line, column in re.findall(r'loc\("([^"\n]+)":(\d+):(\d+)\)', content):
                        locations.add((filename, int(line), int(column)))
                except OSError:
                    unavailable.append(path)
    return dict(binding_status='unmatched' if not candidates else
                'unique_handle_match' if len(candidates) == 1 else 'ambiguous_handle_match',
                bindings=candidates, source_locations=[dict(path=p, line=l, column=c)
                                                       for p, l, c in sorted(locations)],
                source_files_read=sorted(set(files_read)),
                source_files_unavailable=sorted(set(unavailable)))


def workload_windows(run_dir):
    """Validation path is authoritative; filename epoch is never a workload key."""
    records, seen = [], set()
    for validation_path in sorted(run_dir.glob('diagnostic-*-profile-validation.json')):
        match = re.fullmatch(r'diagnostic-(4096|16384)-profile-validation\.json', validation_path.name)
        if not match:
            continue
        length = int(match[1])
        validation = read_json(validation_path)
        for entry in validation.get('windows', []):
            recorded = Path(entry['path'])
            local = run_dir / 'observations' / recorded.name
            # Prefer the exact mirror location, never another run's same basename.
            path = local if local.is_file() else recorded
            require(path.is_file(), f'Worker window missing: {entry["path"]}')
            identity = str(path.resolve())
            require(identity not in seen, f'Window assigned to multiple validations: {path}')
            seen.add(identity)
            records.append(dict(path=path, input_length=length, record=entry,
                                validation_path=str(validation_path), validation=validation))
    return records


def group_ranking(rows, graph_nodes, mode):
    """Rank medians of per-sample interval sums; sums explicitly not wall time."""
    groups = collections.defaultdict(list)
    for row in rows:
        group = row['name'] if mode == 'name' else (row['pid'], row['window'], row['function_handle'])
        groups[group].append(row)
    result = []
    for group, values in groups.items():
        by_sample = collections.defaultdict(float)
        nodes = set()
        mapping = {}
        for row in values:
            by_sample[row['sample_id']] += row['elapsed_ms']
            nodes.add((row['window'], row['graph_id'], row['kernel_index']))
            mapping[(row['pid'], row['window'], row['function_handle'])] = row['source_mapping']
        static_nodes = [n for n in graph_nodes if (n['name'] == group if mode == 'name'
                        else (n['pid'], n['window'], n['function_handle']) == group)]
        result.append(dict(group_by=mode, name=group if mode == 'name' else None,
                           pid=group[0] if mode == 'function' else None,
                           window=group[1] if mode == 'function' else None,
                           function_handle=group[2] if mode == 'function' else None,
                           driver_names=sorted(set(r['name'] for r in values)),
                           category='unclassified', sampled_interval_count=len(values),
                           distinct_node_count=len(nodes), observed_sample_count=len(by_sample),
                           nodes_per_graph=dict(collections.Counter(n['graph_id'] for n in static_nodes)),
                           scheduled_graph_kernel_invocation_estimate=sum(n['graph_replay_count'] for n in static_nodes),
                           invocation_estimate_excludes_extra_diagnostic_replays=True,
                           interval_ms=distribution([r['elapsed_ms'] for r in values]),
                           per_sample_interval_sum_ms=distribution(list(by_sample.values())),
                           source_mappings=[dict(pid=p, window=w, function_handle=f, **m)
                                            for (p, w, f), m in mapping.items()]))
    result.sort(key=lambda r: (-r['per_sample_interval_sum_ms']['median'], str(r['name']), str(r['function_handle'])))
    for rank, item in enumerate(result, 1):
        item['rank'] = rank
    return result


def analyze(run_dir, *, read_sources=True):
    report = dict(schema_version=1, diagnostic_only=True, official_score=False,
                  run_directory=str(run_dir), errors=[], warnings=[], windows=[], graphs=[],
                  samples=[], kernel_intervals=[], graph_summaries=[], rankings=[])
    limitations = [
        'Node intervals include inserted timing nodes/dependencies; they are not native kernel latency.',
        'Full minus boundary and ratios are paired fixed-order instrumentation diagnostics, not unbiased overhead.',
        'Boundary also contains timing events; there is no uninstrumented graph GPU total in these JSON files.',
        'Interval sums and event-window unions must not be reported as native GPU wall time.',
        'DAG comparability describes allowed concurrency; event-window overlap does not prove kernel concurrency.',
        'Selected diagnostic steps repeat computation and synchronize; throughput is not a score.',
        'Scheduler metadata is factual. Cached/new membership never classifies decode/prefill.',
        'Sample GPU offsets are graph-relative; scheduler monotonic_ns is not an absolute GPU timeline anchor.',
        'Function handles are process-local and only matched within their own worker window.',
        'Call estimates count graph nodes times scheduled replay counts, excluding extra diagnostic replays.',
        'Rankings are per graph, not workload-wide hotspots; sampled shapes/steps exclude eager work and unsampled graphs.',
    ]
    report['limitations'] = limitations
    specifications = workload_windows(run_dir)
    report['available_workloads'] = sorted({s['input_length'] for s in specifications})
    report['missing_workloads'] = sorted({4096, 16384} - set(report['available_workloads']))
    report['complete_workload_pair'] = not report['missing_workloads']
    if report['missing_workloads']:
        report['warnings'].append('Partial run: missing mapped workload lengths ' + str(report['missing_workloads']))
    manifest = run_dir / 'manifest.json'
    if manifest.is_file():
        report['manifest'] = read_json(manifest)
    if not specifications:
        report['errors'].append('No driver-validated worker windows found; workload identity cannot be inferred')
    mapped = {str(s['path'].resolve()) for s in specifications}
    report['unmapped_worker_files'] = [str(p) for p in sorted((run_dir / 'observations').glob('worker-*-window-*.json'))
                                      if str(p.resolve()) not in mapped]
    if report['unmapped_worker_files']:
        report['warnings'].append('Unmapped worker files excluded; no epoch-based workload inference')
    source_cache = {}
    graph_nodes = []
    for spec in specifications:
        checkpoints = {key: len(report[key]) for key in ('windows', 'graphs', 'samples', 'kernel_intervals')}
        node_checkpoint = len(graph_nodes)
        path, length = spec['path'], spec['input_length']
        window = read_json(path)
        window_id = path.name
        context = f'{length}/{window_id}'
        try:
            require(window['pid'] == spec['record']['pid'] and window['epoch'] == spec['record']['epoch'],
                    'Driver validation pid/epoch mismatch')
            require(window.get('diagnostic_only') is True and window.get('scored_throughput') is False,
                    'Window diagnostic flags not established')
            require(spec['validation'].get('passed') is True and spec['validation'].get('errors') == [],
                    'Driver window validation failed')
            require(window.get('errors') == [], 'Worker graph errors are present')
            require(window.get('kernel_binding_errors') == [], 'Triton binding callback errors are present')
            scopes = window['scope_diagnostics']
            require(scopes.get('errors') == [] and scopes.get('error_total') == 0
                    and scopes.get('callback_disabled') is False, 'Scope/callback integrity failed')
            scheduled = {s['step']: s for s in window['scheduled_steps']}
            require(len(scheduled) == len(window['scheduled_steps']), 'Duplicate scheduler step')
            require(len(window['samples']) == spec['record']['samples'], 'Validation sample count differs')
            require(len(scheduled) == spec['record']['scheduled_steps'], 'Validation step count differs')
            graphs = {g['key']: g['metadata'] for g in window['graphs']}
            require(len(graphs) == len(window['graphs']), 'Duplicate graph key')
            bindings = binding_index(window['triton_kernel_bindings'])
            graph_ids, node_maps = {}, {}
            for key, metadata in graphs.items():
                require(metadata.get('original_graph_unchanged') is True
                        and metadata.get('original_edges_retained') is True, 'Original graph preservation unproven')
                graph_id = f'{window_id}:{hashlib.sha256(key.encode()).hexdigest()[:12]}'
                graph_ids[key] = graph_id
                dag = topology(metadata)
                nodes = {n['kernel_index']: n for n in metadata['nodes'] if n['type'] == 0}
                require(len(nodes) == dag['kernel_count'] == metadata['kernel_count'], 'Kernel count/index mismatch')
                require(set(metadata['selected_kernel_indices']) == set(nodes), 'Full probe does not cover every top-level kernel')
                node_maps[key] = nodes
                replay_count = window['graph_replay_counts'][key]
                require(isinstance(replay_count, int) and replay_count > 0, 'Invalid graph replay count')
                report['graphs'].append(dict(input_length=length, window=window_id, pid=window['pid'],
                    graph_id=graph_id, key=key, graph_replay_count=replay_count, topology=dag, metadata=metadata))
                for n in nodes.values():
                    handle = int(n['function_handle'])
                    cache_key = (window_id, handle)
                    if cache_key not in source_cache:
                        source_cache[cache_key] = binding_sources(bindings.get(handle, []), read_sources)
                    graph_nodes.append(dict(input_length=length, window=window_id, graph_id=graph_id,
                        pid=window['pid'], graph_replay_count=replay_count,
                        source_mapping=source_cache[cache_key], **n))
            report['windows'].append(dict(input_length=length, path=str(path), pid=window['pid'],
                epoch=window['epoch'], validation_path=spec['validation_path'],
                validation_record=spec['record'], scheduled_steps=window['scheduled_steps'],
                graph_replay_counts=window['graph_replay_counts'],
                sampled_graph_keys=list(graphs),
                unsampled_graph_keys=sorted(set(window['graph_replay_counts']) - set(graphs)),
                total_scheduled_graph_replays=sum(window['graph_replay_counts'].values()),
                scheduled_replays_of_sampled_graphs=sum(window['graph_replay_counts'][key] for key in graphs),
                eager_work_measured=False))
            seen_samples = set()
            for index, sample in enumerate(window['samples']):
                key = sample['key']
                sample_id = f'{window_id}:sample-{index}'
                require(key in graphs, 'Sample has no graph metadata')
                sample_identity = (key, sample['graph_replay'])
                require(sample_identity not in seen_samples, 'Duplicate graph/replay sample')
                seen_samples.add(sample_identity)
                require(isinstance(sample['graph_replay'], int) and
                        0 < sample['graph_replay'] <= window['graph_replay_counts'][key],
                        'Sample replay index outside observed graph replay count')
                scheduler = sample.get('scheduler')
                require(isinstance(scheduler, dict) and scheduler.get('step') in scheduled,
                        f'{sample_id}: missing exact scheduler association')
                step = scheduled[scheduler['step']]
                require(all(step.get(k) == v for k, v in scheduler.items()), 'Sample scheduler metadata mismatch')
                require(scheduler.get('phase_inferred') is False, 'Unexpected inferred scheduler phase')
                require(sum(scheduler['num_scheduled_tokens'].values()) == scheduler['total_num_scheduled_tokens']
                        == scheduler['scheduled_token_sum'], 'Scheduler token totals disagree')
                require(sample.get('output_alias_audit', {}).get('validated') is True, 'Output alias audit absent')
                require(all(sample.get('original_control', {}).get(k) is True for k in
                            ('original_poison_recovery', 'numeric_exact', 'finite')), 'Original poison control absent')
                observations = {o['variant']: o for o in sample['observations']}
                require(set(observations) == {'boundary', 'full'} and len(sample['observations']) == 2,
                        'Boundary/full pair missing or duplicated')
                for observation in observations.values():
                    require(all(observation.get(k) is True for k in
                        ('output_poisoned', 'numeric_exact', 'finite', 'all_api_calls_succeeded', 'diagnostic_only')),
                        'Clone numeric/API verification failed')
                boundary, full = observations['boundary'], observations['full']
                boundary_ms = number(boundary['graph_total_ms'], 'boundary total', positive=True)
                full_ms = number(full['graph_total_ms'], 'full total', positive=True)
                require(boundary['kernels'] == [], 'Boundary probe unexpectedly has per-kernel events')
                nodes = node_maps[key]
                require(len(full['kernels']) == len(nodes) and
                        {r['kernel_index'] for r in full['kernels']} == set(nodes), 'Full interval indices differ from graph')
                rows = []
                for raw in full['kernels']:
                    n = nodes[raw['kernel_index']]
                    start = number(raw['start_ms'], 'kernel start')
                    end = number(raw['end_ms'], 'kernel end')
                    elapsed = number(raw['elapsed_ms'], 'kernel elapsed')
                    tolerance = max(.02, full_ms * 1e-5)
                    require(start <= end and end <= full_ms + tolerance
                            and abs(end - start - elapsed) <= tolerance, 'Invalid event interval bounds')
                    rows.append(dict(input_length=length, window=window_id, pid=window['pid'],
                        graph_id=graph_ids[key], sample_id=sample_id, graph_replay=sample['graph_replay'],
                        scheduler_step=scheduler['step'], kernel_index=raw['kernel_index'],
                        node_index=n['node_index'], name=n['name'], function_handle=n['function_handle'],
                        grid=n['grid'], block=n['block'], shared_mem_bytes=n['shared_mem_bytes'],
                        start_ms=start, end_ms=end, elapsed_ms=elapsed, category='unclassified',
                        source_mapping=source_cache[(window_id, int(n['function_handle']))]))
                report['kernel_intervals'].extend(rows)
                report['samples'].append(dict(input_length=length, window=window_id,
                    graph_id=graph_ids[key], sample_id=sample_id, graph_replay=sample['graph_replay'],
                    descriptor=sample['descriptor'], scheduler=scheduler,
                    scheduler_begin_monotonic_ns=step['monotonic_ns'],
                    boundary_graph_total_ms=boundary_ms, full_graph_total_ms=full_ms,
                    paired_full_minus_boundary_ms=full_ms-boundary_ms,
                    paired_full_over_boundary=full_ms/boundary_ms,
                    paired_added_timing_percent=100*(full_ms/boundary_ms-1),
                    output_alias_audit=sample['output_alias_audit'], original_control=sample['original_control'],
                    interval_geometry=interval_geometry(rows)))
        except (KeyError, TypeError, ValueError) as error:
            # A failed window contributes no rankings, even if earlier samples passed.
            for key, checkpoint in checkpoints.items():
                del report[key][checkpoint:]
            del graph_nodes[node_checkpoint:]
            report['errors'].append(f'{context}: {error}')
    report['graph_nodes'] = graph_nodes
    for graph in report['graphs']:
        samples = [s for s in report['samples'] if s['graph_id'] == graph['graph_id']]
        rows = [r for r in report['kernel_intervals'] if r['graph_id'] == graph['graph_id']]
        nodes = [n for n in graph_nodes if n['graph_id'] == graph['graph_id']]
        summary = dict(input_length=graph['input_length'], graph_id=graph['graph_id'],
                       topology=graph['topology'], graph_replay_count=graph['graph_replay_count'])
        for metric in ('boundary_graph_total_ms', 'full_graph_total_ms', 'paired_full_minus_boundary_ms',
                       'paired_full_over_boundary', 'paired_added_timing_percent'):
            summary[metric] = distribution([s[metric] for s in samples])
        report['graph_summaries'].append(summary)
        for mode in ('name', 'function'):
            report['rankings'].append(dict(scope='graph', input_length=graph['input_length'],
                graph_id=graph['graph_id'], group_by=mode, rows=group_ranking(rows, nodes, mode)))
    report['valid_workloads'] = sorted({w['input_length'] for w in report['windows']})
    report['complete_valid_pair'] = report['valid_workloads'] == [4096, 16384] and not report['errors']
    report['analysis_passed'] = not report['errors']
    return report


def csv_file(path, rows):
    if not rows:
        path.write_text('')
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def markdown(report):
    lines = ['# BI-V150 graph event diagnostics', '',
             f"Analysis integrity: **{'PASS' if report['analysis_passed'] else 'FAIL'}**. Complete valid 4K/16K pair: **{report['complete_valid_pair']}**. Diagnostic only; no official throughput claim.", '',
             'All node durations below include instrumentation. The full/boundary delta is a fixed-order diagnostic comparison.', '',
             '| Input | Graph | Samples | Boundary median ms | Full median ms | Paired added timing median % | DAG kernels ordered |',
             '|---:|---|---:|---:|---:|---:|---|']
    for g in report['graph_summaries']:
        def fmt(metric):
            value = g[metric]['median']
            return '—' if value is None else f'{value:.6f}'
        lines.append(f"| {g['input_length']} | {g['graph_id']} | {g['boundary_graph_total_ms']['count']} | "
                     f"{fmt('boundary_graph_total_ms')} | {fmt('full_graph_total_ms')} | "
                     f"{fmt('paired_added_timing_percent')} | {g['topology']['kernels_totally_ordered']} |")
    for ranking in report['rankings']:
        if ranking['group_by'] != 'name':
            continue
        lines += ['', f"## Input {ranking['input_length']}, graph {ranking['graph_id']}: top driver names", '',
                  'Ranked by median per-sample **sum of instrumented intervals**, not native wall time.', '',
                  '| Rank | Driver kernel name | Timed calls | Median interval µs | Median sample interval sum ms |',
                  '|---:|---|---:|---:|---:|']
        for r in ranking['rows'][:20]:
            name = r['name'].replace('|', '\\|').replace('\n', ' ')
            lines.append(f"| {r['rank']} | `{name}` | {r['sampled_interval_count']} | "
                         f"{1000*r['interval_ms']['median']:.3f} | {r['per_sample_interval_sum_ms']['median']:.6f} |")
    for title, key in [('Errors', 'errors'), ('Warnings', 'warnings'), ('Interpretation limits', 'limitations')]:
        if report[key]:
            lines += ['', f'## {title}', ''] + ['- ' + str(v) for v in report[key]]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New output directory; never overwrite')
    parser.add_argument('--no-read-sources', action='store_true', help='Keep handle/file mapping without reading IR')
    args = parser.parse_args(argv)
    report = analyze(args.run_dir.resolve(), read_sources=not args.no_read_sources)
    args.out.mkdir(parents=False, exist_ok=False)
    (args.out / 'analysis.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    (args.out / 'summary.md').write_text(markdown(report))
    csv_file(args.out / 'samples.csv', report['samples'])
    csv_file(args.out / 'kernel-intervals.csv', report['kernel_intervals'])
    csv_file(args.out / 'graph-summaries.csv', report['graph_summaries'])
    csv_file(args.out / 'kernel-rankings.csv', [dict(scope=r['scope'], input_length=r['input_length'],
             graph_id=r.get('graph_id'), **row) for r in report['rankings'] for row in r['rows']])
    print(json.dumps(dict(analysis_passed=report['analysis_passed'], windows=len(report['windows']),
        samples=len(report['samples']), intervals=len(report['kernel_intervals']),
        errors=report['errors'], output=str(args.out.resolve()))))
    return 0 if report['analysis_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
