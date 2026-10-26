"""Read archived post-launch resources on CPU; never initialize GPU handles."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def summarize(archive):
    kernels = {}

    def visit(value, location):
        if isinstance(value, dict):
            if value.get('mode') == 'selected' and 'loaded_resources' in value:
                before = value['candidate_resources']
                loaded = value['loaded_resources']
                key = value['candidate_hash']
                assert before['hash'] == loaded['hash'] == key
                assert before['metadata'] == loaded['metadata']
                assert before['asm_sha256'] == loaded['asm_sha256']
                assert isinstance(loaded['n_regs'], int) and loaded['n_regs'] > 0
                assert isinstance(loaded['n_spills'], int) and loaded['n_spills'] >= 0
                row = dict(candidate_hash=key, metadata=loaded['metadata'],
                           compiled_n_regs=before['n_regs'], compiled_n_spills=before['n_spills'],
                           loaded_n_regs=loaded['n_regs'], loaded_n_spills=loaded['n_spills'],
                           asm_sha256=loaded['asm_sha256'])
                if key in kernels:
                    assert kernels[key]['resources'] == row
                else:
                    kernels[key] = dict(resources=row, locations=[])
                kernels[key]['locations'].append(location)
                return
            for name, child in value.items():
                visit(child, location + '/' + name)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, location + '/' + str(index))

    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert len({member.name for member in members}) == len(members)
        for member in members:
            path = PurePosixPath(member.name)
            assert not path.is_absolute() and '..' not in path.parts
            if path.name != 'result.json':
                continue
            assert member.isfile() and member.size < 20 * 1024 * 1024
            with tar.extractfile(member) as stream:
                result = json.load(stream)
            assert result['completed'] and not result['errors'], member.name
            visit(result, member.name)
    assert kernels, 'No actual post-launch resource records'
    return dict(archive=archive.name, bytes=archive.stat().st_size,
                sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), kernels=kernels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = dict(cpu_only=True, gpu_work_started=False,
                  basis='Existing selected compile records and their post-launch loaded_resources',
                  caveat='Driver-reported counts do not establish occupancy, register-file capacity, or absence of all local-memory traffic.',
                  archives=[summarize(path) for path in args.archives])
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(archives=len(result['archives']),
                         kernels=sum(len(a['kernels']) for a in result['archives']), out=str(args.out))))


if __name__ == '__main__':
    main()
