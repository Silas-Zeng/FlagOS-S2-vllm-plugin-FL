"""Make an immutable SQLite backup for matched, privately warmed service arms."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sqlite3


def main(source, destination):
    source, destination = source.resolve(), destination.resolve()
    databases = sorted(source.glob("*.db"))
    assert source.is_dir() and databases, "No tuning databases found"
    assert not destination.is_relative_to(source), "Snapshot must be separate"
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for path in databases:
        output = destination / path.name
        # Online backup gives a consistent database even when the source uses
        # WAL. A byte copy of a live .db would omit uncheckpointed pages.
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=10) as original:
            with sqlite3.connect(output) as backup:
                original.backup(backup, pages=128, sleep=0.01)
                checks = backup.execute("PRAGMA quick_check").fetchall()
                assert checks == [("ok",)], (path, checks)
        with output.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        records.append(dict(name=path.name, source=str(path), sha256=digest,
                            bytes=output.stat().st_size, quick_check="ok"))
    manifest = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    method="SQLite online backup from read-only connections",
                    databases=records)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.out)
