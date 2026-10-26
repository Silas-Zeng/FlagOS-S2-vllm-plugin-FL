# Terminal archive for PR72 probability-only five-FMA candidate

The observed median speedups were 0.9952833856780836x (packed 4K/B64) and
0.9936934017515131x (16K/B32). The candidate did not show a component gain and
was not promoted to full model or official service acceptance.

`final-verdict.json` qualifies the evidence and preserves the precise scope.
`result-original.json` is byte-identical to `raw/result.json` in the archive.
Its `passed=true` fields remain historical data: four amplitude1 cases report
GPU FP64 RMS zero despite nonzero max_abs. Changed-input graph RMS shares that
unvalidated diagnostic path. These flags cannot establish complete numerical
equivalence. CPU sampled references and all raw graph/stream records are retained;
no numerical kernel was changed and no GPU run was repeated for archiving.

The single 301,578B `.tar.gz` contains 40 payload files and `manifest.json`:
all generated/code/result/source-proof files, all 30 actual primitive and both
shape baseline/candidate compiler resources (source, TTIR, TTGIR, LLVM, ASM,
cubin), owner, run log, original input Git bundle and four source files from
frozen commit40e14f37081edaa88527b7ef6ac6a31c68c8e940. It excludes pycache.
The original result folder was read only; new CPU exports were created separately.

`verification.json` records remotely observed SHA256 values for the downloaded
archive, manifest and index, and verifies each payload hash, every original
resource hash and frozen source identity. Recheck the archive without extracting
files to disk, from this directory:

```bash
python3 - <<'PY'
import hashlib, json, pathlib, tarfile
root = pathlib.Path('.')
verification = json.loads((root/'verification.json').read_text())
for name, digest in verification['download_hashes_verified'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
index_path = next(root.glob('*-index.json'))
index = json.loads(index_path.read_text())
assert index['archive_bytes'] <= 2*1024*1024
manifest_bytes = (root/index['manifest']).read_bytes()
manifest = json.loads(manifest_bytes)
with tarfile.open(root/index['archive'], 'r:gz') as archive:
    assert len(archive.getmembers()) == len(manifest['files'])+1
    assert archive.extractfile('manifest.json').read() == manifest_bytes
    for record in manifest['files']:
        data = archive.extractfile(record['path']).read()
        assert len(data) == record['bytes']
        assert hashlib.sha256(data).hexdigest() == record['sha256']
print('Archive and every payload hash verified; this is not numerical acceptance.')
PY
```

Original owner identity: PID602351/start340484724, terminal Z. Test command,
source commit and log location are retained in `owner.json` inside the archive.
This closure applies to this five-FMA probability replacement only. The `both`
variant, other native/software exp mixtures and other FA4 mechanisms remain
unevaluated by this run. `precision_gate=false`, `official_score=false`.
