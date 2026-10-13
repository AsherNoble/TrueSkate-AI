"""Restore only bundled raw data to ignored tmp, then repeat offline analysis."""
import gzip
import hashlib
import json
from pathlib import Path

import study
import audit

ROOT = Path(__file__).resolve().parent
raw = study.WORK/'tmp/sim-compensation/requests.jsonl'
expected = json.loads((ROOT/'results.json').read_text())['raw_requests_sha256']
data = gzip.decompress((ROOT/'requests.jsonl.gz').read_bytes())
assert hashlib.sha256(data).hexdigest()==expected
raw.parent.mkdir(parents=True,exist_ok=True)
if raw.exists():
    assert raw.read_bytes()==data, 'Preserve differing existing measurements'
else:
    raw.write_bytes(data)
study.analyze()
audit.main()
