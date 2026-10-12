"""Generate the Colab benchmark notebook from the current harness files.

Every code cell bootstraps the same way on a fresh runtime: install the pinned
rclone, load the drive.file-scoped credential from Colab Secrets, then fetch
the harness and input lock from the benchmark folder and refuse any byte that
differs from the SHA-256 embedded here. Rebuild after any harness change.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOCK = ROOT / "research/evidence/M1-COLAB-20261006/input-lock.json"
NOTEBOOK = HERE / "Model1-Colab-benchmark-20261006.ipynb"
SECRET = "TRUESKATE_BENCH_RCLONE"
RCLONE = "v1.75.1"
RCLONE_LINUX_SHA256 = "982b5aa772841168f8e380f139e9e787b2a105403e32b94da8676a0e1c0a13ab"

HEADER = """# Isolated Model 1 Colab benchmark (M1-COLAB-20261006)
Free compute only; one account and runtime. 40 epochs, seed 0, frozen source e0424f99e3f668e1722e56fac4f17372cd60a62d.
12 cumulative GPU allocation hours within 72 calendar hours from first allocation, including setup, diagnostics, staging and recovery.
At most two connection attempts per UTC day; no retry loops. Keep this notebook private.
Storage: one Drive folder through a drive.file-scoped rclone credential (Colab secret `{secret}`). It can reach only files its own OAuth client created; never mount Drive.
Use CPU for preparation and ledger cells. Each GPU cell releases its runtime after the phase, failure or budget expiry.
No test, training-set or new-park holdout evaluation. Sequential decoding, full precision and content verification; caching disabled throughout Colab.
Do not Run All. Execute the numbered steps individually."""

BOOTSTRAP = """# Bootstrap: pinned rclone, drive.file-scoped credential, harness verified against this notebook.
import hashlib, importlib.util, io, os, subprocess, time, urllib.request, zipfile
from pathlib import Path
from google.colab import userdata
BIN, CONF, STORE = Path('/content/rclone/rclone'), Path('/content/rclone/rclone.conf'), Path('/content/bench-store')
EXPECTED = {expected!r}
if not BIN.exists():
    # GitHub release asset: downloads.rclone.org rejects urllib's default User-Agent (HTTP 403).
    archive = urllib.request.urlopen('https://github.com/rclone/rclone/releases/download/{rclone}/rclone-{rclone}-linux-amd64.zip', timeout=300).read()
    if hashlib.sha256(archive).hexdigest() != '{rclone_sha}':
        raise RuntimeError('rclone archive checksum mismatch')
    BIN.parent.mkdir(parents=True, exist_ok=True)
    BIN.write_bytes(zipfile.ZipFile(io.BytesIO(archive)).read('rclone-{rclone}-linux-amd64/rclone')); BIN.chmod(0o700)
conf = userdata.get('{secret}')
if 'scope = drive.file' not in conf.splitlines() or not any(l.startswith('root_folder_id = ') for l in conf.splitlines()):
    raise RuntimeError('refusing a credential broader than one drive.file folder')
CONF.parent.mkdir(parents=True, exist_ok=True); CONF.write_text(conf); CONF.chmod(0o600)
os.environ.update(RCLONE_CONFIG=str(CONF), RCLONE_BIN=str(BIN))
for rel, digest in EXPECTED.items():
    subprocess.run([str(BIN), 'copyto', 'bench:' + rel, str(STORE / rel)], check=True)
    if hashlib.sha256((STORE / rel).read_bytes()).hexdigest() != digest:
        raise RuntimeError(rel + ' does not match this notebook')
spec = importlib.util.spec_from_file_location('supervisor', STORE / 'code/supervisor.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
"""

STEPS = [
    ("1. CPU: verify the scoped inputs by Drive checksums and record the first connection attempt",
     "s.prepare()"),
    ("2. Select free T4 (standard RAM), then run preflight and epochs 1–5. The 40-epoch cosine schedule is unchanged.",
     "s.execute('timing')"),
    ("3. After deletion is observed in Runtime → Manage sessions: select CPU, close that allocation and record the next "
     "attempt. After an interruption, set SESSION_ID to the dropped session and rerun the matching GPU step.",
     "SESSION_ID = 'session-1'\ns.confirm_release(SESSION_ID, time.time())\ns.next_attempt()"),
    ("4. Select free T4 again; resume at epoch 6 toward epoch 40. The harness checks restored "
     "model/optimizer/RNG/shuffle state and the cosine learning rate.",
     "s.execute('completion')"),
    ("5. After final deletion is observed: on CPU, close the last allocation and write the report. "
     "Record no further attempt.",
     "SESSION_ID = 'session-2'\ns.confirm_release(SESSION_ID, time.time())"),
    ("Only if a GPU connection failed: on CPU, mark the prepared attempt failed (then use step 3's next_attempt within the cap).",
     "s.record_failed_connection('describe the observed failure')"),
    ("At the end, on CPU: record active operator time; unknown effort stays unknown.",
     "s.record_manual_effort([{'what': 'describe', 'seconds': 0}], complete=False)"),
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cell(kind, text):
    lines = text.splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


def build():
    expected = {"code/benchmark.py": sha256(HERE / "benchmark.py"),
                "code/supervisor.py": sha256(HERE / "supervisor.py"),
                "inputs/input-lock.json": sha256(LOCK)}
    bootstrap = BOOTSTRAP.format(expected=expected, rclone=RCLONE, rclone_sha=RCLONE_LINUX_SHA256, secret=SECRET)
    cells = [cell("markdown", HEADER.format(secret=SECRET))]
    for title, body in STEPS:
        cells += [cell("markdown", "## " + title), cell("code", bootstrap + body + "\n")]
    return {"cells": cells, "metadata": {"colab": {"name": NOTEBOOK.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def render():
    return json.dumps(build(), indent=1, ensure_ascii=False) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="fail if the committed notebook is stale")
    a = p.parse_args()
    if a.check:
        if NOTEBOOK.read_text() != render():
            raise SystemExit(f"{NOTEBOOK.name} is stale; run build_notebook.py")
        return
    NOTEBOOK.write_text(render())
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
