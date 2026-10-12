"""Offline representability diagnostic; never trains or changes the decoder.

Run from any checkout with --source pointing at the frozen implementation.
Inputs are the already-reviewed round-6 manifest and local metadata mirror.
Boundary logits are perfect for the oracle trajectory (including a chosen shift).
The shifted cases are sensitivity probes, not estimates of real label noise.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--mirror", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.source / "src"))
    from trueskate_ai.model1.curved.contract import Gesture
    from trueskate_ai.model1.curved.dataset import targets
    from trueskate_ai.model1.curved.decoder import decode, evaluate_path

    manifest_path = args.mirror / "audit-6/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records = []
    for shift_ms in (0., -16.666667, -8.333333, 8.333333, 16.666667):
        for sample in manifest["samples"]:
            meta = json.loads((args.mirror / sample["sample_dir"] / "meta.json").read_text())
            label = targets(meta)
            times = np.asarray(meta["frame_times_s"], float)
            gesture = Gesture.from_dict(meta["gesture"])
            down = meta["onset_s"] + shift_ms / 1000
            up = meta["release_s"] + shift_ms / 1000
            a = int(np.argmin(abs(times-down)))
            b = int(np.argmin(abs(times-up)))
            xy = gesture.positions(times-down, rate=meta["clock_rate"])
            xy[a], xy[b] = gesture.command_points()[[0, -1]]
            if shift_ms == 0:
                assert np.allclose(xy, label["xy"], atol=1e-7)
            onset = np.full(120, -1.e6)
            release = np.full(120, -1.e6)
            onset[a] = release[b] = 0
            pred = decode(xy, onset, release, times)
            metrics = evaluate_path(pred, meta)
            records.append(dict(id=sample["id"], shift_ms=shift_ms,
                                duration_ms=gesture.times_ms[-1],
                                true_segments=len(gesture.times_ms)-1,
                                predicted_segments=pred["segment_count"],
                                onset_frame=a, release_frame=b,
                                prediction=pred, **metrics))

    summary = {}
    for shift_ms in sorted(set(r["shift_ms"] for r in records)):
        rows = [r for r in records if r["shift_ms"] == shift_ms]
        summary[str(shift_ms)] = dict(
            n=len(rows), passed=sum(r["passed"] for r in rows),
            fit_tolerance_met=sum(r["prediction"]["fit_met_tolerance"] for r in rows),
            median_good_contact_fraction=float(np.median([r["good_contact_fraction"] for r in rows])),
            median_fit_residual=float(np.median([r["fit_residual"] for r in rows])),
            max_fit_residual=max(r["fit_residual"] for r in rows),
            failed_ids=[r["id"] for r in rows if not r["passed"]],
        )
    provenance = dict(
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        implementation={n: hashlib.sha256((args.source / "src/trueskate_ai/model1/curved" / n).read_bytes()).hexdigest()
                        for n in ("contract.py", "dataset.py", "decoder.py")},
        python=sys.version, numpy=np.__version__,
        description="Perfect frame targets and boundary logits; shifts are sensitivity only, not measured noise.",
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(dict(provenance=provenance, summary=summary, samples=records), indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
