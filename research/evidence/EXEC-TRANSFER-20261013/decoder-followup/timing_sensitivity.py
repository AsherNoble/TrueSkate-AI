"""Repeat original timing-shift diagnostic with research decoder ablations.

Shifts are imposed sensitivity cases, NOT estimated error/noise distributions.
No retraining, holdout test, gesture execution, or production code modification.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from decoder_followup import (fractional_decode, integer_interval_prediction,
                             interval_boundary_decode, quantized,
                             refit_grid_coordinates, summary)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--mirror",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    sys.path.insert(0,str(args.source/"src"))
    from trueskate_ai.model1.curved.contract import Gesture
    from trueskate_ai.model1.curved.decoder import decode,evaluate_path
    manifest=json.loads((args.mirror/"audit-6/manifest.json").read_text())
    all_rows=[]
    for shift_ms in (-16.666667,-8.333333,0.,8.333333,16.666667):
        for sample in manifest["samples"]:
            meta=json.loads((args.mirror/sample["sample_dir"]/"meta.json").read_text())
            g=Gesture.from_dict(meta["gesture"])
            t=np.asarray(meta["frame_times_s"],float)
            down,up=meta["onset_s"]+shift_ms/1000,meta["release_s"]+shift_ms/1000
            a,b=int(np.argmin(abs(t-down))),int(np.argmin(abs(t-up)))
            xy=g.positions(t-down,rate=meta["clock_rate"])
            xy[a],xy[b]=g.command_points()[[0,-1]]
            xy=xy.astype(np.float32).astype(float)
            onset,release=np.full(120,-1.e6),np.full(120,-1.e6)
            onset[a]=release[b]=0.
            base=decode(xy,onset,release,t)
            variants=dict(baseline=base,
                          refit_grid_coords100_quantized=quantized(refit_grid_coordinates(xy,t,a,b,base)),
                          fitted_integer100_quantized=quantized(fractional_decode(xy,t,a,b,base,round_output=True)))
            interval=interval_boundary_decode(xy,t,a,b,base)
            interval=integer_interval_prediction(interval,t)
            variants["interval_boundaries_integer100_quantized"]=quantized(interval)
            for name,pred in variants.items():
                all_rows.append(dict(id=sample["id"],shift_ms=shift_ms,variant=name,
                                     **evaluate_path(pred,meta)))
        print(shift_ms, {v:sum(r["passed"] for r in all_rows if r["shift_ms"]==shift_ms and r["variant"]==v)
                         for v in variants},flush=True)
    compact={str(s):{v:dict(n=len(rr:=[r for r in all_rows if r["shift_ms"]==s and r["variant"]==v]),
                           passed=sum(r["passed"] for r in rr),failed_ids=[r["id"] for r in rr if not r["passed"]])
                     for v in variants} for s in sorted({r["shift_ms"] for r in all_rows})}
    provenance=dict(description=__doc__,argv=sys.argv,python=sys.version,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    decoder_followup_sha256=hashlib.sha256(Path(__file__).with_name("decoder_followup.py").read_bytes()).hexdigest(),
                    manifest_sha256=hashlib.sha256((args.mirror/"audit-6/manifest.json").read_bytes()).hexdigest())
    (args.out/"timing-sensitivity.json.gz").write_bytes(gzip.compress((json.dumps(dict(provenance=provenance,summary=compact,samples=all_rows),separators=(",",":"))+"\n").encode(),mtime=0))
    (args.out/"timing-sensitivity-summary.json").write_text(json.dumps(compact,indent=2)+"\n")


if __name__=="__main__":
    main()
