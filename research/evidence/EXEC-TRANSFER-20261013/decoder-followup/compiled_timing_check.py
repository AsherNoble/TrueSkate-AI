"""Check compiled integer-ms commands under the existing per-clip clock rate.

The frozen decoder evaluates predicted ms directly in video time. Compilation
emits command-clock ms. This separate replay hypothesis applies the known
calibration rate and rechecks boundaries/paths with the unchanged evaluator.
It is not a device replay or evidence of executable start-time accuracy.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from decoder_followup import synthetic_cases


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--mirror",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    sys.path.insert(0,str(args.source/"src"))
    from trueskate_ai.model1.curved.contract import Gesture
    from trueskate_ai.model1.curved.decoder import evaluate_path
    manifest=json.loads((args.mirror/"audit-6/manifest.json").read_text())
    audit={r["id"]:json.loads((args.mirror/r["sample_dir"]/"meta.json").read_text()) for r in manifest["samples"]}
    synth={ident:meta for ident,meta,factors in synthetic_cases(Gesture)}
    variants=("refit_grid_coords100_quantized","fitted_integer100_quantized","interval_boundaries_integer100_quantized")
    outputs={}
    for cohort,metadata in (("audit",audit),("synthetic",synth)):
        rows=json.loads(gzip.decompress((args.out/f"{cohort}-results.json.gz").read_bytes()))["samples"]
        results=[]
        for row in rows:
            if row["variant"] not in variants:continue
            meta=metadata[row["id"]]
            pred=dict(row["prediction"])
            pred["times_ms"]=(np.asarray(pred["times_ms"])*meta["clock_rate"]).tolist()
            pred["release_s"]=pred["onset_s"]+pred["times_ms"][-1]/1000
            pred["release_frame"]=int(np.argmin(abs(np.asarray(meta["frame_times_s"])-pred["release_s"])))
            results.append(dict(id=row["id"],variant=row["variant"],**evaluate_path(pred,meta)))
        outputs[cohort]=dict(summary={v:dict(n=len(rr:=[r for r in results if r["variant"]==v]),
                                           passed=sum(r["passed"] for r in rr),
                                           failed_ids=[r["id"] for r in rr if not r["passed"]]) for v in variants},samples=results)
    result=dict(provenance=dict(description=__doc__,argv=sys.argv,
                               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),cohorts=outputs)
    (args.out/"compiled-clock-check.json.gz").write_bytes(gzip.compress((json.dumps(result,separators=(",",":"))+"\n").encode(),mtime=0))
    (args.out/"compiled-clock-summary.json").write_text(json.dumps({k:v["summary"] for k,v in outputs.items()},indent=2)+"\n")
    print(json.dumps({k:v["summary"] for k,v in outputs.items()},indent=2))


if __name__=="__main__":main()
