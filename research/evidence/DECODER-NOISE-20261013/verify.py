"""Verify the result ledger and deterministically replay 120 prespecified rows."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import study

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--source",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);args=ap.parse_args()
    sys.path.insert(0,str(args.source/"src"))
    from trueskate_ai.model1.curved.contract import Gesture
    from trueskate_ai.model1.curved.dataset import targets
    from trueskate_ai.model1.curved.decoder import decode,evaluate_path
    from trueskate_ai.data.control_hitboxes import segment_is_safe
    prior=args.source/"research/evidence/EXEC-TRANSFER-20261013/decoder-followup/decoder_followup.py"
    spec=importlib.util.spec_from_file_location("prior",prior);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    provenance=json.loads((args.out/"provenance.json").read_text());cohort=json.loads((args.out/"cohort-frozen.json").read_text())
    assert study.sha(args.out/"PLAN-FROZEN.md")==cohort["plan_sha256"]
    assert study.sha(args.out/"study.py")==provenance["script_sha256"]
    assert study.sha(prior)==provenance["refit_sha256"]
    for path,digest in provenance["source_hashes"].items():assert study.sha(args.source/path)==digest
    fixtures,rejected=study.generate(Gesture,segment_is_safe)
    assert hashlib.sha256(study.stable_bytes(fixtures)).hexdigest()==cohort["fixture_sha256"]
    assert gzip.decompress((args.out/"fixtures.json.gz").read_bytes())==study.stable_bytes(fixtures)
    rows=json.loads(gzip.decompress((args.out/"paired-results.json.gz").read_bytes()))
    assert len(rows)==43200
    keys={(r["id"],r["noise"],r["level"],r["onset_shift"],r["release_shift"]) for r in rows}
    assert len(keys)==len(rows)
    for r in rows:
        for name in ("baseline","candidate"):
            x=r[name];s=x["score"];guard=x["guard"]
            assert x["metric_pass"]==bool(s and s["passed"])
            assert x["admissible_pass"]==bool(x["metric_pass"] and guard["valid"])
            assert guard["valid"]==(not guard["reasons"])
            if s:
                assert s["passed"]==bool(s["good_contact_fraction"]>=.95 and s["onset_error_frames"]<=2 and s["release_error_frames"]<=2)
        assert r["baseline"]["selected_boundaries"]==r["candidate"]["selected_boundaries"]
        assert r["baseline"]["segment_count"]==r["candidate"]["segment_count"]
    selected=[r for r in rows if r["id"].endswith("-00") and (r["level"]==.03 or r["noise"]=="clean") and (r["onset_shift"],r["release_shift"]) in ((0,0),(-1,1),(1,-1))]
    assert len(selected)==120
    byid={f["id"]:(i,f) for i,f in enumerate(fixtures)}
    for r in selected:
        i,f=byid[r["id"]];meta=f["meta"];label=targets(meta);a,b=int(label["onset"]),int(label["release"]);times=np.asarray(meta["frame_times_s"])
        clean=np.asarray(label["xy"],float)
        noise=np.zeros_like(clean) if r["noise"]=="clean" else study.noise_fields(i,a,b)[r["noise"]]*r["level"]
        xy=np.clip(clean+noise,0,1).astype(np.float32)
        onset=-.5*((np.arange(120)-a-r["onset_shift"])/.75)**2
        release=-.5*((np.arange(120)-b-r["release_shift"])/.75)**2
        base=decode(xy,onset,release,times);candidate=mod.refit_grid_coordinates(xy,times,base["onset_frame"],base["release_frame"],base)
        for name,p in (("baseline",base),("candidate",candidate)):
            quantized,guard=study.check_compile(p,Gesture,segment_is_safe)
            score=evaluate_path(quantized,meta)
            assert score==r[name]["score"],(r["id"],r["noise"],name)
            assert {k:v for k,v in guard.items() if k!="latency_ms"}=={k:v for k,v in r[name]["guard"].items() if k!="latency_ms"}
    repeats=json.loads((args.out/"repeat-checks.json").read_text())
    assert all(r["baseline_equal"] and r["candidate_equal"] for r in repeats)
    result=dict(result_rows=len(rows),unique_keys=len(keys),fixture_regeneration=True,source_hashes_match=True,
                frozen_plan_hash_matches=True,replayed_rows=len(selected),exact_replay_scores_and_payloads=True,
                paired_boundaries_segment_counts_equal=True,deterministic_repeats=len(repeats),passed=True)
    study.save(args.out/"verification.json",result);print(json.dumps(result,indent=2))

if __name__=="__main__":main()
