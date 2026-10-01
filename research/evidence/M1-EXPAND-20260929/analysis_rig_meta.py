import json, sys
from pathlib import Path
RT = Path("/Users/training-server/trueskate-ai-runtime")
F = RT / "tmp/final-manifest-20260927"; E = RT / "tmp/expand-manifest-20260930"
out = {"train_park_counts": {}}
for name, p in [("2293", F/"subsets/linear_train_n2293.json"), ("4585", F/"subsets/linear_train_n4585.json"),
                ("9170", F/"split/model1_linear_good13100_20260927.train.json"),
                ("18394", E/"model1_linear_expand_20260930.train.json")]:
    out["train_park_counts"][name] = json.load(open(p))["coverage"]["park"]
hold = json.load(open(E/"expand.screened.holdout.json"))
rows = []
for e in hold["samples"]:
    m = json.load(open(RT / e["path"] / "meta.json"))
    rows.append({"path": e["path"], "park": m["park"], "device": m["device"], "session": m["session"],
                 "duration": m["duration"], "waypoints": m["waypoints"]})
out["holdout_meta"] = rows
val = json.load(open(F/"split/model1_linear_good13100_20260927.validation.json"))
vrows = []
for e in val["samples"]:
    m = json.load(open(RT / e["path"] / "meta.json"))
    vrows.append({"path": e["path"], "park": m["park"], "duration": m["duration"], "waypoints": m["waypoints"], "session": m["session"]})
out["validation_meta"] = vrows
json.dump(out, open("/tmp/m1_analysis_meta.json", "w"))
print("ok", len(rows), len(vrows))
