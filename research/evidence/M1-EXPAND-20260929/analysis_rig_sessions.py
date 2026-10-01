import json, collections
from pathlib import Path
RT = Path("/Users/training-server/trueskate-ai-runtime")
F = RT / "tmp/final-manifest-20260927"; E = RT / "tmp/expand-manifest-20260930"
out = {}
for name, p in [("9170", F/"split/model1_linear_good13100_20260927.train.json"),
                ("18394", E/"model1_linear_expand_20260930.train.json"),
                ("test", F/"split/model1_linear_good13100_20260927.test.json"),
                ("validation", F/"split/model1_linear_good13100_20260927.validation.json")]:
    out[name] = collections.Counter(e["session"] for e in json.load(open(p))["samples"])
json.dump(out, open("/tmp/m1_sessions.json", "w")); print({k: (len(v), sum(v.values())) for k, v in out.items()})
