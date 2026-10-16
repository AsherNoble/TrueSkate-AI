"""Train or evaluate the independently versioned curved Model 1 corpus."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("train", "test", "human"))
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--human", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--cache", action="store_true")
    a = p.parse_args()
    from trueskate_ai.model1.curved.training import train, test_once
    if a.command == "train":
        if not a.human:
            p.error("--human frozen manifest is required before training")
        result = train(a.manifest, a.human, a.out, device=a.device, cache=a.cache, resume=a.resume)
    elif a.command == "test":
        result = test_once(a.manifest, a.out, device=a.device)
    else:
        if not a.human:
            p.error("--human required")
        from trueskate_ai.model1.curved.inference import human_review
        result = human_review(a.human, a.out, device=a.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
