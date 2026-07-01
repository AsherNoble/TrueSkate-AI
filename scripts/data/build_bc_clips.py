"""Build Model 2 training clips: run Model 1 over expert recordings -> clip.json.

This is the missing bridge (step D+E) between the two BC models. Model 1 (the
trained trace extractor) predicts a per-frame touch from each frame of an expert
recording; `bc.assemble.assemble_strokes` turns that per-frame track into the
discrete strokes Asher actually made; we write them as the `clip.json` that
`bc.sequence_dataset.SequenceDataset` consumes to train Model 2.

    expert clip dir (frame_*.png @ fps)
        --[Model 1 per frame]-->  (active, x, y) track
        --[assemble_strokes]---->  strokes [x0..y2,dur,easing,delay]
        --[write_clip_json]----->  <out>/<clip>/{frame_000000.png(symlink), clip.json}

The output clip dir is a valid SequenceDataset clip (frames named frame_%06d.png
+ clip.json = {"fps", "strokes":[{"params":[9], "t_start", "t_end"}]}).

Model 1 inference is pure torch; this script has no device/Appium deps.

Usage:
    # real: label every clip subdir under an expert corpus
    python scripts/data/build_bc_clips.py \
        --model notebooks/models/trace_extractor_v1.pth \
        --clips-root data/expert/<session> --out data/bc_clips/<session> --fps 30

    # smoke: synthesize a known clip, write clip.json, round-trip via SequenceDataset
    python scripts/data/build_bc_clips.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trueskate_ai.bc.assemble import Stroke, assemble_strokes  # noqa: E402


# --- Model 1 inference -----------------------------------------------------

def load_trace_extractor(model_path: Path, device):
    """Load a Model 1 checkpoint -> (model.eval(), h, w). Pure torch."""
    import torch  # local so --smoke path importing this module stays light
    from trueskate_ai.vision.gaussian_bump_predictor import GaussianBumpPredictor

    ckpt = torch.load(model_path, map_location=device)
    model = GaussianBumpPredictor(in_channels=3, base_channels=ckpt.get("base_channels", 32))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, int(ckpt["h"]), int(ckpt["w"])


def frames_to_touch_track(model, frame_paths: list[Path], h: int, w: int, device,
                          *, active_thresh: float = 0.30, batch: int = 32):
    """Run Model 1 over frames -> (active, x, y) arrays in normalised [0,1].

    A frame is `active` when its peak heatmap response clears `active_thresh`;
    (x, y) is the heatmap argmax normalised by (w, h). Inactive frames get -1.
    """
    import cv2
    import torch

    n = len(frame_paths)
    active = np.zeros(n, dtype=bool)
    xs = np.full(n, -1.0, dtype=np.float64)
    ys = np.full(n, -1.0, dtype=np.float64)

    with torch.no_grad():
        for lo in range(0, n, batch):
            paths = frame_paths[lo:lo + batch]
            imgs = []
            for p in paths:
                img = cv2.imread(str(p))
                if img is None:
                    raise FileNotFoundError(p)
                img = cv2.cvtColor(cv2.resize(img, (w, h)), cv2.COLOR_BGR2RGB)
                imgs.append(img.astype(np.float32) / 255.0)
            x = torch.from_numpy(np.stack(imgs).transpose(0, 3, 1, 2)).to(device)
            hm = model(x)                                    # (B,1,h,w) sigmoid
            hm = hm.squeeze(1).cpu().numpy()                 # (B,h,w)
            for j, hmap in enumerate(hm):
                peak = float(hmap.max())
                if peak >= active_thresh:
                    py, px = np.unravel_index(int(hmap.argmax()), hmap.shape)
                    i = lo + j
                    active[i] = True
                    xs[i] = px / float(w)
                    ys[i] = py / float(h)
    return active, xs, ys


# --- clip.json writer ------------------------------------------------------

def _sorted_frames(clip_dir: Path) -> list[Path]:
    """Frames in a clip dir, in temporal order. Prefer frame_*.png; else any png."""
    fp = sorted(clip_dir.glob("frame_*.png"))
    return fp or sorted(clip_dir.glob("*.png"))


def write_clip_json(out_dir: Path, fps: float, strokes: list[Stroke],
                    src_frames: list[Path]) -> Path:
    """Write a SequenceDataset clip: frame_%06d.png symlinks + clip.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(src_frames):
        link = out_dir / f"frame_{i:06d}.png"
        if link.exists() or link.is_symlink():
            link.unlink()
        # symlink so we don't duplicate a 31GB corpus; resolve to absolute.
        link.symlink_to(src.resolve())
    meta = {
        "fps": float(fps),
        "strokes": [
            {"params": [float(v) for v in s.params], "t_start": s.t_start, "t_end": s.t_end}
            for s in strokes
        ],
    }
    (out_dir / "clip.json").write_text(json.dumps(meta, indent=2))
    return out_dir / "clip.json"


def build_clip(clip_dir: Path, out_dir: Path, model, h: int, w: int, device,
               *, fps: float, active_thresh: float, latency_s: float) -> int:
    """Label one expert clip dir -> its clip.json. Returns #strokes assembled.

    Model 1 is trained (self_label `latency_s`) to predict the LAGGING orange
    trace — its touch at frame time `t` reflects the real finger at `t - latency`
    (see self_label.label_frames: `t = ft - latency_s`). So we place each frame's
    touch at `t - latency_s`, otherwise every assembled stroke's t_start/t_end is
    ~latency late and Model 2 would condition on post-touch frames. The shift is
    constant, so run-segmentation gaps/durations are unchanged.
    """
    frames = _sorted_frames(clip_dir)
    if not frames:
        raise RuntimeError(f"no frames in {clip_dir}")
    times = np.arange(len(frames), dtype=np.float64) / fps - latency_s
    active, xs, ys = frames_to_touch_track(model, frames, h, w, device, active_thresh=active_thresh)
    strokes = assemble_strokes(active, xs, ys, times)
    write_clip_json(out_dir, fps, strokes, frames)
    return len(strokes)


# --- entry points ----------------------------------------------------------

def _run(args) -> None:
    import torch
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    model, h, w = load_trace_extractor(args.model, device)
    clip_dirs = [d for d in sorted(args.clips_root.glob("**/")) if _sorted_frames(d)]
    if not clip_dirs:
        raise SystemExit(f"no clips with frames under {args.clips_root}")
    print(f"model {args.model.name}  ({h}x{w})  device={device}  clips={len(clip_dirs)}")
    total = 0
    for cd in clip_dirs:
        rel = cd.relative_to(args.clips_root)
        out = args.out / rel
        n = build_clip(cd, out, model, h, w, device, fps=args.fps,
                       active_thresh=args.active_thresh, latency_s=args.latency_s)
        total += n
        print(f"  {rel}: {n} strokes -> {out/'clip.json'}")
    print(f"done: {len(clip_dirs)} clips, {total} strokes total")


def _smoke() -> None:
    """Synthesize a known clip, write clip.json, and load it via SequenceDataset.

    Proves the writer + schema without a trained model or real corpus: build a
    per-frame touch track from two known gestures (as in assemble's self-test),
    assemble it, write a clip, then confirm SequenceDataset yields decision
    points with the right tensor shapes.
    """
    import tempfile

    import cv2
    from trueskate_ai.vision.self_label import label_frames
    from trueskate_ai.bc.model2 import SequencePolicyConfig
    from trueskate_ai.bc.sequence_dataset import SequenceDataset

    fps = 30.0
    g1 = dict(waypoints=[(0.30, 0.72), (0.46, 0.50), (0.74, 0.44)], dur=0.40)
    g2 = dict(waypoints=[(0.60, 0.40), (0.50, 0.60), (0.40, 0.80)], dur=0.30)
    t1 = np.arange(0.10, 0.10 + g1["dur"] + 1e-9, 1 / fps)
    t2 = np.arange(0.80, 0.80 + g2["dur"] + 1e-9, 1 / fps)
    clip_t = np.concatenate([np.arange(0.0, 0.10, 1 / fps), t1, np.arange(0.55, 0.80, 1 / fps), t2,
                             np.arange(1.15, 1.30, 1 / fps)])
    l1 = label_frames(g1["waypoints"], g1["dur"], 1.0, list(clip_t - t1[0]))
    l2 = label_frames(g2["waypoints"], g2["dur"], 1.0, list(clip_t - t2[0]))
    active = np.array([a.active or b.active for a, b in zip(l1, l2)])
    xs = np.array([a.x if a.active else b.x for a, b in zip(l1, l2)])
    ys = np.array([a.y if a.active else b.y for a, b in zip(l1, l2)])
    times = clip_t

    strokes = assemble_strokes(active, xs, ys, times)
    assert len(strokes) == 2, f"expected 2 strokes, got {len(strokes)}"

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        src.mkdir()
        frame_paths = []
        for i in range(len(times)):
            p = src / f"frame_{i:06d}.png"
            cv2.imwrite(str(p), np.zeros((208, 96, 3), dtype=np.uint8))  # blank portrait frames
            frame_paths.append(p)
        out = Path(td) / "clip0"
        cj = write_clip_json(out, fps, strokes, frame_paths)
        assert cj.exists()
        loaded = json.loads(cj.read_text())
        assert loaded["fps"] == fps and len(loaded["strokes"]) == 2
        assert len(loaded["strokes"][0]["params"]) == 9

        cfg = SequencePolicyConfig()
        ds = SequenceDataset(Path(td), cfg=cfg)
        assert len(ds) >= 1, "no decision points produced"
        item = ds[0]
        assert tuple(item["frames"].shape) == (cfg.n_frames, 3, cfg.img_h, cfg.img_w), item["frames"].shape
        assert tuple(item["target"].shape) == (cfg.m_out, 9), item["target"].shape
    print("SMOKE OK — synthesized clip.json round-trips through SequenceDataset "
          f"(2 strokes, frames {cfg.n_frames}×3×{cfg.img_h}×{cfg.img_w})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Model 2 clip.json from expert recordings via Model 1.")
    ap.add_argument("--smoke", action="store_true", help="Synthetic writer+schema round-trip; no model/corpus.")
    ap.add_argument("--model", type=Path, help="Model 1 checkpoint (trace_extractor_*.pth).")
    ap.add_argument("--clips-root", type=Path, help="Dir whose subdirs are expert clips (frames each).")
    ap.add_argument("--out", type=Path, help="Output root; mirrors --clips-root structure.")
    ap.add_argument("--fps", type=float, default=30.0, help="Frame rate of the expert recordings.")
    ap.add_argument("--active-thresh", type=float, default=0.30,
                    help="Min peak heatmap response to count a frame as a touch.")
    ap.add_argument("--latency-s", type=float, default=0.45,
                    help="Model 1 predicts the LAGGING trace; subtract its training latency_s so "
                         "stroke times track the real finger, not the swoosh. MUST match the "
                         "Model 1 checkpoint's training latency (default 0.45, the validated value).")
    args = ap.parse_args()

    if args.smoke:
        _smoke()
        return
    if not (args.model and args.clips_root and args.out):
        ap.error("real mode needs --model, --clips-root, and --out (or use --smoke)")
    _run(args)


if __name__ == "__main__":
    main()
