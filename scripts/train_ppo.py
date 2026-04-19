"""CLI entry point for trick-conditioned PPO training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trueskate_ai.rl.ppo.trainer import PPOConfig, run_training


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a trick-conditioned policy with PPO on connected iOS devices."
    )
    parser.add_argument("--updates", type=int, default=100, help="Number of PPO updates.")
    parser.add_argument(
        "--steps-per-update", type=int, default=24, help="Rollout samples per PPO update."
    )
    parser.add_argument(
        "--epochs-per-update", type=int, default=4, help="Gradient epochs per PPO update."
    )
    parser.add_argument(
        "--minibatch-size", type=int, default=24, help="PPO minibatch size."
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Adam learning rate.")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clip epsilon.")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient.")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus coefficient.")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clipping norm.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--wait-time", type=float, default=0.0, help="Seconds to wait before OCR.")
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.5,
        help="Seconds to wait after reset before the next rollout.",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=10, help="Checkpoint interval in updates."
    )
    parser.add_argument(
        "--log-dir", type=Path, default=_REPO_ROOT / "logs", help="Run output directory."
    )
    parser.add_argument(
        "--spin-x",
        type=float,
        default=None,
        help="Spin button X coordinate in logical points (optional override).",
    )
    parser.add_argument(
        "--spin-y",
        type=float,
        default=None,
        help="Spin button Y coordinate in logical points (optional override).",
    )
    parser.add_argument(
        "--device-count",
        type=int,
        default=None,
        help="Optional number of configured devices to use (default: all).",
    )
    parser.add_argument(
        "--use-cuda", action="store_true", help="Use CUDA if available for policy updates."
    )
    args = parser.parse_args()
    if (args.spin_x is None) != (args.spin_y is None):
        parser.error("Provide both --spin-x and --spin-y together, or neither.")

    config = PPOConfig(
        updates=args.updates,
        steps_per_update=args.steps_per_update,
        epochs_per_update=args.epochs_per_update,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        wait_time=args.wait_time,
        settle_time=args.settle_time,
        checkpoint_every=args.checkpoint_every,
        log_dir=args.log_dir,
        spin_x=args.spin_x,
        spin_y=args.spin_y,
        device_count=args.device_count,
        use_cuda=args.use_cuda,
    )
    run_training(config)


if __name__ == "__main__":
    main()
