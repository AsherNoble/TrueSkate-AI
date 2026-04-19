"""End-to-end trick-conditioned PPO trainer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from trueskate_ai.nn.policy import TrickConditionedPolicy
from trueskate_ai.rl.collectors.trick_conditioned_collector import (
    RolloutTask,
    collect_rollouts,
)
from trueskate_ai.rl.device_worker import DEVICES, DeviceWorker
from trueskate_ai.rl.ppo.buffer import RolloutBatch

TRICK_LIST: tuple[str, ...] = (
    "OLLIE", "NOLLIE", "KICKFLIP", "DOUBLE KICKFLIP", "TRIPLE KICKFLIP",
    "HEELFLIP", "DOUBLE HEELFLIP", "TRIPLE HEELFLIP", "POP SHOVE-IT", "FS POP SHOVE-IT",
    "360 POP SHOVE-IT", "FS 360 POP SHOVE-IT", "FRONTSIDE 180", "BACKSIDE 180", "FRONTSIDE 360",
    "BACKSIDE 360", "VARIAL KICKFLIP", "VARIAL HEELFLIP", "NIGHTMARE FLIP", "HARD FLIP",
    "DOUBLE HARD FLIP", "360 HARD FLIP", "INWARD HEELFLIP", "LASER FLIP", "360 FLIP",
    "360 DOUBLE FLIP", "BACKSIDE FLIP", "BACKSIDE DOUBLE FLIP", "FRONTSIDE FLIP",
    "FRONTSIDE DOUBLE FLIP", "BACKSIDE HEEL FLIP", "FRONTSIDE HEEL FLIP", "BACKSIDE 360 FLIP",
    "FRONTSIDE 360 FLIP", "BACKSIDE 360 HEEL", "FRONTSIDE 360 HEEL",
)


@dataclass(frozen=True)
class PPOConfig:
    updates: int = 100
    steps_per_update: int = 24
    epochs_per_update: int = 4
    minibatch_size: int = 24
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    seed: int = 42
    wait_time: float = 0.0
    settle_time: float = 0.5
    checkpoint_every: int = 10
    log_dir: Path = Path("logs")
    spin_x: float | None = None
    spin_y: float | None = None
    device_count: int | None = None
    use_cuda: bool = False


def _open_run_log(log_dir: Path) -> tuple[Path, object]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = log_dir / "runs" / f"ppo_run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_fh = (run_dir / f"ppo_run_{run_id}.jsonl").open("w", buffering=1)
    return run_dir, log_fh


def _normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    std = advantages.std(unbiased=False)
    if float(std) < 1e-8:
        return advantages - advantages.mean()
    return (advantages - advantages.mean()) / (std + 1e-8)


def _write_jsonl(fh, record: dict) -> None:
    fh.write(json.dumps(record) + "\n")


def _config_to_json_dict(config: PPOConfig) -> dict:
    data = asdict(config)
    data["log_dir"] = str(data["log_dir"])
    return data


def run_training(config: PPOConfig) -> None:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)

    device = torch.device("cuda" if config.use_cuda and torch.cuda.is_available() else "cpu")
    tricks = list(TRICK_LIST)
    workers_cfg = DEVICES if config.device_count is None else DEVICES[: config.device_count]
    workers = [DeviceWorker(cfg) for cfg in workers_cfg]
    if not workers:
        raise RuntimeError("No devices configured for rollout collection")

    for worker in workers:
        worker.connect()

    run_dir, log_fh = _open_run_log(config.log_dir)
    policy = TrickConditionedPolicy(num_tricks=len(tricks)).to(device)
    optimizer = Adam(policy.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)

    spin_override = (
        (config.spin_x, config.spin_y)
        if config.spin_x is not None and config.spin_y is not None
        else None
    )

    eval_num = 0
    try:
        for update_idx in range(config.updates):
            trick_idxs_np = rng.integers(0, len(tricks), size=config.steps_per_update, endpoint=False)
            trick_idxs = torch.as_tensor(trick_idxs_np, dtype=torch.long, device=device)

            with torch.no_grad():
                sample = policy.act(trick_idxs)
                actions = sample.action
                old_log_probs = sample.log_prob
                values = sample.value

            actions_np = actions.detach().cpu().numpy()
            tasks = []
            for sample_idx in range(config.steps_per_update):
                eval_num += 1
                tasks.append(
                    RolloutTask(
                        sample_idx=sample_idx,
                        action=actions_np[sample_idx],
                        target_trick=tricks[int(trick_idxs_np[sample_idx])],
                        eval_num=eval_num,
                        update_idx=update_idx,
                    )
                )

            rollout_results = collect_rollouts(
                workers=workers,
                tasks=tasks,
                wait_time=config.wait_time,
                settle_time=config.settle_time,
                spin_button_xy=spin_override,
            )

            rewards = torch.as_tensor(
                [r.reward for r in rollout_results], dtype=torch.float32, device=device
            )
            returns = rewards
            advantages = _normalize_advantages(returns - values)

            batch = RolloutBatch(
                trick_idx=trick_idxs.detach(),
                actions=actions.detach(),
                old_log_probs=old_log_probs.detach(),
                returns=returns.detach(),
                advantages=advantages.detach(),
            )

            policy.train()
            mean_policy_loss = 0.0
            mean_value_loss = 0.0
            mean_entropy = 0.0
            n_steps = 0

            for _ in range(config.epochs_per_update):
                for mb in batch.iter_minibatches(config.minibatch_size, generator=generator):
                    mb_trick_idx, mb_actions, mb_old_logp, mb_returns, mb_advantages = mb
                    new_logp, entropy, new_values = policy.evaluate_actions(mb_trick_idx, mb_actions)

                    ratio = torch.exp(new_logp - mb_old_logp)
                    unclipped = ratio * mb_advantages
                    clipped = torch.clamp(
                        ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon
                    ) * mb_advantages
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    value_loss = nn.functional.mse_loss(new_values, mb_returns)
                    entropy_bonus = entropy.mean()
                    loss = (
                        policy_loss
                        + config.value_coef * value_loss
                        - config.entropy_coef * entropy_bonus
                    )

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
                    optimizer.step()

                    mean_policy_loss += float(policy_loss.detach().cpu())
                    mean_value_loss += float(value_loss.detach().cpu())
                    mean_entropy += float(entropy_bonus.detach().cpu())
                    n_steps += 1

            if n_steps > 0:
                mean_policy_loss /= n_steps
                mean_value_loss /= n_steps
                mean_entropy /= n_steps

            mean_reward = float(rewards.mean().detach().cpu())
            max_reward = float(rewards.max().detach().cpu())
            n_samples = len(rollout_results)
            n_errors = sum(1 for r in rollout_results if r.error is not None)
            n_detected = sum(1 for r in rollout_results if r.detected_trick is not None)
            n_landed = sum(1 for r in rollout_results if r.detected_status == "landed")
            n_matches = sum(1 for r in rollout_results if r.reward > 0.0)
            detection_rate = (n_detected / n_samples) if n_samples else 0.0
            landed_rate = (n_landed / n_samples) if n_samples else 0.0
            match_rate = (n_matches / n_samples) if n_samples else 0.0
            error_rate = (n_errors / n_samples) if n_samples else 0.0

            device_summary: dict[str, dict[str, int]] = {}
            for result in rollout_results:
                stats = device_summary.setdefault(
                    result.device_id,
                    {"samples": 0, "detected": 0, "landed": 0, "matched": 0, "errors": 0},
                )
                stats["samples"] += 1
                if result.detected_trick is not None:
                    stats["detected"] += 1
                if result.detected_status == "landed":
                    stats["landed"] += 1
                if result.reward > 0.0:
                    stats["matched"] += 1
                if result.error is not None:
                    stats["errors"] += 1

            for result in rollout_results:
                _write_jsonl(
                    log_fh,
                    {
                        "type": "sample",
                        "update": update_idx,
                        "eval_num": result.eval_num,
                        "sample_idx": result.sample_idx,
                        "device_id": result.device_id,
                        "target_trick": result.target_trick,
                        "detected_trick": result.detected_trick,
                        "detected_status": result.detected_status,
                        "error": result.error,
                        "reward": result.reward,
                        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    },
                )

            _write_jsonl(
                log_fh,
                {
                    "type": "update_summary",
                    "update": update_idx,
                    "mean_reward": mean_reward,
                    "max_reward": max_reward,
                    "match_rate": round(match_rate, 4),
                    "detection_rate": round(detection_rate, 4),
                    "landed_rate": round(landed_rate, 4),
                    "error_rate": round(error_rate, 4),
                    "device_summary": device_summary,
                    "policy_loss": round(mean_policy_loss, 6),
                    "value_loss": round(mean_value_loss, 6),
                    "entropy": round(mean_entropy, 6),
                    "config": _config_to_json_dict(config) if update_idx == 0 else None,
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                },
            )

            print(
                f"[update {update_idx:04d}] mean_reward={mean_reward:.3f} "
                f"max_reward={max_reward:.3f} "
                f"match_rate={match_rate:.2%} detect_rate={detection_rate:.2%} "
                f"error_rate={error_rate:.2%} "
                f"policy_loss={mean_policy_loss:.4f} value_loss={mean_value_loss:.4f}"
            )

            if (update_idx + 1) % config.checkpoint_every == 0:
                ckpt_path = run_dir / f"policy_update_{update_idx + 1:04d}.pt"
                torch.save(
                    {"policy_state_dict": policy.state_dict(), "config": _config_to_json_dict(config)},
                    ckpt_path,
                )
                print(f"Saved checkpoint: {ckpt_path}")

        final_ckpt = run_dir / "policy_final.pt"
        torch.save({"policy_state_dict": policy.state_dict(), "config": _config_to_json_dict(config)}, final_ckpt)
        print(f"Saved final checkpoint: {final_ckpt}")

    finally:
        log_fh.close()
        for worker in workers:
            worker.disconnect()
