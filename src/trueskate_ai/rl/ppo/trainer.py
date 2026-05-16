"""End-to-end trick-conditioned PPO trainer."""

from __future__ import annotations

import logging
from typing import Any
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from trueskate_ai.rl.ppo.policy import TrickConditionedPolicy
from trueskate_ai.rl.ppo.collector import (
    RolloutTask,
    collect_rollouts,
)
from trueskate_ai.rl.ppo.metrics import summarize_rollouts
from trueskate_ai.rl.device_worker import DEVICES
from trueskate_ai.rl.ppo.buffer import RolloutBatch
from trueskate_ai.rl.run_logger import RunLogger
from trueskate_ai.rl.worker_pool import WorkerPool
from trueskate_ai.rl.reward import normalize_trick_name
from trueskate_ai.sim.trick_info_reader import ensure_ocr_backend_ready

logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

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
    capture_count: int = 14
    capture_interval: float = 0.15
    settle_time: float = 0.5
    checkpoint_every: int = 10
    log_dir: Path = Path("logs")
    spin_x: float | None = None
    spin_y: float | None = None
    device_count: int | None = None
    use_cuda: bool = False
    hindsight_relabel: bool = True
    resume_from: Path | None = None


def _normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    std = advantages.std(unbiased=False)
    if float(std) < 1e-8:
        return advantages - advantages.mean()
    return (advantages - advantages.mean()) / (std + 1e-8)


def _config_to_json_dict(config: PPOConfig) -> dict:
    data = asdict(config)
    data["log_dir"] = str(data["log_dir"])
    data["resume_from"] = str(data["resume_from"]) if data["resume_from"] is not None else None
    return data


def _save_checkpoint(
    *,
    path: Path,
    policy: TrickConditionedPolicy,
    optimizer: Adam,
    config: PPOConfig,
    next_update_idx: int,
    eval_num: int,
    run_metadata: dict[str, Any],
) -> None:
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "trainer_state": {
                "next_update_idx": next_update_idx,
                "eval_num": eval_num,
            },
            "config": _config_to_json_dict(config),
            "run_metadata": run_metadata,
        },
        path,
    )


def _load_resume_checkpoint(
    *,
    checkpoint_path: Path,
    policy: TrickConditionedPolicy,
    optimizer: Adam,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "policy_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["policy_state_dict"])
    elif isinstance(checkpoint, dict):
        # Backward compatibility for raw state_dict-style checkpoint.
        policy.load_state_dict(checkpoint)
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    optimizer_restored = False
    if isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        optimizer_restored = True

    trainer_state = checkpoint.get("trainer_state", {}) if isinstance(checkpoint, dict) else {}
    next_update_idx = int(trainer_state.get("next_update_idx", 0))
    eval_num = int(trainer_state.get("eval_num", 0))
    source_run_metadata = checkpoint.get("run_metadata", {}) if isinstance(checkpoint, dict) else {}

    return {
        "resume_checkpoint": str(checkpoint_path.resolve()),
        "resume_source_run": source_run_metadata.get("run_id"),
        "optimizer_restored": optimizer_restored,
        "next_update_idx": next_update_idx,
        "eval_num": eval_num,
        "checkpoint_config": checkpoint.get("config") if isinstance(checkpoint, dict) else None,
    }


def _extract_relabel_components(
    detected_trick: str, *, target_norm: str, trick_to_idx: dict[str, int]
) -> list[int]:
    """Return relabel trick indices for recognized landed components.

    OCR can return combo strings like "KICKFLIP + 50-50 GRIND". We split by
    " + ", normalize each component, and keep only known trick targets that are
    different from the originally sampled target trick.
    """
    relabel_idxs: list[int] = []
    seen: set[str] = set()
    for component in detected_trick.split(" + "):
        normalized = normalize_trick_name(component)
        if normalized == target_norm or normalized in seen:
            continue
        seen.add(normalized)
        relabeled_idx = trick_to_idx.get(normalized)
        if relabeled_idx is not None:
            relabel_idxs.append(relabeled_idx)
    return relabel_idxs


def run_training(config: PPOConfig) -> None:
    ensure_ocr_backend_ready()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)

    device = torch.device("cuda" if config.use_cuda and torch.cuda.is_available() else "cpu")
    tricks = list(TRICK_LIST)
    trick_to_idx = {normalize_trick_name(trick): idx for idx, trick in enumerate(tricks)}
    workers_cfg = DEVICES if config.device_count is None else DEVICES[: config.device_count]
    pool = WorkerPool(workers_cfg)
    if len(pool) == 0:
        raise RuntimeError("No devices configured for rollout collection")

    pool.connect_all()

    logger = RunLogger(config.log_dir, "ppo")
    run_id, run_dir = logger.run_id, logger.run_dir
    policy = TrickConditionedPolicy(num_tricks=len(tricks)).to(device)
    optimizer = Adam(policy.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)

    run_metadata: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "started_at": datetime.now().isoformat(timespec="milliseconds"),
        "resumed": False,
    }

    start_update_idx = 0
    eval_num = 0
    if config.resume_from is not None:
        resume_info = _load_resume_checkpoint(
            checkpoint_path=config.resume_from,
            policy=policy,
            optimizer=optimizer,
            device=device,
        )
        start_update_idx = int(resume_info["next_update_idx"])
        eval_num = int(resume_info["eval_num"])
        run_metadata["resumed"] = True
        run_metadata["resumed_from"] = resume_info["resume_checkpoint"]
        run_metadata["resume_source_run"] = resume_info["resume_source_run"]
        run_metadata["optimizer_restored"] = resume_info["optimizer_restored"]

        logger.write(
            {
                "type": "resume_start",
                "run_id": run_id,
                "resumed": True,
                "resume_checkpoint": resume_info["resume_checkpoint"],
                "resume_source_run": resume_info["resume_source_run"],
                "optimizer_restored": resume_info["optimizer_restored"],
                "resume_start_update": start_update_idx,
                "resume_start_eval": eval_num,
                "loaded_config_summary": resume_info["checkpoint_config"],
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            },
        )

    spin_override = (
        (config.spin_x, config.spin_y)
        if config.spin_x is not None and config.spin_y is not None
        else None
    )

    try:
        end_update_idx = start_update_idx + config.updates
        for update_idx in range(start_update_idx, end_update_idx):
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
                pool=pool,
                tasks=tasks,
                wait_time=config.wait_time,
                settle_time=config.settle_time,
                capture_count=config.capture_count,
                capture_interval=config.capture_interval,
                spin_button_xy=spin_override,
            )

            sampled_rewards = torch.as_tensor(
                [r.reward for r in rollout_results], dtype=torch.float32, device=device
            )
            hindsight_trick_idxs: list[int] = []
            hindsight_actions: list[torch.Tensor] = []
            if config.hindsight_relabel:
                for result in rollout_results:
                    if result.detected_status != "landed" or result.detected_trick is None:
                        continue
                    target_norm = normalize_trick_name(result.target_trick)
                    relabeled_components = _extract_relabel_components(
                        result.detected_trick,
                        target_norm=target_norm,
                        trick_to_idx=trick_to_idx,
                    )
                    if not relabeled_components:
                        continue
                    source_action = actions[result.sample_idx].detach()
                    for relabeled_idx in relabeled_components:
                        hindsight_trick_idxs.append(relabeled_idx)
                        hindsight_actions.append(source_action)

            trick_idx_batch = trick_idxs.detach()
            action_batch = actions.detach()
            old_log_prob_batch = old_log_probs.detach()
            value_batch = values.detach()
            reward_batch = sampled_rewards

            hindsight_added_count = len(hindsight_trick_idxs)
            if hindsight_added_count > 0:
                relabeled_tricks = torch.as_tensor(
                    hindsight_trick_idxs, dtype=torch.long, device=device
                )
                relabeled_actions = torch.stack(hindsight_actions).to(device)
                with torch.no_grad():
                    relabeled_old_logp, _, relabeled_values = policy.evaluate_actions(
                        relabeled_tricks, relabeled_actions
                    )
                relabeled_rewards = torch.ones(hindsight_added_count, dtype=torch.float32, device=device)

                trick_idx_batch = torch.cat([trick_idx_batch, relabeled_tricks], dim=0)
                action_batch = torch.cat([action_batch, relabeled_actions], dim=0)
                old_log_prob_batch = torch.cat([old_log_prob_batch, relabeled_old_logp.detach()], dim=0)
                value_batch = torch.cat([value_batch, relabeled_values.detach()], dim=0)
                reward_batch = torch.cat([reward_batch, relabeled_rewards], dim=0)

            returns = reward_batch
            advantages = _normalize_advantages(returns - value_batch)

            batch = RolloutBatch(
                trick_idx=trick_idx_batch,
                actions=action_batch,
                old_log_probs=old_log_prob_batch,
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

            summary = summarize_rollouts(
                rollout_results, hindsight_added_count=hindsight_added_count
            )
            effective_batch_size = int(batch.size)

            for result in rollout_results:
                logger.write(
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
                        "action_exec_s": round(result.action_exec_s, 4),
                        "reward_eval_s": round(result.reward_eval_s, 4),
                        "eval_total_s": round(result.eval_total_s, 4),
                        "post_eval_wait_s": round(result.post_eval_wait_s, 4),
                        "reset_s": round(result.reset_s, 4),
                        "capture_attempts": result.capture_attempts,
                        "skipped_captures": result.skipped_captures,
                        "detection_capture_idx": result.detection_capture_idx,
                        "capture_elapsed_s": round(result.capture_elapsed_s, 4),
                        "monitor_frames_checked": result.monitor_frames_checked,
                        "monitor_elapsed_s": round(result.monitor_elapsed_s, 4),
                        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    },
                )

            logger.write(
                {
                    "type": "update_summary",
                    "update": update_idx,
                    "mean_reward": summary.mean_reward,
                    "max_reward": summary.max_reward,
                    "match_rate": round(summary.match_rate, 4),
                    "detection_rate": round(summary.detection_rate, 4),
                    "landed_rate": round(summary.landed_rate, 4),
                    "error_rate": round(summary.error_rate, 4),
                    "hindsight_added_count": hindsight_added_count,
                    "hindsight_rate": round(summary.hindsight_rate, 4),
                    "effective_batch_size": effective_batch_size,
                    "mean_action_exec_s": round(summary.mean_action_exec_s, 4),
                    "mean_reward_eval_s": round(summary.mean_reward_eval_s, 4),
                    "mean_eval_total_s": round(summary.mean_eval_total_s, 4),
                    "mean_post_eval_wait_s": round(summary.mean_post_eval_wait_s, 4),
                    "mean_reset_s": round(summary.mean_reset_s, 4),
                    "mean_capture_attempts": round(summary.mean_capture_attempts, 4),
                    "mean_capture_elapsed_s": round(summary.mean_capture_elapsed_s, 4),
                    "mean_skipped_captures": round(summary.mean_skipped_captures, 4),
                    "mean_monitor_frames_checked": round(summary.mean_monitor_frames_checked, 4),
                    "mean_monitor_elapsed_s": round(summary.mean_monitor_elapsed_s, 4),
                    "device_summary": summary.device_summary,
                    "policy_loss": round(mean_policy_loss, 6),
                    "value_loss": round(mean_value_loss, 6),
                    "entropy": round(mean_entropy, 6),
                    "config": _config_to_json_dict(config) if update_idx == 0 else None,
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                },
            )

            print(
                f"[update {update_idx:04d}] mean_reward={summary.mean_reward:.3f} "
                f"max_reward={summary.max_reward:.3f} "
                f"match_rate={summary.match_rate:.2%} detect_rate={summary.detection_rate:.2%} "
                f"error_rate={summary.error_rate:.2%} hindsight_added={hindsight_added_count} "
                f"policy_loss={mean_policy_loss:.4f} value_loss={mean_value_loss:.4f}"
            )

            if (update_idx + 1) % config.checkpoint_every == 0:
                ckpt_path = run_dir / f"policy_update_{update_idx + 1:04d}.pt"
                _save_checkpoint(
                    path=ckpt_path,
                    policy=policy,
                    optimizer=optimizer,
                    config=config,
                    next_update_idx=update_idx + 1,
                    eval_num=eval_num,
                    run_metadata=run_metadata,
                )
                print(f"Saved checkpoint: {ckpt_path}")

        final_ckpt = run_dir / "policy_final.pt"
        _save_checkpoint(
            path=final_ckpt,
            policy=policy,
            optimizer=optimizer,
            config=config,
            next_update_idx=end_update_idx,
            eval_num=eval_num,
            run_metadata=run_metadata,
        )
        print(f"Saved final checkpoint: {final_ckpt}")

    finally:
        logger.close()
        pool.disconnect_all()
