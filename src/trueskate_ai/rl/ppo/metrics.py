"""Aggregate metrics over a batch of PPO rollout results.

`run_training` collects one `RolloutResult` per sample per update; this module
reduces that list into a single `RolloutSummary` of rates and per-field means.
Kept separate from the trainer so the training loop only coordinates strategy.
"""
from __future__ import annotations

from dataclasses import dataclass

from trueskate_ai.rl.ppo.collector import RolloutResult


@dataclass(frozen=True)
class RolloutSummary:
    """Reduced metrics for one update's batch of rollout results.

    Values are raw (unrounded); callers round at the logging/display site.
    """

    n_samples: int
    mean_reward: float
    max_reward: float
    detection_rate: float
    landed_rate: float
    match_rate: float
    error_rate: float
    hindsight_rate: float
    mean_action_exec_s: float
    mean_reward_eval_s: float
    mean_eval_total_s: float
    mean_post_eval_wait_s: float
    mean_reset_s: float
    mean_capture_attempts: float
    mean_capture_elapsed_s: float
    mean_skipped_captures: float
    mean_monitor_frames_checked: float
    mean_monitor_elapsed_s: float
    device_summary: dict[str, dict[str, int]]


def _device_summary(results: list[RolloutResult]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for r in results:
        stats = summary.setdefault(
            r.device_id,
            {"samples": 0, "detected": 0, "landed": 0, "matched": 0, "errors": 0},
        )
        stats["samples"] += 1
        if r.detected_trick is not None:
            stats["detected"] += 1
        if r.detected_status == "landed":
            stats["landed"] += 1
        if r.reward > 0.0:
            stats["matched"] += 1
        if r.error is not None:
            stats["errors"] += 1
    return summary


def summarize_rollouts(
    results: list[RolloutResult], *, hindsight_added_count: int
) -> RolloutSummary:
    """Reduce a batch of rollout results into a RolloutSummary.

    Args:
        results: One RolloutResult per sample in the update's batch.
        hindsight_added_count: Number of hindsight-relabeled samples added this
            update (used only for `hindsight_rate`; not present in `results`).
    """
    n = len(results)

    def mean(values) -> float:
        return sum(values) / n if n else 0.0

    def rate(count: int) -> float:
        return count / n if n else 0.0

    rewards = [r.reward for r in results]
    return RolloutSummary(
        n_samples=n,
        mean_reward=mean(rewards),
        max_reward=max(rewards) if rewards else 0.0,
        detection_rate=rate(sum(1 for r in results if r.detected_trick is not None)),
        landed_rate=rate(sum(1 for r in results if r.detected_status == "landed")),
        match_rate=rate(sum(1 for r in results if r.reward > 0.0)),
        error_rate=rate(sum(1 for r in results if r.error is not None)),
        hindsight_rate=rate(hindsight_added_count),
        mean_action_exec_s=mean(r.action_exec_s for r in results),
        mean_reward_eval_s=mean(r.reward_eval_s for r in results),
        mean_eval_total_s=mean(r.eval_total_s for r in results),
        mean_post_eval_wait_s=mean(r.post_eval_wait_s for r in results),
        mean_reset_s=mean(r.reset_s for r in results),
        mean_capture_attempts=mean(r.capture_attempts for r in results),
        mean_capture_elapsed_s=mean(r.capture_elapsed_s for r in results),
        mean_skipped_captures=mean(r.skipped_captures for r in results),
        mean_monitor_frames_checked=mean(r.monitor_frames_checked for r in results),
        mean_monitor_elapsed_s=mean(r.monitor_elapsed_s for r in results),
        device_summary=_device_summary(results),
    )
