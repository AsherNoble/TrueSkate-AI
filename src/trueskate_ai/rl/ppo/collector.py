"""Parallel rollout collection for trick-conditioned policies."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace

import numpy as np

from trueskate_ai.rl.device_worker import DeviceWorker, _ALL_DEAD_TIMEOUT
from trueskate_ai.rl.reward import (
    ContinuousTrickMonitor,
    compute_conditioned_reward,
    get_conditioned_reward,
    merge_trick_results,
)
from trueskate_ai.rl.ppo.trick_conditioned_action import execute_gesture_params_vector

_TARGET_COL_WIDTH = 28
_DETECTED_COL_WIDTH = 32


def _fmt_col(value: str | None, width: int) -> str:
    text = "None" if value is None else value
    if len(text) > width:
        return f"{text[: max(0, width - 3)]}..."
    return text.ljust(width)


def _format_eval_line(
    *,
    device_id: str,
    eval_num: int,
    update_idx: int,
    target_trick: str,
    detected_trick: str | None,
    reward: float,
    status: str | None,
) -> str:
    target_col = _fmt_col(target_trick, _TARGET_COL_WIDTH)
    detected_col = _fmt_col(detected_trick, _DETECTED_COL_WIDTH)
    status_col = _fmt_col(status, 8)
    return (
        f"[{device_id}] [eval {eval_num:5d} | update {update_idx:4d}] "
        f"target={target_col}  detected={detected_col}  reward={reward:5.2f}  status={status_col}"
    )


@dataclass(frozen=True)
class RolloutTask:
    sample_idx: int
    action: np.ndarray
    target_trick: str
    eval_num: int
    update_idx: int


@dataclass(frozen=True)
class RolloutResult:
    sample_idx: int
    eval_num: int
    update_idx: int
    reward: float
    target_trick: str
    detected_trick: str | None
    detected_status: str | None
    device_id: str
    action_exec_s: float = 0.0
    reward_eval_s: float = 0.0
    eval_total_s: float = 0.0
    post_eval_wait_s: float = 0.0
    reset_s: float = 0.0
    capture_attempts: int = 0
    skipped_captures: int = 0
    detection_capture_idx: int | None = None
    capture_elapsed_s: float = 0.0
    monitor_frames_checked: int = 0
    monitor_elapsed_s: float = 0.0
    error: str | None = None


def _timed_reset(worker: DeviceWorker) -> tuple[str, float, float]:
    reset_started_at = time.monotonic()
    worker.reset()
    return worker.device_id, reset_started_at, time.monotonic() - reset_started_at


def _collect_one(
    worker: DeviceWorker,
    task: RolloutTask,
    *,
    wait_time: float,
    settle_time: float,
    capture_count: int,
    capture_interval: float,
    spin_button_xy: tuple[float, float] | None,
) -> RolloutResult:
    eval_start = time.monotonic()
    worker.ensure_foreground()
    time.sleep(settle_time)
    monitor = ContinuousTrickMonitor()

    def _on_post_push() -> None:
        monitor.start(worker.mjpeg_url)

    action_start_time = time.monotonic()
    execute_gesture_params_vector(
        worker.driver,
        task.action,
        device_w=worker.device_w,
        device_h=worker.device_h,
        spin_button_xy=worker.spin_button_xy if spin_button_xy is None else spin_button_xy,
        on_post_push=_on_post_push,
    )
    action_end_time = time.monotonic()
    monitor_result, monitor_diag = monitor.stop()
    reward, trick_result, capture_diag = get_conditioned_reward(
        worker.driver,
        target_trick=task.target_trick,
        wait_time=wait_time,
        capture_count=capture_count,
        capture_interval=capture_interval,
        action_start_time=action_end_time,
        return_diagnostics=True,
    )
    combined_result = merge_trick_results(monitor_result, trick_result)
    reward = compute_conditioned_reward(combined_result, target_trick=task.target_trick)
    reward_end_time = time.monotonic()
    detected_trick = combined_result.trick if combined_result is not None else None
    detected_status = combined_result.status if combined_result is not None else None
    print(
        _format_eval_line(
            device_id=worker.device_id,
            eval_num=task.eval_num,
            update_idx=task.update_idx,
            target_trick=task.target_trick,
            detected_trick=detected_trick,
            reward=reward,
            status=detected_status,
        )
    )
    return RolloutResult(
        sample_idx=task.sample_idx,
        eval_num=task.eval_num,
        update_idx=task.update_idx,
        reward=reward,
        target_trick=task.target_trick,
        detected_trick=detected_trick,
        detected_status=detected_status,
        device_id=worker.device_id,
        action_exec_s=action_end_time - action_start_time,
        reward_eval_s=reward_end_time - action_end_time,
        eval_total_s=reward_end_time - eval_start,
        capture_attempts=int(capture_diag["captures_attempted"]) + int(monitor_diag["frames_checked"]),
        skipped_captures=int(capture_diag["skipped_captures"]),
        detection_capture_idx=(
            None
            if capture_diag["detection_capture_idx"] is None
            else int(capture_diag["detection_capture_idx"])
        ),
        capture_elapsed_s=float(capture_diag["capture_elapsed_s"]),
        monitor_frames_checked=int(monitor_diag["frames_checked"]),
        monitor_elapsed_s=float(monitor_diag["monitor_elapsed_s"]),
    )


def collect_rollouts(
    *,
    workers: list[DeviceWorker],
    tasks: list[RolloutTask],
    wait_time: float,
    settle_time: float,
    capture_count: int,
    capture_interval: float,
    spin_button_xy: tuple[float, float] | None = None,
) -> list[RolloutResult]:
    """Collect rollout tasks in parallel across available workers."""
    if not workers:
        raise ValueError("At least one DeviceWorker is required for rollout collection")

    results: dict[int, RolloutResult] = {}
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        task_idx = 0
        while task_idx < len(tasks):
            alive = [w for w in workers if w.alive]
            if not alive:
                logging.warning("All workers dead — retrying with all workers.")
                alive = workers
            batch = tasks[task_idx : task_idx + len(alive)]
            task_idx += len(batch)

            futures = {}
            for worker, task in zip(alive, batch):
                f = executor.submit(
                    _collect_one,
                    worker,
                    task,
                    wait_time=wait_time,
                    settle_time=settle_time,
                    capture_count=capture_count,
                    capture_interval=capture_interval,
                    spin_button_xy=spin_button_xy,
                )
                futures[f] = (worker, task)

            completion_times: dict[int, float] = {}
            reset_futures: dict[object, tuple[str, int]] = {}
            for future in as_completed(futures):
                worker, task = futures[future]
                try:
                    result = future.result()
                    worker._failure_streak = 0
                    worker._dead_since = None
                except Exception as exc:
                    was_alive = worker.alive
                    worker._failure_streak += 1
                    if was_alive and not worker.alive:
                        worker._dead_since = time.monotonic()
                    logging.warning(
                        "[%s] rollout failed (update=%d eval=%d sample=%d): %s",
                        worker.device_id,
                        task.update_idx,
                        task.eval_num,
                        task.sample_idx,
                        exc,
                    )
                    result = RolloutResult(
                        sample_idx=task.sample_idx,
                        eval_num=task.eval_num,
                        update_idx=task.update_idx,
                        reward=0.0,
                        target_trick=task.target_trick,
                        detected_trick=None,
                        detected_status="error",
                        device_id=worker.device_id,
                        error=str(exc),
                    )
                    print(
                        _format_eval_line(
                            device_id=worker.device_id,
                            eval_num=task.eval_num,
                            update_idx=task.update_idx,
                            target_trick=task.target_trick,
                            detected_trick=None,
                            reward=0.0,
                            status="error",
                        )
                    )
                results[result.sample_idx] = result
                completion_times[result.sample_idx] = time.monotonic()
                reset_future = executor.submit(_timed_reset, worker)
                reset_futures[reset_future] = (worker.device_id, result.sample_idx)

            for reset_future in as_completed(reset_futures):
                _, sample_idx = reset_futures[reset_future]
                try:
                    _, reset_started_at, reset_duration = reset_future.result()
                except Exception as exc:
                    logging.warning(
                        "Reset future failed (sample=%d): %s",
                        sample_idx,
                        exc,
                    )
                    reset_started_at = completion_times.get(sample_idx, time.monotonic())
                    reset_duration = 0.0
                post_eval_wait = max(
                    0.0, reset_started_at - completion_times.get(sample_idx, reset_started_at)
                )
                if sample_idx in results:
                    results[sample_idx] = replace(
                        results[sample_idx],
                        post_eval_wait_s=post_eval_wait,
                        reset_s=reset_duration,
                    )

            for w in workers:
                w._try_revive()

            if not any(w.alive for w in workers):
                now = time.monotonic()
                oldest_death = min(
                    (w._dead_since for w in workers if w._dead_since is not None),
                    default=now,
                )
                if now - oldest_death > _ALL_DEAD_TIMEOUT:
                    raise RuntimeError(
                        f"All devices have been unreachable for >"
                        f"{_ALL_DEAD_TIMEOUT / 60:.0f} minutes — aborting."
                    )

    return [results[i] for i in range(len(tasks))]
