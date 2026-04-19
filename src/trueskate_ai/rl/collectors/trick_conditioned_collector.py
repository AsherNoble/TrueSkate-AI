"""Parallel rollout collection for trick-conditioned policies."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

from trueskate_ai.rl.device_worker import DeviceWorker
from trueskate_ai.rl.reward import get_conditioned_reward
from trueskate_ai.rl.trick_conditioned_action import execute_action_vector


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
    error: str | None = None


def _collect_one(
    worker: DeviceWorker,
    task: RolloutTask,
    *,
    wait_time: float,
    settle_time: float,
    spin_button_xy: tuple[float, float] | None,
) -> RolloutResult:
    worker.ensure_foreground()
    time.sleep(settle_time)
    execute_action_vector(
        worker.driver,
        task.action,
        device_w=worker.device_w,
        device_h=worker.device_h,
        spin_button_xy=worker.spin_button_xy if spin_button_xy is None else spin_button_xy,
    )
    reward, trick_result = get_conditioned_reward(
        worker.driver,
        target_trick=task.target_trick,
        wait_time=wait_time,
    )
    detected_trick = trick_result.trick if trick_result is not None else None
    detected_status = trick_result.status if trick_result is not None else None
    print(
        f"[{worker.device_id}] [eval {task.eval_num} | update {task.update_idx}] "
        f"target={task.target_trick}  detected={detected_trick}  reward={reward:.2f}  "
        f"status={detected_status}"
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
    )


def collect_rollouts(
    *,
    workers: list[DeviceWorker],
    tasks: list[RolloutTask],
    wait_time: float,
    settle_time: float,
    spin_button_xy: tuple[float, float] | None = None,
) -> list[RolloutResult]:
    """Collect rollout tasks in parallel across available workers."""
    if not workers:
        raise ValueError("At least one DeviceWorker is required for rollout collection")

    results: dict[int, RolloutResult] = {}
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        for batch_start in range(0, len(tasks), len(workers)):
            batch = tasks[batch_start : batch_start + len(workers)]
            futures = {}
            for worker, task in zip(workers, batch):
                f = executor.submit(
                    _collect_one,
                    worker,
                    task,
                    wait_time=wait_time,
                    settle_time=settle_time,
                    spin_button_xy=spin_button_xy,
                )
                futures[f] = (worker, task)

            for future in as_completed(futures):
                worker, task = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
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
                        f"[{worker.device_id}] [eval {task.eval_num} | update {task.update_idx}] "
                        f"target={task.target_trick}  detected=None  reward=0.00  status=error"
                    )
                results[result.sample_idx] = result

            reset_futures = [executor.submit(worker.reset) for worker, _ in futures.values()]
            for reset_future in reset_futures:
                reset_future.result()

    return [results[i] for i in range(len(tasks))]
