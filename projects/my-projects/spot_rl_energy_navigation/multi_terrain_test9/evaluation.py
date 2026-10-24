"""Deterministic evaluation, using the existing simulator between SAC rollouts."""

from contextlib import contextmanager
import csv
import json
from pathlib import Path
import random

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from multi_terrain_test9.config import NAV_CFG
from multi_terrain_test9.utils import number, reason, stop_actions


@contextmanager
def evaluation_rng(seed: int):
    """Repeat reset jitter without consuming training's Python/NumPy/Torch RNGs."""
    py_state, np_state, cpu_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def evaluate_vector(model, env, conditions, seed: int) -> list[dict]:
    """One completed episode per robot; automatic-reset later episodes are ignored."""
    with evaluation_rng(seed):
        obs = env.reset()
        active = np.ones(env.num_envs, dtype=bool)
        returns = np.zeros(env.num_envs)
        lengths = np.zeros(env.num_envs, dtype=int)
        rows = []
        for _ in range(NAV_CFG.max_episode_steps + 2):
            actions, _ = model.predict(obs, deterministic=True)
            actions[~active] = stop_actions(int((~active).sum()))
            obs, rewards, dones, infos = env.step(actions)
            returns[active] += rewards[active]
            lengths[active] += 1
            for env_id in np.flatnonzero(active & np.asarray(dones, dtype=bool)):
                info = infos[env_id]
                rows.append({
                    "timesteps": model.num_timesteps, "env_id": int(env_id),
                    "layout": conditions[env_id].layout.name,
                    "success": int(number(info, "is_success")), "reason": reason(info),
                    "reward": float(returns[env_id]), "actions": int(lengths[env_id]),
                    "energy_j": number(info, "episode_energy_j"),
                    "path_length_m": number(info, "path_length_m"),
                    "final_distance_m": number(info, "final_distance_m"),
                })
                active[env_id] = False
            if not active.any():
                return rows
        raise RuntimeError("Evaluation did not terminate within the episode limit")


def summarize(rows: list[dict]) -> tuple[dict, tuple]:
    per_layout = {}
    for layout in sorted({row["layout"] for row in rows}):
        group = [row for row in rows if row["layout"] == layout]
        arrivals = [row for row in group if row["success"]]
        per_layout[layout] = {
            "episodes": len(group), "arrivals": len(arrivals),
            "success_rate": len(arrivals) / len(group),
            "mean_arrival_energy_j": float(np.mean([r["energy_j"] for r in arrivals])) if arrivals else None,
        }
    rates = [item["success_rate"] for item in per_layout.values()]
    arrivals = [r for r in rows if r["success"]]
    # First improve the worse map, then overall success, then successful energy.
    # With zero successes, use distance rather than rewarding early termination.
    tie_break = (-float(np.mean([r["energy_j"] for r in arrivals])) if arrivals
                 else -float(np.mean([r["final_distance_m"] for r in rows])))
    score = (min(rates), float(np.mean(rates)), tie_break)
    return {"per_layout": per_layout, "selection_score": list(score)}, score


class PeriodicEvaluation(BaseCallback):
    """Runs only after the last collected transition has entered the replay buffer.

    Reuses/reset the training simulator, excludes evaluation from replay and
    num_timesteps, and refreshes the learner observation before collection resumes.
    """

    def __init__(self, run_dir: Path, conditions, every: int, seed: int):
        super().__init__()
        self.run_dir, self.conditions, self.every, self.seed = run_dir, conditions, every, seed
        self.next_eval = every
        self.best_score = None

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.every and self.num_timesteps >= self.next_eval:
            self.evaluate_now()
            self.next_eval = self.num_timesteps + self.every

    def evaluate_now(self, phase: str = "periodic") -> None:
        env = self.training_env
        try:
            rows = evaluate_vector(self.model, env, self.conditions, self.seed)
        finally:
            # SB3/SAC stores _last_obs at rollout end; a reset here must update it.
            # No normalization wrapper is used in this experiment.
            obs = env.reset()
            self.model._last_obs = obs
            self.model._last_original_obs = obs.copy()
            self.model._last_episode_starts = np.ones(env.num_envs, dtype=bool)
        for row in rows:
            row["phase"] = phase
        output = self.run_dir / "evaluations.csv"
        exists = output.is_file()
        with output.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            if not exists:
                writer.writeheader()
            writer.writerows(rows)
        summary, score = summarize(rows)
        summary["timesteps"] = self.num_timesteps
        summary["phase"] = phase
        print(f"[TEST9-EVAL] {json.dumps(summary)}", flush=True)
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.model.save(str(self.run_dir / "policy_best"))
            (self.run_dir / "best_evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
