"""Local test9 episode, timing and training summaries."""

import csv
from collections import defaultdict
from pathlib import Path
import time
from stable_baselines3.common.callbacks import BaseCallback
from multi_terrain_test9.config import TERRAIN_LABELS
from multi_terrain_test9.utils import reason

def episode_summary(path: Path) -> dict:
    """Summarize the saved episodes by condition without another simulation."""
    groups = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            groups[f"{row['layout']}__{row['geometry']}"].append(row)
    summary = {}
    for name, rows in groups.items():
        arrivals = [row for row in rows if row["success"] == "1"]
        summary[name] = {
            "episodes": len(rows), "arrivals": len(arrivals),
            "mean_arrival_energy_j": (sum(float(row["energy_j"]) for row in arrivals)
                                      / len(arrivals) if arrivals else None),
            "mean_arrival_path_m": (sum(float(row["path_length_m"]) for row in arrivals)
                                    / len(arrivals) if arrivals else None),
        }
    return summary


class TrainingLog(BaseCallback):
    def __init__(self, run_dir: Path, target: int, conditions, initial_steps: int = 0):
        super().__init__()
        self.run_dir = run_dir
        self.target = target
        self.conditions = conditions
        self.previous_steps = initial_steps
        self.start_time = self.previous_time = 0.0
        self.visual_ms = self.dino_ms = 0.0
        self.count = self.episodes = self.successes = 0

    def _on_training_start(self) -> None:
        self.start_time = self.previous_time = time.perf_counter()
        self.timing_handle = (self.run_dir / "timing.csv").open("w", newline="", encoding="utf-8")
        self.timing = csv.writer(self.timing_handle)
        self.timing.writerow(("timesteps", "wall_seconds", "interval_seconds",
                              "transitions_per_second", "mean_dino_ms_per_robot",
                              "mean_visual_ms_per_robot", "eta_seconds"))
        self.timing_handle.flush()
        self.episode_handle = (self.run_dir / "episodes.csv").open("w", newline="", encoding="utf-8")
        self.episode_writer = csv.writer(self.episode_handle)
        self.episode_writer.writerow(("timesteps", "env_id", "layout", "geometry",
                                      "pair", "costly_side", "reward", "actions", "success",
                                      "reason", "energy_j", "path_length_m", "final_distance_m",
                                      *(f"{label}_entry" for label in TERRAIN_LABELS)))

    @staticmethod
    def value(info, key):
        value = info[key]
        return float(value.item()) if hasattr(value, "item") else float(value)

    def _on_step(self) -> bool:
        for env_id, info in enumerate(self.locals["infos"]):
            self.visual_ms += self.value(info, "visual_ms")
            self.dino_ms += self.value(info, "dino_ms")
            self.count += 1
            episode = info.get("episode")
            if episode:
                condition = self.conditions[env_id]
                success = self.value(info, "is_success")
                self.episodes += 1
                self.successes += int(success)
                outcome = reason(info)
                self.episode_writer.writerow((
                    self.num_timesteps, env_id, condition.layout.name, condition.geometry,
                    "/".join(condition.layout.pair), condition.layout.costly_side,
                    episode["r"], episode["l"], int(success), outcome,
                    self.value(info, "episode_energy_j"), self.value(info, "path_length_m"),
                    self.value(info, "final_distance_m"),
                    *(int(self.value(info, f"{label}_entry")) for label in TERRAIN_LABELS),
                ))
                self.episode_handle.flush()
        if self.num_timesteps - self.previous_steps >= 1024:
            now = time.perf_counter()
            elapsed = now - self.previous_time
            rate = (self.num_timesteps - self.previous_steps) / elapsed
            remaining = max(0, self.target - self.num_timesteps) / rate
            row = (self.num_timesteps, round(now - self.start_time, 2), round(elapsed, 2),
                   round(rate, 3), round(self.dino_ms / self.count, 2),
                   round(self.visual_ms / self.count, 2), round(remaining))
            self.timing.writerow(row)
            self.timing_handle.flush()
            print(f"[TEST9] transitions={row[0]}/{self.target} rate={row[3]:.2f}/s "
                  f"DINO={row[4]:.1f}ms/robot ETA={remaining / 60:.1f}min "
                  f"success={self.successes}/{self.episodes}", flush=True)
            self.previous_steps = self.num_timesteps
            self.previous_time = now
            self.visual_ms = self.dino_ms = 0.0
            self.count = 0
        return True

    def _on_training_end(self) -> None:
        self.close_files()

    def close_files(self) -> None:
        for name in ("timing_handle", "episode_handle"):
            handle = getattr(self, name, None)
            if handle is not None and not handle.closed:
                handle.close()

