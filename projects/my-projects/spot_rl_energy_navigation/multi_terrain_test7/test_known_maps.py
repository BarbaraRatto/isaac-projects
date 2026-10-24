"""Run the trained test7 policy for 20 episodes on each known map.

Run with system Python from my-projects; this launcher starts Isaac Lab twice
in sequence. It never trains or changes the checkpoint. --dry-run only prints
the commands and creates no output directory.
"""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

from multi_terrain_test7.layouts import BY_NAME

DEFAULT_ISAACLAB = Path("/home/isaac/isaaclab/2.3.0/isaaclab.sh")


def summarize(path: Path, expected_episodes: int, layout: str) -> dict:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_episodes or any(row["layout"] != layout for row in rows):
        raise RuntimeError(f"Unexpected episode count or layout in {path}")
    arrivals = [row for row in rows if int(row["success"])]
    return {
        "layout": layout, "episodes": len(rows), "arrivals": len(arrivals),
        "success_rate": len(arrivals) / len(rows),
        "mean_arrival_energy_j": statistics.mean(float(row["energy_j"]) for row in arrivals) if arrivals else None,
        "mean_arrival_path_m": statistics.mean(float(row["path_length_m"]) for row in arrivals) if arrivals else None,
        "mean_final_distance_m": statistics.mean(float(row["final_distance_m"]) for row in rows),
        "outcomes": {reason: sum(row["reason"] == reason for row in rows)
                     for reason in sorted({row["reason"] for row in rows})},
        "episodes_csv": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20, help="episodes per map; default 40 total")
    parser.add_argument("--seed", type=int, default=20000,
                        help="reset RNG seed, different from automatic training evaluation seed 10000")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--agent-device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--isaaclab", type=Path, default=DEFAULT_ISAACLAB)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    checkpoint, isaaclab = args.checkpoint.resolve(), args.isaaclab.resolve()
    if not checkpoint.is_file() or not (checkpoint.parent / "config.json").is_file():
        parser.error("checkpoint and its adjacent config.json must exist")
    if not isaaclab.is_file():
        parser.error(f"Isaac Lab launcher not found: {isaaclab}")
    if args.episodes < 1 or args.torch_threads < 1:
        parser.error("--episodes and --torch-threads must be positive")
    output_dir = HERE / "results" / datetime.now().strftime("known_maps_%Y%m%d_%H%M%S_%f")
    commands = []
    for layout in BY_NAME:
        output = output_dir / f"{layout}.csv"
        command = [str(isaaclab), "-p", str(HERE / "play.py"),
                   "--checkpoint", str(checkpoint), "--layout", layout,
                   "--episodes", str(args.episodes), "--seed", str(args.seed),
                   "--agent-device", args.agent_device, "--torch-threads", str(args.torch_threads),
                   "--output", str(output)]
        if args.headless:
            command.append("--headless")
        commands.append((layout, output, command))
    if args.dry_run:
        for _, _, command in commands:
            print(shlex.join(command), flush=True)
        return
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "checkpoint": str(checkpoint), "episodes_per_map": args.episodes,
        "seed": args.seed, "deterministic": True, "headless": args.headless,
        "layouts": list(BY_NAME), "agent_device": args.agent_device,
        "commands": [command for _, _, command in commands],
    }
    (output_dir / "test_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summaries = []
    for layout, output, command in commands:
        print(f"[TEST7-KNOWN] {layout}: {args.episodes} episodes", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        summaries.append(summarize(output, args.episodes, layout))
        # Preserve completed results even if the second simulation is interrupted.
        (output_dir / "summary.json").write_text(json.dumps({
            **manifest, "results": summaries,
        }, indent=2), encoding="utf-8")
    print("\nMap | Arrivals | Mean arrival energy (J) | Mean arrival path (m)")
    for row in summaries:
        energy = f"{row['mean_arrival_energy_j']:.1f}" if row["arrivals"] else "n/a"
        distance = f"{row['mean_arrival_path_m']:.2f}" if row["arrivals"] else "n/a"
        print(f"{row['layout']} | {row['arrivals']}/{row['episodes']} | {energy} | {distance}")
    print(f"[TEST7-KNOWN] results={output_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[TEST7-KNOWN] interrupted by user", file=sys.stderr)
        raise SystemExit(130)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)
