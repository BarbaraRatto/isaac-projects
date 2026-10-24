"""Shared single-map launcher for the two held-out layout entry points."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess

from multi_terrain_test6.heldout_layouts import HELDOUT_BY_NAME
from multi_terrain_test6.test_known_maps import DEFAULT_CHECKPOINT, DEFAULT_ISAACLAB, summarize

HERE = Path(__file__).resolve().parent


def main(layout_name: str) -> None:
    if layout_name not in HELDOUT_BY_NAME:
        raise ValueError(layout_name)
    parser = argparse.ArgumentParser(description=f"Test6: 20 episodes on unseen map {layout_name}.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20000)
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
    output_dir = HERE / "results" / datetime.now().strftime(f"heldout_{layout_name}_%Y%m%d_%H%M%S_%f")
    output = output_dir / f"{layout_name}.csv"
    command = [str(isaaclab), "-p", str(HERE / "play.py"),
               "--checkpoint", str(checkpoint), "--layout", layout_name,
               "--episodes", str(args.episodes), "--seed", str(args.seed),
               "--agent-device", args.agent_device, "--torch-threads", str(args.torch_threads),
               "--output", str(output)]
    if args.headless:
        command.append("--headless")
    if args.dry_run:
        print(shlex.join(command), flush=True)
        return
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "checkpoint": str(checkpoint), "layout": layout_name, "split": "heldout",
        "episodes": args.episodes, "seed": args.seed, "deterministic": True,
        "headless": args.headless, "grid": HELDOUT_BY_NAME[layout_name].grid,
        "command": command,
    }
    (output_dir / "test_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[TEST6-HELDOUT] {layout_name}: {args.episodes} episodes", flush=True)
    try:
        subprocess.run(command, cwd=HERE.parents[1], check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)
    result = summarize(output, args.episodes, layout_name)
    (output_dir / "summary.json").write_text(json.dumps({**manifest, "result": result}, indent=2), encoding="utf-8")
    print(f"[TEST6-HELDOUT] arrivals={result['arrivals']}/{result['episodes']} "
          f"mean_arrival_energy_j={result['mean_arrival_energy_j']}", flush=True)
    print(f"[TEST6-HELDOUT] results={output_dir}", flush=True)
