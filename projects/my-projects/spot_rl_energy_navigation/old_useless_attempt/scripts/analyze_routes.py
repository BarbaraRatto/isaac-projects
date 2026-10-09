"""Summarize guided-route outcomes without supplying labels to the RL policy."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


DEFAULT_CSV = Path(__file__).resolve().parent / "results" / "route_baselines.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    with args.csv.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No trials in {args.csv}")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["route"], float(row["speed_cap_m_s"]))].append(row)
    for (route, speed), trials in sorted(grouped.items()):
        completed = [r for r in trials if r["status"] == "ok"]
        energy = f"{mean(float(r['energy_j']) for r in completed):.1f} J" if completed else "n/a"
        print(f"{route} @ {speed:.2f} m/s: {len(completed)}/{len(trials)} arrived; "
              f"mean energy on arrivals: {energy}")
    by_trial = defaultdict(dict)
    for row in rows:
        by_trial[(row["trial"], row.get("lateral_offset_m", "0"))][row["route"]] = row
    paired = [pair for pair in by_trial.values() if set(pair) == {"direct", "bypass"}]
    both_arrived = [pair for pair in paired if all(row["status"] == "ok" for row in pair.values())]
    if both_arrived:
        difference = mean(float(pair["direct"]["energy_j"]) -
                          float(pair["bypass"]["energy_j"]) for pair in both_arrived)
        winner = "bypass" if difference > 0 else "direct"
        print(f"Matched arrivals: {len(both_arrived)}; lower energy: {winner} "
              f"({abs(difference):.1f} J mean difference)")
    print("Only guided, fixed-speed trajectories are compared; failed partial energies are excluded.")


if __name__ == "__main__":
    main()
