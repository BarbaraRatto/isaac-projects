"""Summarize completed multi-terrain evaluation CSVs; no Isaac Sim required."""

import argparse
import csv
from pathlib import Path
from statistics import mean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    files = sorted(args.results_dir.glob("*episodes.csv"))
    if not files:
        parser.error(f"No episode CSVs found in {args.results_dir}")

    print("layout | split | geometry | arrivals | unsafe | mean energy of arrivals (J) | mean path of arrivals (m)")
    print("--- | --- | --- | ---: | ---: | ---: | ---:")
    for path in files:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            continue
        arrivals = [row for row in rows if int(row["success"])]
        unsafe = sum(row["reason"] == "unsafe" for row in rows)
        energy = f"{mean(float(row['energy_j']) for row in arrivals):.1f}" if arrivals else "—"
        distance = f"{mean(float(row['path_length_m']) for row in arrivals):.2f}" if arrivals else "—"
        print(f"{rows[0]['layout']} | {rows[0]['split']} | {rows[0]['geometry']} | "
              f"{len(arrivals)}/{len(rows)} | {unsafe}/{len(rows)} | {energy} | {distance}")


if __name__ == "__main__":
    main()
