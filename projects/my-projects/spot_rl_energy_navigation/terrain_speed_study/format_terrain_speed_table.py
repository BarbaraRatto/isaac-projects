"""Combine the existing 0.5-0.9 m/s benchmark with the newest 1.1-1.6 m/s run.

This script reads CSV files only; it does not start Isaac Sim or run trials.
"""

import argparse
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OLD = RESULTS / "terrain_speeds_20261006_163528_summary.csv"
NEW_SPEEDS = (1.1, 1.3, 1.6)
ALL_SPEEDS = (0.5, 0.7, 0.9) + NEW_SPEEDS
TERRAIN_NAMES = {
    "t1_asphalt": "Asfalto",
    "t2_slippery": "Ghiaccio",
    "t3_ramp": "Rampa",
    "t4_fine_gravel": "Ghiaia fine",
    "t5_large_rocks": "Rocce",
    "t6_obstacles": "Ostacoli",
    "t7_stairs": "Scale",
}


def read_summary(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty summary: {path}")
    return rows


def select_new_summary() -> Path:
    candidates = []
    for path in RESULTS.glob("terrain_speeds_*_summary.csv"):
        if path == OLD:
            continue
        try:
            rows = read_summary(path)
        except (OSError, ValueError):
            continue
        if {float(row["command_vx_m_s"]) for row in rows} == set(NEW_SPEEDS):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No new 1.1/1.3/1.6 m/s summary found in results/")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def validate_raw(summary: Path, expected_speeds: set[float]) -> None:
    raw = summary.with_name(summary.name.removesuffix("_summary.csv") + ".csv")
    with raw.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(TERRAIN_NAMES) * len(expected_speeds) * 3:
        raise ValueError(f"Incomplete trial file: {raw} ({len(rows)} rows)")
    if ({float(row["target_distance_m"]) for row in rows} != {2.0}
            or {float(row["settle_s"]) for row in rows} != {2.0}):
        raise ValueError(f"Distance or settling time differs from the previous benchmark: {raw}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=OLD)
    parser.add_argument("--new", type=Path, default=None,
                        help="New summary CSV; default is newest complete high-speed run")
    args = parser.parse_args()
    old_path = args.old.resolve()
    new_path = (args.new or select_new_summary()).resolve()
    old_rows = read_summary(old_path)
    new_rows = read_summary(new_path)
    validate_raw(old_path, {0.5, 0.7, 0.9})
    validate_raw(new_path, set(NEW_SPEEDS))

    by_key = {}
    cells = {}
    for source, rows, speeds in ((old_path, old_rows, {0.5, 0.7, 0.9}),
                                 (new_path, new_rows, set(NEW_SPEEDS))):
        if len(rows) != len(TERRAIN_NAMES) * len(speeds):
            raise ValueError(f"Expected all seven terrains and {len(speeds)} speeds in {source}")
        for row in rows:
            terrain = row["terrain"]
            speed = float(row["command_vx_m_s"])
            key = terrain, speed
            if terrain not in TERRAIN_NAMES or speed not in speeds or key in by_key:
                raise ValueError(f"Unexpected or duplicated terrain/speed {key} in {source}")
            if int(row["attempted"]) != 3:
                raise ValueError(f"Expected 3 trials for {key} in {source}")
            if terrain in cells and cells[terrain] != row["cell_prim"]:
                raise ValueError(f"Different benchmark cell for {terrain}: {cells[terrain]} vs {row['cell_prim']}")
            cells[terrain] = row["cell_prim"]
            by_key[key] = row

    lines = ["# Consumo meccanico misurato, J/m", "",
             f"Dati precedenti: `{old_path.name}`. Nuove velocità: `{new_path.name}`.",
             "Stesse celle, 2 m di cammino e 2 s di assestamento. Velocità nelle intestazioni = comandi, non velocità effettive.",
             "Grassetto = minor consumo fra le velocità con 3/3 prove completate per quel terreno.", "",
             "| Terreno | " + " | ".join(f"{speed:.1f} m/s".replace(".", ",") for speed in ALL_SPEEDS) + " |",
             "| --- | " + " | ".join("---:" for _ in ALL_SPEEDS) + " |"]
    for terrain, label in TERRAIN_NAMES.items():
        rows = [by_key[(terrain, speed)] for speed in ALL_SPEEDS]
        complete = [row for row in rows if int(row["completed"]) == 3]
        best = (min(complete, key=lambda row: float(row["measured_j_per_m_mean"]))
                if complete else None)
        cells_text = []
        for row in rows:
            completed = int(row["completed"])
            energy = row["measured_j_per_m_mean"]
            value = (f"{float(energy):.1f}".replace(".", ",") if energy else "—")
            cell = f"{value} ({completed}/3)"
            if best is row:
                cell = f"**{cell}**"
            cells_text.append(cell)
        lines.append(f"| {label} | " + " | ".join(cells_text) + " |")
    lines.extend(("", "J/m = energia meccanica misurata divisa per la distanza percorsa; i fallimenti non entrano nella media.",
                  "Le velocità fisiche effettive sono nella colonna `actual_speed_m_s_mean` dei CSV di riepilogo.", ""))
    report = "\n".join(lines)
    output = new_path.with_name(new_path.name.removesuffix("_summary.csv") + "_combined.md")
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"[SPEED-TABLE] saved={output}")


if __name__ == "__main__":
    main()
