"""Draw top-down route plots from a recorded medium-distance evaluation.

Uses only the Python standard library. The map comes from the terrain types
read from the USD during evaluation, not from a hand-copied layout.
"""

import argparse
import colorsys
import csv
import math
from collections import defaultdict
from html import escape
from pathlib import Path

TERRAINS = {
    "t1_asphalt": ("A", "Asfalto", "#c4c3c0"),
    "t2_slippery": ("I", "Ghiaccio", "#c8e5f6"),
    "t3_ramp": ("P", "Rampa", "#e8ddcb"),
    "t4_fine_gravel": ("G", "Ghiaia", "#e0d4c5"),
    "t5_large_rocks": ("R", "Rocce", "#bbb0a9"),
    "t6_obstacles": ("O", "Ostacoli", "#aaa5a2"),
    "t7_stairs": ("S", "Scale", "#eee9dd"),
}
CANVAS_W, CANVAS_H = 1840, 920
MAP_X, MAP_Y, SCALE = 120, 150, 30
MAP_W, MAP_H = 42 * SCALE, 21 * SCALE


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; rerun play.py to record the trajectories")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_evaluation(summary_path: Path):
    stem = summary_path.stem
    summaries = {int(row["episode"]): row for row in read_csv(summary_path)}
    if not summaries:
        raise ValueError("The evaluation summary is empty")
    traces = defaultdict(list)
    for row in read_csv(summary_path.with_name(stem + "_trajectory.csv")):
        episode = int(row["episode"])
        point = (int(row["decision"]), float(row["x_m"]), float(row["y_m"]))
        if not all(math.isfinite(value) for value in point):
            raise ValueError(f"Non-finite trajectory point in episode {episode}")
        traces[episode].append(point)
    grid = {}
    for row in read_csv(summary_path.with_name(stem + "_grid.csv")):
        key = int(row["row"]), int(row["col"])
        terrain = row["terrain"]
        if key in grid or terrain not in TERRAINS:
            raise ValueError(f"Invalid or repeated terrain cell {key}: {terrain}")
        grid[key] = terrain
    if len(grid) != 49 or set(grid) != {(r, c) for r in range(7) for c in range(7)}:
        raise ValueError("Expected all 49 terrain cells from the USD")
    if set(traces) != set(summaries):
        raise ValueError("Summary and trajectory episode numbers differ")
    for episode, points in traces.items():
        points.sort(key=lambda point: point[0])
        expected = int(summaries[episode]["decisions"])
        if points[0][0] != 0 or points[-1][0] != expected or len(points) != expected + 1:
            raise ValueError(f"Missing trajectory samples in episode {episode}")
    return summaries, traces, grid


def xy_to_svg(x: float, y: float) -> tuple[float, float]:
    # USD centers: column c at X=6c, row r at Y=3r. Y grows upward.
    return MAP_X + (x + 3.0) * SCALE, MAP_Y + (19.5 - y) * SCALE


def colored_episode(index: int) -> str:
    hue = (0.02 + index * 0.61803398875) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.82, 0.72)
    return "#" + "".join(f"{round(component * 255):02x}" for component in rgb)


def text(x, y, label, size=20, fill="#263142", weight="normal", **extra):
    attrs = " ".join(f'{key.replace("_", "-")}="{escape(str(value))}"' for key, value in extra.items())
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" {attrs}>{escape(str(label))}</text>')


def background(title: str, subtitle: str, grid: dict[tuple[int, int], str]) -> list[str]:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{CANVAS_H}" '
        f'viewBox="0 0 {CANVAS_W} {CANVAS_H}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        text(120, 67, title, 38, "#1b2735", "bold"),
        text(120, 105, subtitle, 19, "#536276"),
    ]
    for row in range(7):
        for col in range(7):
            code, _, color = TERRAINS[grid[(row, col)]]
            left, top = xy_to_svg(col * 6 - 3, row * 3 + 1.5)
            parts.append(f'<rect x="{left:.1f}" y="{top:.1f}" width="{6*SCALE}" '
                         f'height="{3*SCALE}" fill="{color}" stroke="#7b8490" stroke-width="1"/>')
            parts.append(text(left + 8, top + 19, f"{row},{col}", 12, "#526071"))
            parts.append(text(left + 6*SCALE/2, top + 3*SCALE/2 + 13, code, 36, "#465262",
                              "bold", text_anchor="middle", opacity="0.42"))
    parts.append(f'<rect x="{MAP_X}" y="{MAP_Y}" width="{MAP_W}" height="{MAP_H}" '
                 'fill="none" stroke="#314050" stroke-width="2"/>')
    for col in range(7):
        px, _ = xy_to_svg(6 * col, 0)
        parts.append(text(round(px, 1), MAP_Y + MAP_H + 28, str(6 * col), 14, "#475569", text_anchor="middle"))
    for row in range(7):
        _, py = xy_to_svg(0, 3 * row)
        parts.append(text(97, round(py + 5, 1), str(3 * row), 14, "#475569", text_anchor="end"))
    parts.append(text(700, MAP_Y + MAP_H + 54, "X (m)", 16, "#475569", "bold"))
    parts.append(text(38, 470, "Y (m)", 16, "#475569", "bold",
                      transform="rotate(-90 38 470)"))
    parts.append(text(1420, 172, "Terreni", 23, "#1b2735", "bold"))
    for index, (code, label, color) in enumerate(TERRAINS.values()):
        y = 196 + index * 27
        parts.append(f'<rect x="1420" y="{y - 15}" width="20" height="20" fill="{color}" '
                     'stroke="#6b7280" stroke-width="1"/>')
        parts.append(text(1450, y + 1, f"{code}  {label}", 16))
    return parts


def route(parts: list[str], points: list[tuple[int, float, float]], color: str,
          episode: int, outcome: str, opacity: float) -> None:
    coordinates = [xy_to_svg(x, y) for _, x, y in points]
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in coordinates)
    parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" '
                 f'stroke-width="3.6" stroke-linecap="round" stroke-linejoin="round" '
                 f'opacity="{opacity}"><title>Episodio {episode}: {escape(outcome)}</title></polyline>')
    x, y = coordinates[-1]
    parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.3" '
                 f'fill="{color}" stroke="#fff" stroke-width="1.5"/>')


def markers(parts: list[str]) -> None:
    for x, y, label, color, dx, dy in (
        (24.0, 3.0, "START", "#8338bd", 17, -12),
        (0.0, 15.0, "GOAL", "#1464c4", 18, 27),
    ):
        px, py = xy_to_svg(x, y)
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="8" fill="{color}" '
                     'stroke="#fff" stroke-width="2.5"/>')
        parts.append(text(round(px + dx, 1), round(py + dy, 1), label, 17, color, "bold",
                          stroke="#f8fafc", stroke_width="3", paint_order="stroke"))


def finish(parts: list[str], note: str) -> str:
    parts.append(text(120, 863, note, 17, "#526071"))
    parts.append(text(120, 894, "Coordinate XY del corpo di Spot; un punto ogni decisione RL (0,2 s).", 16, "#526071"))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render_trajectory_plots(summary_path: Path) -> tuple[Path, Path]:
    summary_path = Path(summary_path)
    summaries, traces, grid = load_evaluation(summary_path)
    ordered = sorted(summaries)
    arrivals = [episode for episode in ordered if int(summaries[episode]["success"])]
    failures = [episode for episode in ordered if episode not in arrivals]

    parts = background("Percorsi: arrivi e fallimenti",
                       f"{len(ordered)} episodi | {len(arrivals)} arrivi | {len(failures)} fallimenti", grid)
    # Draw successful and failed routes in separate layers so failed endpoints remain visible.
    for episode in arrivals + failures:
        success = episode in arrivals
        route(parts, traces[episode], "#168247" if success else "#cf354c",
              episode, "goal" if success else summaries[episode]["reason"], 0.48 if success else 0.75)
    markers(parts)
    parts.append(text(1420, 435, "Percorsi", 23, "#1b2735", "bold"))
    for y, color, label in ((464, "#168247", f"Arrivi ({len(arrivals)})"),
                            (499, "#cf354c", f"Fallimenti ({len(failures)})")):
        parts.append(f'<line x1="1422" y1="{y - 5}" x2="1464" y2="{y - 5}" '
                     f'stroke="{color}" stroke-width="4"/>')
        parts.append(text(1476, y, label, 17))
    outcomes_path = summary_path.with_name(summary_path.stem + "_routes_outcomes.svg")
    outcomes_path.write_text(finish(parts, "Verde = goal raggiunto; rosso = episodio terminato prima del goal."),
                             encoding="utf-8")

    parts = background("Percorsi riusciti, episodio per episodio",
                       f"{len(arrivals)} arrivi su {len(ordered)} episodi | un colore per arrivo", grid)
    for index, episode in enumerate(arrivals):
        color = colored_episode(index)
        route(parts, traces[episode], color, episode, "goal", 0.85)
    markers(parts)
    parts.append(text(1420, 420, "Arrivi", 23, "#1b2735", "bold"))
    # Two columns keep all 20 possible arrivals visible in the side legend.
    rows_per_column = max(10, math.ceil(len(arrivals) / 2))
    for index, episode in enumerate(arrivals):
        color = colored_episode(index)
        x = 1422 + (index // rows_per_column) * 202
        y = 452 + (index % rows_per_column) * 28
        parts.append(f'<line x1="{x}" y1="{y - 5}" x2="{x + 34}" y2="{y - 5}" '
                     f'stroke="{color}" stroke-width="4"/>')
        parts.append(text(x + 44, y, f"Episodio {episode}", 16))
    if not arrivals:
        parts.append(text(1420, 454, "Nessun arrivo in questa valutazione", 17))
    successes_path = summary_path.with_name(summary_path.stem + "_routes_successes.svg")
    successes_path.write_text(finish(parts, "Ogni linea colorata rappresenta un episodio concluso al goal."),
                              encoding="utf-8")
    return outcomes_path, successes_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True,
                        help="CSV summary written by medium_distance_choice/play.py")
    args = parser.parse_args()
    for path in render_trajectory_plots(args.summary):
        print(f"[PLOT] {path.resolve()}")


if __name__ == "__main__":
    main()
