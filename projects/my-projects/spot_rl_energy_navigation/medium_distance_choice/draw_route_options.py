"""Draw the two proposed safe routes over the actual 7x7 terrain grid.

This is a route plan, not a recorded Spot trajectory or an energy result.
"""

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
DEFAULT_GRID = HERE / "results" / "medium_eval_20261007_114130_grid.csv"
DEFAULT_OUTPUT = HERE / "results" / "medium_route_options.png"
WIDTH, HEIGHT = 1900, 1040
MAP_X, MAP_Y, SCALE = 140, 235, 28
CELL_W, CELL_H = 6 * SCALE, 3 * SCALE
MAP_W, MAP_H = 7 * CELL_W, 7 * CELL_H
FONT_FILE = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD_FILE = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

TERRAINS = {
    "t1_asphalt": ("Asfalto", "#bbb9b7"),
    "t2_slippery": ("Ghiaccio", "#bfe2f4"),
    "t3_ramp": ("Rampa", "#e8dcc8"),
    "t4_fine_gravel": ("Ghiaia", "#dfd5c9"),
    "t5_large_rocks": ("Rocce", "#b6aaa2"),
    "t6_obstacles": ("Ostacoli", "#aaa6a3"),
    "t7_stairs": ("Scale", "#eee9df"),
}

# Both alternatives share the same start and goal. Their difference is the
# middle section: asphalt to the west, rocks farther into the grid.
SHARED_START = [(24.0, 3.0), (18.0, 3.0), (12.0, 3.0), (6.0, 3.0), (5.4, 5.8)]
ASPHALT = [(5.4, 5.8), (2.1, 6.5), (1.3, 9.0), (0.4, 11.9)]
ROCKS = [(5.4, 5.8), (5.0, 8.8), (3.6, 9.7), (2.3, 10.0), (0.4, 11.9)]
SHARED_END = [(0.4, 11.9), (0.0, 15.0)]


def font(size, bold=False):
    path = BOLD_FILE if bold else FONT_FILE
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def center(draw, xy, value, fill, face):
    box = draw.textbbox((0, 0), value, font=face)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2),
              value, font=face, fill=fill)


def map_xy(x, y):
    return MAP_X + (x + 3.0) * SCALE, MAP_Y + (19.5 - y) * SCALE


def terrain_grid(path):
    grid = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for entry in csv.DictReader(stream):
            row, col = int(entry["row"]), int(entry["col"])
            terrain = entry["terrain"]
            if (row, col) in grid or terrain not in TERRAINS:
                raise ValueError(f"Invalid terrain cell: {entry}")
            grid[row, col] = terrain
    if len(grid) != 49 or set(grid) != {(r, c) for r in range(7) for c in range(7)}:
        raise ValueError("Expected the complete 7x7 USD terrain grid")
    return grid


def route(draw, points, color):
    pixels = [tuple(round(v) for v in map_xy(x, y)) for x, y in points]
    draw.line(pixels, fill="#ffffff", width=25, joint="curve")
    draw.line(pixels, fill=color, width=15, joint="curve")
    for pixel in (pixels[0], pixels[-1]):
        draw.ellipse((pixel[0] - 7, pixel[1] - 7, pixel[0] + 7, pixel[1] + 7),
                     fill=color, outline="#ffffff", width=2)


def draw_map(grid, output):
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f8fafc")
    draw = ImageDraw.Draw(image)
    draw.text((140, 56), "Due vie possibili: asfalto o rocce", font=font(42, True), fill="#203044")
    draw.text((140, 118), "Partenza (24, 3)  →  goal (0, 15) | griglia letta dall'USD completo",
              font=font(23), fill="#536173")
    for row in range(7):
        for col in range(7):
            label, color = TERRAINS[grid[row, col]]
            x0, y0 = map_xy(col * 6 - 3, row * 3 + 1.5)
            x0, y0 = round(x0), round(y0)
            draw.rectangle((x0, y0, x0 + CELL_W, y0 + CELL_H),
                           fill=color, outline="#758091", width=2)
            draw.text((x0 + 10, y0 + 7), f"r{row} c{col}", font=font(17), fill="#465365")
            center(draw, (x0 + CELL_W / 2, y0 + CELL_H / 2 + 8),
                   label, "#303b49", font(19, True))
    draw.rectangle((MAP_X, MAP_Y, MAP_X + MAP_W, MAP_Y + MAP_H),
                   outline="#314052", width=3)
    for col in range(7):
        x, _ = map_xy(col * 6, 0)
        center(draw, (x, MAP_Y + MAP_H + 23), str(col * 6), "#465365", font(18))
    for row in range(7):
        _, y = map_xy(0, row * 3)
        center(draw, (MAP_X - 26, y), str(row * 3), "#465365", font(18))
    draw.text((MAP_X + MAP_W / 2 - 35, MAP_Y + MAP_H + 49), "X (m)",
              font=font(21, True), fill="#465365")
    draw.text((37, MAP_Y + MAP_H / 2), "Y (m) ↑", font=font(21, True), fill="#465365")

    # Branches first, then the portions shared by both routes.
    route(draw, ASPHALT, "#087fbe")
    route(draw, ROCKS, "#ec7a23")
    route(draw, SHARED_START, "#7c3fb3")
    route(draw, SHARED_END, "#7c3fb3")
    for point, label, color, offset in (
        (SHARED_START[0], "START", "#663092", (18, -47)),
        (SHARED_END[-1], "GOAL", "#174a93", (18, 16)),
    ):
        x, y = map_xy(*point)
        draw.ellipse((x - 13, y - 13, x + 13, y + 13), fill=color, outline="white", width=4)
        tx, ty = x + offset[0], y + offset[1]
        draw.text((tx - 2, ty - 2), label, font=font(23, True), fill="#ffffff",
                  stroke_width=3, stroke_fill="#ffffff")
        draw.text((tx, ty), label, font=font(23, True), fill=color)

    lx = 1380
    draw.text((lx, 239), "Legenda percorsi", font=font(27, True), fill="#203044")
    for y, color, title, description in (
        (315, "#7c3fb3", "Tratto comune", "stessa partenza e stesso goal"),
        (414, "#087fbe", "Via A: asfalto", "devia a ovest nelle celle r2c0 e r3c0"),
        (513, "#ec7a23", "Via B: rocce", "attraversa la cella r3c1"),
    ):
        draw.line((lx, y, lx + 68, y), fill="#ffffff", width=24)
        draw.line((lx, y, lx + 68, y), fill=color, width=14)
        draw.text((lx + 87, y - 20), title, font=font(22, True), fill="#203044")
        draw.text((lx + 87, y + 16), description, font=font(16), fill="#536173")
    draw.text((lx, 634), "Le due vie si separano dopo", font=font(19), fill="#344256")
    draw.text((lx, 665), "la rampa r2c1 e si ricongiungono", font=font(19), fill="#344256")
    draw.text((lx, 696), "prima di raggiungere il goal.", font=font(19), fill="#344256")
    draw.text((140, 944), "Percorsi proposti da confrontare: non sono traiettorie registrate né risultati energetici.",
              font=font(20), fill="#536173")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(output.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    draw_map(terrain_grid(args.grid), args.output)


if __name__ == "__main__":
    main()
