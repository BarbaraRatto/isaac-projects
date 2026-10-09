"""Draw the three proposed routes over the saved USD terrain grid."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from draw_route_options import (
    CELL_H, CELL_W, HEIGHT, MAP_H, MAP_W, MAP_X, MAP_Y, TERRAINS, WIDTH,
    center, font, map_xy, terrain_grid,
)
from three_route_plan import GOAL, START, ROUTE_NAMES, route_points

HERE = Path(__file__).resolve().parent
DEFAULT_GRID = HERE / "results" / "medium_eval_20261007_114130_grid.csv"
DEFAULT_OUTPUT = HERE / "results" / "medium_three_routes_from_sketch.png"
COLORS = {"pink": "#e83483", "orange": "#f07827", "yellow": "#ddba00"}
LABELS = {
    "pink": ("Rosa", "centro ostacoli, fra le scale"),
    "orange": ("Arancione", "rampa e rocce; evita ostacoli"),
    "yellow": ("Gialla", "ovest, poi curva verso l'asfalto"),
}


def draw_path(draw, name):
    points = [tuple(round(value) for value in map_xy(*point))
              for point in route_points(name)]
    draw.line(points, fill="#ffffff", width=23, joint="curve")
    draw.line(points, fill=COLORS[name], width=13, joint="curve")


def render(grid_file: Path, output: Path):
    grid = terrain_grid(grid_file)
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f8fafc")
    draw = ImageDraw.Draw(image)
    draw.text((140, 55), "Tre percorsi proposti nel tuo disegno", font=font(40, True), fill="#203044")
    draw.text((140, 116), "Trascrizione sulla griglia USD completa | START (24, 3) → GOAL (0, 15)",
              font=font(22), fill="#536173")
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

    # The orange and yellow lines share their first leg. Draw yellow underneath
    # orange so both colors remain visible along that common segment.
    for name in ("pink", "yellow", "orange"):
        draw_path(draw, name)
    for point, label, color, offset in (
        (START, "START", "#6833a4", (17, -46)),
        (GOAL, "GOAL", "#173f98", (15, 15)),
    ):
        x, y = map_xy(*point)
        draw.ellipse((x - 13, y - 13, x + 13, y + 13), fill=color, outline="white", width=4)
        draw.text((x + offset[0], y + offset[1]), label, font=font(23, True),
                  fill=color, stroke_width=3, stroke_fill="white")

    lx = 1380
    draw.text((lx, 240), "Le tre linee", font=font(28, True), fill="#203044")
    for index, name in enumerate(ROUTE_NAMES):
        y = 319 + index * 110
        draw.line((lx, y, lx + 69, y), fill="white", width=24)
        draw.line((lx, y, lx + 69, y), fill=COLORS[name], width=13)
        title, description = LABELS[name]
        draw.text((lx + 85, y - 21), title, font=font(23, True), fill="#203044")
        draw.text((lx + 85, y + 15), description, font=font(16), fill="#536173")
    draw.text((lx, 692), "La linea gialla è curva.", font=font(20), fill="#344256")
    draw.text((lx, 731), "Arancione e gialla condividono", font=font(18), fill="#344256")
    draw.text((lx, 762), "il tratto iniziale verso ovest.", font=font(18), fill="#344256")
    draw.text((140, 939), "Coordinate stimate dal disegno prospettico: percorsi da confermare prima del test fisico.",
              font=font(20), fill="#536173")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(output.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.grid, args.output)


if __name__ == "__main__":
    main()
