"""Crop surfaces from real_terrains.usd into a 3x3 choice board.

Run with Isaac's USD Python, without launching Isaac Sim:
    /home/isaac/isaaclab/2.3.0/_isaac_sim/python.sh \
        terrain_generator/create_binary_choice_asphalt_rocks.py

The user-facing grid is numbered 1..3. USD prim names retain the original
zero-based Cell_<row>_<column>_<terrain> convention. Centers are x/y = 0, 2,
4 m. Each cell is 2 x 2 m: rough terrain is cropped from a central 2 x 2 m
patch of an original cell without changing horizontal or vertical scale.
The flat asphalt and ice surfaces are rebuilt at the smaller footprint.
Visual and physics materials come unchanged from the original USD.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, Vt

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "real_terrains.usd"
OUTPUT = HERE / "binary_choice_asphalt_rocks.usd"
CELL_W = 2.0
CELL_H = 2.0

# Rows are listed from row 1 (y=0) to row 3 (y=4).
GRID = (
    ("t4_fine_gravel", "t5_large_rocks", "t5_large_rocks"),
    ("t1_asphalt", "t2_slippery", "t5_large_rocks"),
    ("t1_asphalt", "t1_asphalt", "t4_fine_gravel"),
)


def moved_path(path: Sdf.Path, source: Sdf.Path, destination: Sdf.Path) -> Sdf.Path:
    return path.ReplacePrefix(source, destination) if path.HasPrefix(source) else path


def crop_mesh(mesh: UsdGeom.Mesh, source_center: tuple[float, float],
              target_center: tuple[float, float], flat: bool,
              minimum_top_faces: int = 10000) -> None:
    """Keep an unscaled 2 m square of the original top surface and add skirts."""
    sx, sy = source_center
    tx, ty = target_center
    half = CELL_W / 2.0
    if flat:
        top = [Gf.Vec3f(tx - half, ty - half, 0.0),
               Gf.Vec3f(tx + half, ty - half, 0.0),
               Gf.Vec3f(tx + half, ty + half, 0.0),
               Gf.Vec3f(tx - half, ty + half, 0.0)]
        triangles = [(0, 1, 2), (0, 2, 3)]
    else:
        original = mesh.GetPointsAttr().Get()
        face_sizes = mesh.GetFaceVertexCountsAttr().Get()
        face_indices = mesh.GetFaceVertexIndicesAttr().Get()
        if any(size != 3 for size in face_sizes):
            raise RuntimeError(f"Expected triangles at {mesh.GetPath()}")
        xmin, xmax = sx - half, sx + half
        ymin, ymax = sy - half, sy + half
        top: list[Gf.Vec3f] = []
        triangles: list[tuple[int, int, int]] = []
        vertex_lookup: dict[tuple[float, float, float], int] = {}

        def top_vertex(point: tuple[float, float, float]) -> int:
            translated = (point[0] - sx + tx, point[1] - sy + ty, point[2])
            key = tuple(round(v, 6) for v in translated)
            if key not in vertex_lookup:
                vertex_lookup[key] = len(top)
                top.append(Gf.Vec3f(*translated))
            return vertex_lookup[key]

        def clip(poly: list[tuple[float, float, float]], axis: int,
                 limit: float, keep_greater: bool) -> list[tuple[float, float, float]]:
            if not poly:
                return []
            def inside(point: tuple[float, float, float]) -> bool:
                return point[axis] >= limit - 1e-7 if keep_greater else point[axis] <= limit + 1e-7
            def intersection(a: tuple[float, float, float],
                             b: tuple[float, float, float]) -> tuple[float, float, float]:
                fraction = (limit - a[axis]) / (b[axis] - a[axis])
                result = [a[i] + fraction * (b[i] - a[i]) for i in range(3)]
                result[axis] = limit
                return tuple(result)
            result = []
            previous = poly[-1]
            previous_inside = inside(previous)
            for current in poly:
                current_inside = inside(current)
                if current_inside != previous_inside:
                    result.append(intersection(previous, current))
                if current_inside:
                    result.append(current)
                previous, previous_inside = current, current_inside
            return result

        for i in range(0, len(face_indices), 3):
            face = [tuple(original[j]) for j in face_indices[i:i + 3]]
            if min(p[2] for p in face) < -0.0001:
                continue  # exclude the original perimeter skirts and bottom
            if (max(p[0] for p in face) < xmin or min(p[0] for p in face) > xmax
                    or max(p[1] for p in face) < ymin or min(p[1] for p in face) > ymax):
                continue
            poly = face
            for axis, limit, greater in ((0, xmin, True), (0, xmax, False),
                                         (1, ymin, True), (1, ymax, False)):
                poly = clip(poly, axis, limit, greater)
            for j in range(1, len(poly) - 1):
                a, b, c = poly[0], poly[j], poly[j + 1]
                area2 = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                if abs(area2) > 1e-10:
                    triangles.append((top_vertex(a), top_vertex(b), top_vertex(c)))
        if len(triangles) < minimum_top_faces:
            raise RuntimeError(f"Too few top faces in central patch of {mesh.GetPath()}: {len(triangles)}")

    # Original terrain cells have a skirt down to z=-1. Rebuild it at the
    # new perimeter so the clipped terrain remains a closed collision mesh.
    points = list(top)
    lower: dict[int, int] = {}
    def bottom(index: int) -> int:
        if index not in lower:
            lower[index] = len(points)
            p = top[index]
            points.append(Gf.Vec3f(p[0], p[1], -1.0))
        return lower[index]

    eps = 0.0001
    edges = (
        (0, tx - half, 1, False),  # west: y ascending
        (0, tx + half, 1, True),   # east: y descending
        (1, ty - half, 0, True),   # south: x descending
        (1, ty + half, 0, False),  # north: x ascending
    )
    for axis, value, sort_axis, descending in edges:
        border = sorted((j for j, p in enumerate(top) if abs(p[axis] - value) < eps),
                        key=lambda j: top[j][sort_axis], reverse=descending)
        if len(border) < 2:
            raise RuntimeError(f"Missing perimeter vertices at {mesh.GetPath()}")
        for a, b in zip(border, border[1:]):
            triangles.append((a, b, bottom(b)))
            triangles.append((a, bottom(b), bottom(a)))
    # The underside follows every skirt vertex; a single large quad would
    # leave T-junctions at the skirt's 2 cm segments.
    south = sorted((j for j, p in enumerate(top) if abs(p[1] - (ty - half)) < eps),
                   key=lambda j: top[j][0])
    east = sorted((j for j, p in enumerate(top) if abs(p[0] - (tx + half)) < eps),
                  key=lambda j: top[j][1])
    north = sorted((j for j, p in enumerate(top) if abs(p[1] - (ty + half)) < eps),
                   key=lambda j: top[j][0], reverse=True)
    west = sorted((j for j, p in enumerate(top) if abs(p[0] - (tx - half)) < eps),
                  key=lambda j: top[j][1], reverse=True)
    perimeter = south + east[1:] + north[1:] + west[1:-1]
    center_bottom = len(points)
    points.append(Gf.Vec3f(tx, ty, -1.0))
    for a, b in zip(perimeter, perimeter[1:] + perimeter[:1]):
        triangles.append((center_bottom, bottom(b), bottom(a)))

    mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(triangles)))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([j for face in triangles for j in face]))
    mesh.GetSubdivisionSchemeAttr().Set("none")


def generate(source_file: Path, output_file: Path, overwrite: bool = False,
             grid: tuple[tuple[str, ...], ...] = GRID) -> None:
    source_file = source_file.resolve()
    output_file = output_file.resolve()
    if source_file == output_file:
        raise ValueError("Output must differ from the original real_terrains.usd")
    if not source_file.is_file():
        raise FileNotFoundError(source_file)
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"{output_file} exists; use --overwrite to regenerate it")
    source = Usd.Stage.Open(str(source_file))
    if source is None:
        raise RuntimeError(f"Could not open {source_file}")
    source_grid = source.GetPrimAtPath("/World/Grid")
    if not source_grid.IsValid():
        raise RuntimeError("Original USD has no /World/Grid")

    donors: dict[str, list[Usd.Prim]] = defaultdict(list)
    original_cells = list(source_grid.GetChildren())
    for cell in original_cells:
        parts = cell.GetName().split("_", 3)
        if len(parts) != 4 or parts[0] != "Cell":
            raise RuntimeError(f"Unexpected cell name: {cell.GetPath()}")
        donors[parts[3]].append(cell)
    if len(grid) != 3 or any(len(row) != 3 for row in grid):
        raise ValueError("Expected a 3x3 grid")
    for terrain in {t for row in grid for t in row}:
        needed = sum(row.count(terrain) for row in grid)
        if len(donors[terrain]) < needed:
            raise RuntimeError(f"Only {len(donors[terrain])} original {terrain} cells; need {needed}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_file))
    if stage is None:
        raise RuntimeError(f"Could not create {output_file}")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source))
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source))
    if not Sdf.CopySpec(source.GetRootLayer(), Sdf.Path("/World"),
                        stage.GetRootLayer(), Sdf.Path("/World")):
        raise RuntimeError("Could not copy the original World prim")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    for cell in list(stage.GetPrimAtPath("/World/Grid").GetChildren()):
        stage.RemovePrim(cell.GetPath())

    used: dict[str, int] = defaultdict(int)
    for row, terrains in enumerate(grid):
        for col, terrain in enumerate(terrains):
            donor = donors[terrain][used[terrain]]
            used[terrain] += 1
            source_path = donor.GetPath()
            destination_path = Sdf.Path(f"/World/Grid/Cell_{row}_{col}_{terrain}")
            if not Sdf.CopySpec(source.GetRootLayer(), source_path,
                                stage.GetRootLayer(), destination_path):
                raise RuntimeError(f"Could not copy {source_path} to {destination_path}")
            cell = stage.GetPrimAtPath(destination_path)
            source_row, source_col = (int(v) for v in donor.GetName().split("_")[1:3])
            source_center = (source_col * 6.0, source_row * 3.0)
            target_center = (col * CELL_W, row * CELL_H)
            mesh_count = 0
            for prim in Usd.PrimRange(cell):
                for relationship in prim.GetRelationships():
                    targets = relationship.GetTargets()
                    if targets:
                        relationship.SetTargets([moved_path(p, source_path, destination_path)
                                                 for p in targets])
                for attribute in prim.GetAttributes():
                    connections = attribute.GetConnections()
                    if connections:
                        attribute.SetConnections([moved_path(p, source_path, destination_path)
                                                  for p in connections])
                if prim.IsA(UsdGeom.Mesh):
                    mesh = UsdGeom.Mesh(prim)
                    crop_mesh(mesh, source_center, target_center,
                              flat=terrain in {"t1_asphalt", "t2_slippery"},
                              minimum_top_faces=4 if terrain == "t7_stairs" else 10000)
                    mesh_count += 1
            if mesh_count == 0:
                raise RuntimeError(f"No mesh in {destination_path}")
            print(f"[{row + 1},{col + 1}] {terrain} <= central 2x2 m of {donor.GetName()}")

    # Keep only materials used by the selected terrain types.
    materials = stage.GetPrimAtPath("/World/Materials")
    selected = {t for row in grid for t in row}
    for material in list(materials.GetChildren()):
        if material.GetName().removesuffix("_mat") not in selected:
            stage.RemovePrim(material.GetPath())
    stage.GetRootLayer().Save()

    reopened = Usd.Stage.Open(str(output_file))
    cells = list(reopened.GetPrimAtPath("/World/Grid").GetChildren())
    if len(cells) != 9:
        raise RuntimeError(f"Expected 9 cells in saved USD, found {len(cells)}")
    for row, terrains in enumerate(grid):
        for col, terrain in enumerate(terrains):
            path = Sdf.Path(f"/World/Grid/Cell_{row}_{col}_{terrain}")
            cell = reopened.GetPrimAtPath(path)
            if not cell.IsValid():
                raise RuntimeError(f"Missing {path}")
            for prim in Usd.PrimRange(cell):
                for relationship in prim.GetRelationships():
                    for target in relationship.GetTargets():
                        if target.IsAbsolutePath() and not reopened.GetPrimAtPath(target.GetPrimPath()).IsValid():
                            raise RuntimeError(f"Broken relationship {prim.GetPath()} -> {target}")
                for attribute in prim.GetAttributes():
                    for target in attribute.GetConnections():
                        if target.IsAbsolutePath() and not reopened.GetPrimAtPath(target.GetPrimPath()).IsValid():
                            raise RuntimeError(f"Broken connection {prim.GetPath()} -> {target}")
    print(f"Saved and checked: {output_file} ({output_file.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    generate(args.source, args.output, args.overwrite)


if __name__ == "__main__":
    main()
