"""Rearrange existing 2x2 m terrain cells into two evaluation-only USDs.

Uses pxr only: no Isaac Sim application, rendering, physics or training.
Every donor surface/material is preserved; each donor is used once per map.
"""

from collections import defaultdict
import hashlib
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from pxr import Gf, Sdf, Usd, UsdGeom

from multi_terrain_test6.config import BinaryCase
from multi_terrain_test6.heldout_layouts import HELDOUT_BY_NAME, MIXED_1
from multi_terrain_test6.layouts import ROCKS
from multi_terrain_test6.terrain import read_terrain_codes


def validate(layout, source_hash: str) -> None:
    read_terrain_codes(BinaryCase(layout.name, layout.usd, layout.grid, (0., 0.), (4., 4.)))
    stage = Usd.Stage.Open(str(layout.usd))
    metadata = stage.GetRootLayer().customLayerData
    if metadata.get("source_sha256") != source_hash or metadata.get("layout") != layout.name:
        raise RuntimeError(f"Unexpected source metadata in {layout.usd}")
    for prim in stage.Traverse():
        paths = [p for rel in prim.GetRelationships() for p in rel.GetTargets()]
        paths += [p for attr in prim.GetAttributes() for p in attr.GetConnections()]
        for path in paths:
            if path.IsAbsolutePath() and not stage.GetPrimAtPath(path.GetPrimPath()).IsValid():
                raise RuntimeError(f"Broken USD binding/connection: {prim.GetPath()} -> {path}")


def ensure_heldout_map(name: str) -> Path:
    layout = HELDOUT_BY_NAME[name]
    source_hash = hashlib.sha256(ROCKS.usd.read_bytes()).hexdigest()
    if layout.usd.exists():
        validate(layout, source_hash)
        return layout.usd
    source = Usd.Stage.Open(str(ROCKS.usd))
    donors = defaultdict(list)
    for cell in source.GetPrimAtPath("/World/Grid").GetChildren():
        donors[cell.GetName().split("_", 3)[3]].append(cell)
    # Build one assignment; transpose its indices for map 2 so corresponding
    # cells use exactly the same donor mesh, without rescaling or recropping.
    used = defaultdict(int)
    plan = {}
    for row, terrains in enumerate(MIXED_1.grid):
        for col, terrain in enumerate(terrains):
            plan[row, col] = donors[terrain][used[terrain]]
            used[terrain] += 1
    if name != MIXED_1.name:
        plan = {(col, row): donor for (row, col), donor in plan.items()}
    layout.usd.parent.mkdir(parents=True, exist_ok=True)
    temporary = layout.usd.with_name(layout.usd.stem + ".building.usd")
    if temporary.exists():
        raise RuntimeError(f"An unfinished map already exists: {temporary}")
    try:
        stage = Usd.Stage.CreateNew(str(temporary))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source))
        UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source))
        if not Sdf.CopySpec(source.GetRootLayer(), Sdf.Path("/World"),
                            stage.GetRootLayer(), Sdf.Path("/World")):
            raise RuntimeError("Could not copy source World/materials")
        stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
        for cell in list(stage.GetPrimAtPath("/World/Grid").GetChildren()):
            stage.RemovePrim(cell.GetPath())
        for (row, col), donor in plan.items():
            terrain = layout.grid[row][col]
            if donor.GetName().split("_", 3)[3] != terrain:
                raise RuntimeError("Donor assignment does not match requested terrain")
            src_path = donor.GetPath()
            dst_path = Sdf.Path(f"/World/Grid/Cell_{row}_{col}_{terrain}")
            if UsdGeom.Xformable(donor).GetOrderedXformOps():
                raise RuntimeError("Expected baked donor meshes without cell transforms")
            if not Sdf.CopySpec(source.GetRootLayer(), src_path, stage.GetRootLayer(), dst_path):
                raise RuntimeError(f"Could not copy {src_path}")
            cell = stage.GetPrimAtPath(dst_path)
            cell.SetTypeName("Xform")
            source_row, source_col = map(int, donor.GetName().split("_")[1:3])
            UsdGeom.Xformable(cell).AddTranslateOp(opSuffix="heldout").Set(
                Gf.Vec3d(2. * (col - source_col), 2. * (row - source_row), 0.)
            )
            for prim in Usd.PrimRange(cell):
                for rel in prim.GetRelationships():
                    targets = rel.GetTargets()
                    if targets:
                        rel.SetTargets([p.ReplacePrefix(src_path, dst_path) if p.HasPrefix(src_path) else p
                                        for p in targets])
                for attr in prim.GetAttributes():
                    targets = attr.GetConnections()
                    if targets:
                        attr.SetConnections([p.ReplacePrefix(src_path, dst_path) if p.HasPrefix(src_path) else p
                                             for p in targets])
        stage.GetRootLayer().customLayerData = {
            "evaluation_only": True, "layout": name,
            "source_usd": str(ROCKS.usd), "source_sha256": source_hash,
        }
        stage.GetRootLayer().Save()
        # Validate geometry and bindings before publishing the new local map.
        from dataclasses import replace
        validate(replace(layout, usd=temporary), source_hash)
        stage = None
        temporary.rename(layout.usd)
    except BaseException:
        stage = None
        if temporary.exists():
            temporary.unlink()
        raise
    validate(layout, source_hash)
    return layout.usd


if __name__ == "__main__":
    for layout_name in HELDOUT_BY_NAME:
        print(f"[HELDOUT-USD] ready: {ensure_heldout_map(layout_name)}", flush=True)
