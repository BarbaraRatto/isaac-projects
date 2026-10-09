"""Build a small USD variant for the rocks-ramp-rocks energy choice.

Only the detour row is changed: the original rocky start/goal and ramp retain
their geometry, collision and physical materials. The original USD is read-only.
"""

from pathlib import Path

from pxr import Sdf, Usd, UsdGeom, UsdShade


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "terrain_generator" / "real_terrains.usd"
OUTPUT = HERE / "ramp_energy_choice.usda"
SOURCE_RELATIVE = "../../terrain_generator/real_terrains.usd"

KEEP = {
    "Cell_1_3_t5_large_rocks",
    "Cell_1_4_t3_ramp",
    "Cell_1_5_t5_large_rocks",
    "Cell_0_4_t1_asphalt",
}
ASPHALT_TEMPLATE = "/World/Grid/Cell_0_4_t1_asphalt"


def build() -> Path:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    source_stage = Usd.Stage.Open(str(SOURCE))
    for path in (
        "/World/Grid/Cell_1_3_t5_large_rocks",
        "/World/Grid/Cell_1_4_t3_ramp",
        "/World/Grid/Cell_1_5_t5_large_rocks",
        ASPHALT_TEMPLATE,
    ):
        if not source_stage.GetPrimAtPath(path).IsValid():
            raise RuntimeError(f"Required terrain missing from source USD: {path}")

    layer = Sdf.Layer.CreateNew(str(OUTPUT)) if not OUTPUT.exists() else Sdf.Layer.FindOrOpen(str(OUTPUT))
    layer.Clear()
    layer.subLayerPaths.append(SOURCE_RELATIVE)
    stage = Usd.Stage.Open(layer)
    grid = source_stage.GetPrimAtPath("/World/Grid")
    for cell in grid.GetChildren():
        if cell.GetName().startswith("Cell_") and cell.GetName() not in KEEP:
            stage.OverridePrim(str(cell.GetPath())).SetActive(False)

    # Reference the same asphalt cell with translations of one column. Its
    # visual/physical bindings still resolve through the original sublayer.
    for column, offset_x in ((3, -6.0), (5, 6.0)):
        path = f"/World/Grid/Cell_0_{column}_t1_asphalt_detour"
        prim = stage.DefinePrim(path, "Xform")
        prim.GetReferences().AddReference(SOURCE_RELATIVE, ASPHALT_TEMPLATE)
        UsdGeom.Xformable(prim).AddTranslateOp().Set((offset_x, 0.0, 0.0))
        # USD discards absolute visual bindings outside the referenced prim.
        # The internal physics material binding is remapped automatically.
        mesh = stage.GetPrimAtPath(f"{path}/terrain/mesh")
        material = UsdShade.Material.Get(stage, "/World/Materials/t1_asphalt_mat")
        if not mesh.IsValid() or not material:
            raise RuntimeError(f"Missing asphalt mesh or visual material at {path}")
        UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material)

    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    stage.GetRootLayer().Save()
    return OUTPUT


if __name__ == "__main__":
    print(f"[RAMP-SCENE] saved={build()}")
