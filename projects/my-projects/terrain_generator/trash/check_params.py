from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import isaaclab.terrains as terrain_gen
import inspect

# Stampa i campi esatti (nome + default) di ogni config che ci interessa
for cls_name in [
    "TerrainGeneratorCfg",
    "MeshPlaneTerrainCfg",
    "MeshPyramidStairsTerrainCfg",
    "HfRandomUniformTerrainCfg",
    "HfDiscreteObstaclesTerrainCfg",
    "HfWaveTerrainCfg",
]:
    cls = getattr(terrain_gen, cls_name, None)
    print(f"\n=== {cls_name} ===")
    if cls is None:
        print("NON TROVATA in questa versione")
        continue
    for name, field in cls.__dataclass_fields__.items():
        print(f"  {name}: default={field.default!r}")

simulation_app.close()
