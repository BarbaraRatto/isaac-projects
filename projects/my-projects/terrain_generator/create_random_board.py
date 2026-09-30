# =============================================================================
# create_random_board.py
# Genera una scacchiera 7x7 (42m x 21m) usando 7 tipi di terreni.
# Ogni tipo di terreno compare esattamente 7 volte.
#
# Lancia con:
#   /home/isaac/isaaclab/2.3.0/isaaclab.sh -p /home/students/work/barbara/isaac-projects/projects/my-projects/terrain_generator/create_random_board.py
# =============================================================================

# --- 0. AVVIO MOTORE (SEMPRE PER PRIMO!) ---
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import math
import random
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporter, TerrainImporterCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
import isaaclab.sim as sim_utils
import omni.usd
from pxr import UsdGeom, Gf, UsdShade, Sdf, Vt

# =============================================================================
# 1. DEFINIZIONE DEI 7 TERRENI
# =============================================================================
CELL_W = 6.0   # larghezza cella (X)
CELL_H = 3.0   # altezza cella (Y)
ROWS = 7
COLS = 7

TERRAINS = [
    # 1. Asfalto — piano, attrito standard, grigio scuro ruvido
    {
        "name": "t1_asphalt",
        "sub_cfg": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
        "material": RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=0.8, restitution=0.0
        ),
        "color": (0.15, 0.15, 0.15),
        "roughness": 0.9,
    },
    # 2. Ghiaccio (bagnato) — piano, poco attrito, Bianco e lucido?
    {
        "name": "t2_slippery",
        "sub_cfg": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
        "material": RigidBodyMaterialCfg(
            static_friction=0.1, dynamic_friction=0.08, restitution=0.0
        ),
        "color": (0.85, 0.90, 0.95),
        "roughness": 0.2,
    },
    # 3. Rampa 15° — piramide tronca quadrata, attrito standard, grigio cemento
    {
        "name": "t3_ramp",
        "sub_cfg": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0,
            slope_range=(math.tan(math.radians(7.0)), math.tan(math.radians(7.0))),
            platform_width=1.0,
            border_width=0.25,
        ),
        "material": RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=0.8, restitution=0.0
        ),
        "color": (0.45, 0.45, 0.45),
        "roughness": 0.7,
    },
    # 4. Ghiaia fine — sassolini piccoli, piatto con micro-texture
    {
        "name": "t4_fine_gravel",
        "sub_cfg": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(0.005, 0.015),   # max 1.5 cm di irregolarità
            noise_step=0.001,             # passi da 1mm → transizioni lisce
            border_width=0.0,
        ),
        "material": RigidBodyMaterialCfg(
            static_friction=0.4, dynamic_friction=0.3, restitution=0.0
        ),
        "color": (0.60, 0.55, 0.35),
        "roughness": 0.95,
    },
    # 5. Sassi grandi — rocce più grandi ma terreno comunque piatto
    {
        "name": "t5_large_rocks",
        "sub_cfg": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(0.02, 0.05),     # max 5 cm di irregolarità
            noise_step=0.001,             # passi da 1mm → transizioni lisce
            border_width=0.0,
        ),
        "material": RigidBodyMaterialCfg(
            static_friction=0.5, dynamic_friction=0.4, restitution=0.0
        ),
        "color": (0.65, 0.65, 0.63),
        "roughness": 0.85,
    },
    # 6. Ostacoli discreti — blocchi max 25 cm su base asfalto
    {
        "name": "t6_obstacles",
        "sub_cfg": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=1.0,
            obstacle_height_mode="fixed",
            obstacle_height_range=(0.02, 0.08),
            obstacle_width_range=(0.15, 0.40),
            num_obstacles=25,
            platform_width=0.2,
            border_width=0.25,
        ),
        "material": RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=0.8, restitution=0.0
        ),
        "color": (0.15, 0.15, 0.15),
        "roughness": 0.9,
    },
    # 7. Scale edificio — gradini 7 cm, marmo liscio beige chiaro
    {
        "name": "t7_stairs",
        "sub_cfg": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.07, 0.07),
            step_width=0.30,
            platform_width=1.0,
            border_width=0.25,
        ),
        "material": RigidBodyMaterialCfg(
            static_friction=0.9, dynamic_friction=0.7, restitution=0.0
        ),
        "color": (0.85, 0.80, 0.72),
        "roughness": 0.15,
    },
]


# =============================================================================
# 2. FUNZIONE PER APPLICARE MATERIALE VISIVO PBR
# =============================================================================
def apply_visual_material(stage, prim_path, name, color, roughness):
    mat_path = f"/World/Materials/{name}_mat"
    material = stage.GetPrimAtPath(mat_path)
    if not material.IsValid():
        material = UsdShade.Material.Define(stage, Sdf.Path(mat_path))
        shader = UsdShade.Shader.Define(stage, Sdf.Path(f"{mat_path}/Shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(color[0], color[1], color[2])
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    else:
        material = UsdShade.Material(material)

    root_prim = stage.GetPrimAtPath(prim_path)
    if root_prim.IsValid():
        for prim in stage.Traverse():
            if not str(prim.GetPath()).startswith(prim_path):
                continue
            if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Xform):
                binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
                binding_api.Bind(material)


# =============================================================================
# 3. GENERAZIONE ED ESPORTAZIONE SCACCHIERA
# =============================================================================
def main():
    print("=" * 60)
    print("  GENERAZIONE SCACCHIERA RANDOMICA (42m x 21m)")
    print("=" * 60)

    sim_cfg = sim_utils.SimulationCfg(device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    stage = omni.usd.get_context().get_stage()

    # Creiamo un array di 49 elementi (7 per ogni terreno)
    terrain_indices = list(range(len(TERRAINS))) * ROWS
    random.shuffle(terrain_indices)

    cell_count = 0
    for r in range(ROWS):
        for c in range(COLS):
            terrain_idx = terrain_indices[cell_count]
            terrain = TERRAINS[terrain_idx]
            
            name = terrain["name"]
            prim_path = f"/World/Grid/Cell_{r}_{c}_{name}"
            
            x_offset = c * CELL_W
            y_offset = r * CELL_H

            print(f"Generazione Cella {r},{c} (X:{x_offset}m, Y:{y_offset}m) -> {name}")

            single_cfg = TerrainGeneratorCfg(
                size=(CELL_W, CELL_H),
                border_width=0.0,
                num_rows=1,
                num_cols=1,
                horizontal_scale=0.02,
                vertical_scale=0.001,
                use_cache=False,
                curriculum=False,
                sub_terrains={name: terrain["sub_cfg"]},
            )

            importer_cfg = TerrainImporterCfg(
                prim_path=prim_path,
                terrain_type="generator",
                terrain_generator=single_cfg,
                collision_group=-1,
                physics_material=terrain["material"],
                num_envs=1,
            )

            TerrainImporter(importer_cfg)

            # Sposta la mesh della singola cella modificandone i vertici
            for prim in stage.Traverse():
                if not str(prim.GetPath()).startswith(prim_path):
                    continue
                if prim.IsA(UsdGeom.Mesh):
                    mesh = UsdGeom.Mesh(prim)
                    points = mesh.GetPointsAttr().Get()
                    if points:
                        new_points = Vt.Vec3fArray(len(points))
                        for j, p in enumerate(points):
                            new_points[j] = Gf.Vec3f(p[0] + x_offset, p[1] + y_offset, p[2])
                        mesh.GetPointsAttr().Set(new_points)

            # Applica materiale PBR
            apply_visual_material(
                stage, prim_path,
                name, terrain["color"], terrain["roughness"]
            )
            
            cell_count += 1

    sim.reset()

    usd_path = "/home/students/work/barbara/isaac-projects/projects/my-projects/terrain_generator/random_board.usd"
    print(f"\nEsportazione in: {usd_path}")

    world_prim = stage.GetPrimAtPath("/World")
    if world_prim.IsValid():
        stage.SetDefaultPrim(world_prim)

    omni.usd.get_context().save_as_stage(usd_path)
    print("Esportazione completata con successo!")

if __name__ == "__main__":
    main()
    simulation_app.close()
