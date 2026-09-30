# =============================================================================
# create_real_terrains.py
# Genera una scacchiera 7x7 (42m x 21m) usando 7 tipi di terreni.
# Ogni tipo di terreno compare esattamente 7 volte.
#
# Lancia con:
#   /home/isaac/isaaclab/2.3.0/isaaclab.sh -p /home/students/work/barbara/isaac-projects/projects/my-projects/terrain_generator/create_real_terrains.py
# =============================================================================

# --- 0. AVVIO MOTORE (SEMPRE PER PRIMO!) ---
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import math
import random
from pathlib import Path
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporter, TerrainImporterCfg
from isaaclab.sim.spawners.materials import MdlFileCfg, RigidBodyMaterialCfg
import isaaclab.sim as sim_utils
import omni.usd
from pxr import Usd, UsdGeom, Gf, UsdShade, Sdf, Vt

# =============================================================================
# 1. DEFINIZIONE DEI 7 TERRENI
# =============================================================================
CELL_W = 6.0   # larghezza cella (X)
CELL_H = 3.0   # altezza cella (Y)
ROWS = 7
COLS = 7

TEX_DIR = Path(__file__).resolve().parent / "textures"
TEXTURE_TILING = 2.0

TERRAINS = [
    # 1. Asfalto pit lane — piano, attrito standard
    {
        "name": "t1_asphalt",
        "sub_cfg": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
        "material": RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=0.8, restitution=0.0
        ),
        "texture_color": f"{TEX_DIR}/asphalt_pit_lane_diff_1k.png",
        "texture_normal": f"{TEX_DIR}/asphalt_pit_lane_nor_dx_1k.png",
        "roughness": 0.9,
    },
    # 2. Ghiaccio — piano, poco attrito
    {
        "name": "t2_slippery",
        "sub_cfg": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
        "material": RigidBodyMaterialCfg(
            static_friction=0.1, dynamic_friction=0.08, restitution=0.0
        ),
        "texture_color": f"{TEX_DIR}/ice_ground_0031_color_1k.jpg",
        "texture_normal": f"{TEX_DIR}/ice_ground_0031_normal_directx_1k.png",
        "roughness": 0.2,
    },
    # 3. Rampa 7° — stesso asfalto, pendenza visibile con la luce
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
        "texture_color": f"{TEX_DIR}/asphalt_pit_lane_diff_1k.png",
        "texture_normal": f"{TEX_DIR}/asphalt_pit_lane_nor_dx_1k.png",
        "roughness": 0.7,
    },
    # 4. Ghiaia fine — superficie irregolare
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
        "texture_color": f"{TEX_DIR}/gravel_floor_02_diff_1k.png",
        "texture_normal": f"{TEX_DIR}/gravel_floor_02_nor_dx_1k.png",
        "roughness": 0.95,
    },
    # 5. Rocce grigie — irregolarita fino a 5 cm
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
        "texture_color": f"{TEX_DIR}/gray_rocks_diff_1k.png",
        "texture_normal": f"{TEX_DIR}/gray_rocks_nor_dx_1k.png",
        "roughness": 0.85,
    },
    # 6. Ostacoli discreti — blocchi da 2 a 8 cm su base asfalto
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
        "texture_color": f"{TEX_DIR}/asphalt_pit_lane_diff_1k.png",
        "texture_normal": f"{TEX_DIR}/asphalt_pit_lane_nor_dx_1k.png",
        "roughness": 0.9,
    },
    # 7. Scale — gradini da 7 cm con marmo
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
        "texture_color": f"{TEX_DIR}/marble_rock_02_diff_1k.png",
        "texture_normal": f"{TEX_DIR}/marble_rock_02_nor_dx_1k.png",
        "roughness": 0.15,
    },
]


# =============================================================================
# 2. FUNZIONE PER APPLICARE MATERIALE VISIVO PBR
# =============================================================================
def apply_visual_material(stage, prim_path, name, terrain_dict):
    mat_path = f"/World/Materials/{name}_mat"
    if not stage.GetPrimAtPath(mat_path).IsValid():
        # Le mesh di Isaac Lab non hanno primvars:st. OmniPBR genera UV
        # proiettate e applica lo stesso tiling configurabile nella UI.
        cfg = MdlFileCfg(
            mdl_path="OmniPBR.mdl",
            project_uvw=True,
            texture_scale=(TEXTURE_TILING, TEXTURE_TILING),
        )
        cfg.func(mat_path, cfg)
        shader = UsdShade.Shader.Get(stage, f"{mat_path}/Shader")

        if "texture_color" in terrain_dict:
            for key in ("texture_color", "texture_normal"):
                texture_path = terrain_dict.get(key)
                if texture_path and not Path(texture_path).is_file():
                    raise FileNotFoundError(f"Texture mancante per {name}: {texture_path}")
            shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
                Sdf.AssetPath(terrain_dict["texture_color"])
            )
            if terrain_dict.get("texture_normal"):
                shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(
                    Sdf.AssetPath(terrain_dict["texture_normal"])
                )
                # Tutte le normal map attuali sono DirectX (Y-).
                shader.CreateInput("flip_tangent_v", Sdf.ValueTypeNames.Bool).Set(True)
        else:
            color = terrain_dict["color"]
            shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))

        shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(
            terrain_dict.get("roughness", 0.5)
        )

    material = UsdShade.Material.Get(stage, mat_path)
    root_prim = stage.GetPrimAtPath(prim_path)
    if root_prim.IsValid():
        for prim in Usd.PrimRange(root_prim):
            if prim.IsA(UsdGeom.Mesh):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


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
            apply_visual_material(stage, prim_path, name, terrain)
            
            cell_count += 1

    sim.reset()

    usd_path = "/home/students/work/barbara/isaac-projects/projects/my-projects/terrain_generator/real_terrains.usd"
    print(f"\nEsportazione in: {usd_path}")

    world_prim = stage.GetPrimAtPath("/World")
    if world_prim.IsValid():
        stage.SetDefaultPrim(world_prim)

    if not stage.Export(usd_path):
        raise RuntimeError(f"Esportazione USD fallita: {usd_path}")
    print("Esportazione completata con successo!")

if __name__ == "__main__":
    main()
    simulation_app.close()
