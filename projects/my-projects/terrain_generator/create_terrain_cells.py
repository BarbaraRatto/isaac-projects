# =============================================================================
# create_terrain_cells.py
# Genera 7 celle di terreno separate (6x3 m ciascuna) disposte fianco a fianco
# lungo Y, ognuna con il proprio materiale fisico (attrito) e visivo (PBR).
#
# Lancia con:
#   /home/isaac/isaaclab/2.3.0/isaaclab.sh -p /home/students/work/barbara/isaac-projects/projects/my-projects/terrain_generator/create_terrain_cells.py
#
# Output: terrain_cells.usd nella stessa cartella
# =============================================================================

# --- 0. AVVIO MOTORE (SEMPRE PER PRIMO!) ---
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import math
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporter, TerrainImporterCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
import isaaclab.sim as sim_utils
import omni.usd
from pxr import UsdGeom, Gf, UsdShade, Sdf, Vt

# =============================================================================
# 1. DEFINIZIONE DEI 7 TERRENI
# =============================================================================
# Ogni terreno è un dizionario con:
#   - name:      nome identificativo (usato come prim path)
#   - sub_cfg:   configurazione Isaac Lab per il tipo di terreno
#   - material:  materiale fisico (attrito)
#   - color:     colore diffuso RGB (0-1)
#   - roughness: rugosità PBR (0 = lucido, 1 = ruvido)

CELL_W = 6.0   # larghezza cella (X)
CELL_H = 3.0   # altezza cella (Y)

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
    # 2. Scivoloso (erba bagnata) — piano, poco attrito, verde lucido
    {
        "name": "t2_slippery",
        "sub_cfg": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
        "material": RigidBodyMaterialCfg(
            static_friction=0.15, dynamic_friction=0.08, restitution=0.0
        ),
        "color": (0.30, 0.60, 0.20),
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
            static_friction=0.5, dynamic_friction=0.3, restitution=0.0
        ),
        "color": (0.76, 0.70, 0.50),
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
            static_friction=0.8, dynamic_friction=0.6, restitution=0.0
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
    # 7. Scale edificio — gradini 17 cm, marmo liscio beige chiaro
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
    """Crea un materiale UsdPreviewSurface e lo applica a TUTTE le mesh
    trovate ricorsivamente sotto il prim_path."""
    mat_path = f"/World/Materials/{name}_mat"

    # Crea il materiale
    material = UsdShade.Material.Define(stage, Sdf.Path(mat_path))

    # Crea lo shader PBR
    shader = UsdShade.Shader.Define(stage, Sdf.Path(f"{mat_path}/Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(color[0], color[1], color[2])
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    # Collega lo shader al materiale
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    # Trova TUTTE le mesh sotto il prim e applica il materiale direttamente
    root_prim = stage.GetPrimAtPath(prim_path)
    bound_count = 0
    if root_prim.IsValid():
        # Iteriamo ricorsivamente su TUTTI i discendenti
        for prim in stage.Traverse():
            # Controlliamo che sia sotto il nostro prim_path
            if not str(prim.GetPath()).startswith(prim_path):
                continue
            # Applichiamo il materiale a Mesh e a qualsiasi prim con geometria
            if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Xform):
                binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
                binding_api.Bind(material)
                bound_count += 1

    print(f"  [Visual] Materiale PBR '{name}' applicato a {bound_count} prim: "
          f"colore={color}, roughness={roughness}")


# =============================================================================
# 3. GENERAZIONE ED ESPORTAZIONE
# =============================================================================
def main():
    print("=" * 60)
    print("  GENERAZIONE TERRAIN CELLS (7 terreni separati)")
    print("=" * 60)

    # Setup simulazione
    sim_cfg = sim_utils.SimulationCfg(device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)

    stage = omni.usd.get_context().get_stage()

    for i, terrain in enumerate(TERRAINS):
        name = terrain["name"]
        prim_path = f"/World/Terrain_{name}"
        y_offset = i * CELL_H  # Ogni cella è spostata di 3m lungo Y

        print(f"\n--- Terreno {i+1}/7: {name} (Y offset: {y_offset}m) ---")

        # Configurazione per una singola cella
        single_cfg = TerrainGeneratorCfg(
            size=(CELL_W, CELL_H),
            border_width=0.0,       # Nessun bordo — celle adiacenti
            num_rows=1,
            num_cols=1,
            horizontal_scale=0.02,  # 2 cm di risoluzione
            vertical_scale=0.001,   # 1 mm di precisione verticale
            use_cache=False,
            curriculum=False,
            sub_terrains={name: terrain["sub_cfg"]},
        )

        # Importa il terreno con il suo materiale fisico
        importer_cfg = TerrainImporterCfg(
            prim_path=prim_path,
            terrain_type="generator",
            terrain_generator=single_cfg,
            collision_group=-1,
            physics_material=terrain["material"],
            num_envs=1,
        )

        print(f"  Generazione mesh...")
        TerrainImporter(importer_cfg)

        # Sposta il terreno lungo Y modificando DIRETTAMENTE i vertici della mesh
        # (il translate su Xform non funziona perché i vertici sono in coordinate assolute)
        moved = 0
        for prim in stage.Traverse():
            if not str(prim.GetPath()).startswith(prim_path):
                continue
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                points = mesh.GetPointsAttr().Get()
                if points:
                    new_points = Vt.Vec3fArray(len(points))
                    for j, p in enumerate(points):
                        new_points[j] = Gf.Vec3f(p[0], p[1] + y_offset, p[2])
                    mesh.GetPointsAttr().Set(new_points)
                    moved += len(points)
        print(f"  Spostati {moved} vertici di Y+{y_offset}m")

        # Applica materiale visivo PBR
        apply_visual_material(
            stage, prim_path,
            name, terrain["color"], terrain["roughness"]
        )

    # Reset della simulazione per consolidare
    sim.reset()

    # Esportazione USD
    usd_path = "/home/students/work/barbara/isaac-projects/projects/my-projects/terrain_generator/terrain_cells.usd"
    print(f"\n{'=' * 60}")
    print(f"  Esportazione in: {usd_path}")

    world_prim = stage.GetPrimAtPath("/World")
    if world_prim.IsValid():
        stage.SetDefaultPrim(world_prim)

    omni.usd.get_context().save_as_stage(usd_path)
    print("  Esportazione completata!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
    simulation_app.close()
