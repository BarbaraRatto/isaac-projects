# --- 0. AVVIO MOTORE (SEMPRE PER PRIMO!) ---
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# --- 1. IMPORTAZIONI NECESSARIE (dopo l'avvio dell'app!) ---
import math
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg

# --- 2. DEFINIZIONE DEI MATERIALI FISICI ---

# Attrito normale (Asfalto/Roccia)
material_normal = RigidBodyMaterialCfg(
    static_friction=1.0,
    dynamic_friction=0.8,
    restitution=0.0,
)

# Attrito basso (Bagnato/Scivoloso/Ghiaia fine)
material_slippery = RigidBodyMaterialCfg(
    static_friction=0.3,
    dynamic_friction=0.2,
    restitution=0.0,
)

# --- 3. CONFIGURAZIONE DELLA SCACCHIERA (GLI 8 TERRENI) ---

# Convertiamo 15 gradi in radianti per le rampe
max_slope_rad = math.radians(15.0)

terrain_cfg = TerrainGeneratorCfg(
    size=(6.0, 1.5),       # Lunghezza 6m (X), Larghezza corsia 1.5m (Y)
    border_width=3.0,      # Bordo di sicurezza ridotto per adattarsi alle corsie
    num_rows=8,            # <--- MODIFICATO: Ora abbiamo 8 tipi di terreno
    num_cols=8,            # 8 livelli di difficolta' (dal piatto al massimo)
    use_cache=False,
    curriculum=True,       # Terreni in ordine
    
    sub_terrains={

        # 1. Asfalto (Piatto, attrito normale)
        "1_flat_asphalt": terrain_gen.MeshPlaneTerrainCfg(
            proportion=1.0 / 8.0,
        ),

        # 2. Piatto Scivoloso (Bagnato)
        "2_slippery_flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=1.0 / 8.0,
        ),

        # 3. NUOVO: Rampa Liscia (Fino a 15 gradi nominali)
        "3_smooth_ramp": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0 / 8.0,
            slope_range=(0.0, math.tan(max_slope_rad)), # Pendenza da 0 a 15 gradi
            platform_width=1.0,
            border_width=0.25,
        ),

        # 4. Scale (Gradini fino a 4 cm, prima erano 5 cm)
        "4_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0 / 8.0,
            step_height_range=(0.0, 0.04), # <--- MODIFICATO: max 4 cm
            step_width=0.3,
            platform_width=1.0, 
        ),

        # 5. Ghiaia Fine (Quasi liscia)
        "5_fine_gravel": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0 / 8.0,
            noise_range=(0.0, 0.005),      # max 0.5 cm
            noise_step=0.01,
            border_width=0.25,
        ),

        # 6. Sassi Grandi (Piccole irregolarità)
        "6_large_stones": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0 / 8.0,
            noise_range=(0.0, 0.01),       # max 1 cm
            noise_step=0.02,
            border_width=0.25,
        ),

        # 7. Ostacoli Discreti (Scatole basse)
        "7_discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=1.0 / 8.0,
            obstacle_height_mode="fixed",
            obstacle_height_range=(0.02, 0.08), # max 8 cm
            obstacle_width_range=(0.2, 0.4),    
            num_obstacles=30,                   
            platform_width=1.0, 
        ),

        # 8. Onde / Dune (Collinette molto dolci)
        "8_wave_hills": terrain_gen.HfWaveTerrainCfg(
            proportion=1.0 / 8.0,
            amplitude_range=(0.0, 0.1),         # max 10 cm
            num_waves=4,
            border_width=0.25,
        ),
    },
)

# --- 4. GENERAZIONE ED ESPORTAZIONE ---
def main():
    print("Inizio generazione del terreno procedurale...")

    import isaaclab.sim as sim_utils
    from isaaclab.terrains import TerrainImporter, TerrainImporterCfg

    sim_cfg = sim_utils.SimulationCfg(device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)

    importer_cfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_cfg,
        collision_group=-1,
        physics_material=material_normal,
        num_envs=1,
    )

    terrain_importer = TerrainImporter(importer_cfg)
    sim.reset()

    usd_path = "/home/students/work/barbara/isaac-projects/projects/my-projects/terrain_generator/spot_terrains_test.usd"

    print(f"Esportazione del terreno in: {usd_path}")

    import omni.usd
    
    # --- RIGHE PER IL DEFAULT PRIM ---   
    stage = omni.usd.get_context().get_stage()
    world_prim = stage.GetPrimAtPath("/World")
    if world_prim.IsValid():
        stage.SetDefaultPrim(world_prim)
    else:
        print("[ATTENZIONE] /World non trovato: default prim non impostato.")
    # ---------------------------------------
    
    omni.usd.get_context().save_as_stage(usd_path)
    print("Esportazione completata! Puoi ora aprire il file in Isaac Sim.")


if __name__ == "__main__":
    main()
    simulation_app.close()
