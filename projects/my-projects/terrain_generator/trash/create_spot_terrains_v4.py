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

# --- 3. CONFIGURAZIONE DELLA SCACCHIERA (I 7 TERRENI) ---

# Convertiamo 15 gradi in radianti per la rampa (circa 0.26 rad)
max_slope_rad = math.radians(15.0)

terrain_cfg = TerrainGeneratorCfg(
    size=(4.0, 4.0),       # Ogni quadrato e' 4x4 metri
    border_width=5.0,      # Bordo di sicurezza a 5 metri
    num_rows=7,            # 7 tipi di terreno (le nostre corsie)
    num_cols=8,            # 8 livelli di difficolta' (dal piatto al massimo)
    use_cache=False,
    curriculum=True,       # Terreni in ordine
    
    sub_terrains={

        # 1. Asfalto (Piatto, attrito normale)
        "1_flat_asphalt": terrain_gen.MeshPlaneTerrainCfg(
            proportion=1.0 / 7.0,
        ),

        # 2. Piatto Scivoloso (Bagnato)
        "2_slippery_flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=1.0 / 7.0,
        ),

        # 3. Rampa (Fino a 15 gradi)
        "3_ramp_15_deg": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0 / 7.0,
            step_height_range=(0.0, 0.05),
            step_width=0.3,
            platform_width=1.0, 
        ),

        # 4. Ghiaia Fine (Rumore fino a 2 cm)
        "4_fine_gravel": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0 / 7.0,
            noise_range=(0.0, 0.02), # <--- MODIFICATO: max 2 cm
            noise_step=0.01,
            border_width=0.25,
        ),

        # 5. Sassi Grandi (Rumore fino a 3 cm)
        "5_large_stones": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0 / 7.0,
            noise_range=(0.0, 0.03), # <--- MODIFICATO: max 3 cm
            noise_step=0.02,
            border_width=0.25,
        ),

        # 6. Ostacoli Discreti (Scatole rigide, massimo 30 cm)
        "6_discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=1.0 / 7.0,
            obstacle_height_mode="fixed",
            obstacle_height_range=(0.05, 0.30),
            obstacle_width_range=(0.3, 0.5),
            num_obstacles=40,
            platform_width=1.0, 
        ),

        # 7. Onde / Dune (Collinette morbide)
        "7_wave_hills": terrain_gen.HfWaveTerrainCfg(
            proportion=1.0 / 7.0,
            amplitude_range=(0.0, 0.4),
            num_waves=4,
            border_width=0.25,
        ),
    },
)

# --- 4. GENERAZIONE ED ESPORTAZIONE ---
def main():
    print("Inizio generazione del terreno procedurale...")

    # Import qui dentro: servono sia SimulationContext che TerrainImporter
    import isaaclab.sim as sim_utils
    from isaaclab.terrains import TerrainImporter, TerrainImporterCfg

    # TerrainImporter richiede un SimulationContext gia' attivo: lo creiamo esplicitamente
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

    # Questo genera davvero la mesh e la inserisce nello stage USD corrente
    terrain_importer = TerrainImporter(importer_cfg)

    # Serve almeno un reset per finalizzare fisica/handle prima di salvare
    sim.reset()

    usd_path = "/home/students/work/barbara/isaac-projects/projects/my-projects/terrain_generator/spot_terrains_test.usd"

    print(f"Esportazione del terreno in: {usd_path}")

    import omni.usd
    
    # --- RIGHE PER IL DEFAULT PRIM ---   
    
    stage = omni.usd.get_context().get_stage()
    # Imposta come default prim il prim radice del terreno (/World),
    # altrimenti il file non potra' essere referenziato/trascinato in altre scene
    world_prim = stage.GetPrimAtPath("/World")
    if world_prim.IsValid():
        stage.SetDefaultPrim(world_prim)
    else:
        print("[ATTENZIONE] /World non trovato: default prim non impostato, verifica il path radice.")
    # ---------------------------------------
    
    omni.usd.get_context().save_as_stage(usd_path)
    print("Esportazione completata! Puoi ora aprire il file in Isaac Sim.")


if __name__ == "__main__":
    main()
    simulation_app.close()
