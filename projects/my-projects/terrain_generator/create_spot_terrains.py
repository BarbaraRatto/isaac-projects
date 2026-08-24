# --- 0. AVVIO MOTORE (SEMPRE PER PRIMO!) ---
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import math
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg

# --- 2. DEFINIZIONE DEI MATERIALI FISICI ---
material_normal = RigidBodyMaterialCfg(
    static_friction=1.5,	# prima 1.0
    dynamic_friction=1.2,	# prima 0.8
    restitution=0.0,
)

material_slippery = RigidBodyMaterialCfg(
    static_friction=0.3,
    dynamic_friction=0.2,
    restitution=0.0,
)

max_slope_rad = math.radians(7.5) # Dimezzato (prima 15.0)

# --- 3. CONFIGURAZIONE DELLA SCACCHIERA ---
terrain_cfg = TerrainGeneratorCfg(
    size=(6.0, 3.0),       
    border_width=3.0,      
    num_rows=8,            
    num_cols=8,            
    
    # RISOLUZIONE ALTA PER DUNE LISCE E SCATOLE DRITTE
    horizontal_scale=0.02, # Punti della griglia ogni 2 cm
    vertical_scale=0.001,  # Altezza calcolata al millimetro
    
    use_cache=False,
    curriculum=False,       
    
    sub_terrains={
        "1_flat_asphalt": terrain_gen.MeshPlaneTerrainCfg(
            proportion=1.0 / 8.0,
        ),
        "2_slippery_flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=1.0 / 8.0,
        ),
        
        # 3. Rampa Liscia (Tronco di piramide)
        "3_smooth_ramp": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0 / 8.0,
            slope_range=(0.0, math.tan(max_slope_rad)), 
            platform_width=1.0, 
            border_width=0.25, # Uniformato a 0.25
        ),
        
        # 4. Scale
        "4_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0 / 8.0,
            step_height_range=(0.0, 0.02), 
            step_width=0.15,               
            platform_width=1.0, 	   
            border_width=0.25, # Aggiunto per uniformità
        ),
        
        "5_fine_gravel": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0 / 8.0,
            noise_range=(0.0, 0.0025), 
            noise_step=0.01,
            border_width=0.25, # Rimesso al valore originale
        ),
        
        "6_large_stones": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0 / 8.0,
            noise_range=(0.0, 0.005), 
            noise_step=0.02,
            border_width=0.25, # Rimesso al valore originale
        ),
        
        "7_discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=1.0 / 8.0,
            obstacle_height_mode="fixed",
            obstacle_height_range=(0.01, 0.025), 
            obstacle_width_range=(0.1, 0.4),    
            num_obstacles=40,                   
            platform_width=0.2, 
            border_width=0.25, # Aggiunto per uniformità
        ),
        
        "8_wave_hills": terrain_gen.HfWaveTerrainCfg(
            proportion=1.0 / 8.0,
            amplitude_range=(0.0, 0.05), 
            num_waves=4,
            border_width=0.25, # Rimesso al valore originale
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
        # Aggiungo una texture visiva realistica per far lavorare bene la telecamera ZED
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Materials/Base/Masonry/Concrete_Rough.mdl",
            project_uvw=True,      # Fondamentale per mappare la texture sul terreno procedurale
            texture_scale=(0.5, 0.5), # Scala della texture
        ),
    )

    terrain_importer = TerrainImporter(importer_cfg)
    sim.reset()

    usd_path = "/home/students/work/barbara/isaac-projects/projects/my-projects/terrain_generator/spot_terrains.usd"
    print(f"Esportazione del terreno in: {usd_path}")

    import omni.usd
    
    stage = omni.usd.get_context().get_stage()
    world_prim = stage.GetPrimAtPath("/World")
    if world_prim.IsValid():
        stage.SetDefaultPrim(world_prim)
    
    omni.usd.get_context().save_as_stage(usd_path)
    print("Esportazione completata! Puoi ora aprire il file in Isaac Sim.")

if __name__ == "__main__":
    main()
    simulation_app.close()
