#!/usr/bin/env python3
"""
play_rl_isaaclab.py

Esegue in modalità INFERENZA il modello addestrato in Isaac Lab.
Lancia il programma senza l'opzione --headless per vedere i robot muoversi nello spazio 3D.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play PPO Spot (Isaac Lab)")
parser.add_argument("--num_envs", type=int, default=2, help="Numero di robot da visualizzare (default: 2)")
parser.add_argument("--model", type=str, default="models/final_model_isaaclab", help="Percorso al modello (senza .zip)")
parser.add_argument("--config", type=str, default="config/rl_params.yaml", help="Percorso alla config")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import yaml
import torch

import sys
_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from stable_baselines3 import PPO

def make_env(num_envs: int, spot_usd: str):
    from spot_rl_controller.spot_isaaclab_env import SpotEnergyEnv, SpotEnergyEnvCfg
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
    from stable_baselines3.common.vec_env import VecNormalize
    
    env_cfg = SpotEnergyEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.robot_usd_path = spot_usd
    env = SpotEnergyEnv(env_cfg)
    
    # Forza l'avvio della timeline di fisica
    env.sim.play()
    
    vec_env = Sb3VecEnvWrapper(env)
    
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
    ros2_ws = _PROJECT_ROOT / "ros2_ws"
    
    # Risolvi percorsi relativi al workspace
    model_path = args.model if args.model.startswith("/") else str(ros2_ws / args.model)
    vnorm_path = Path(model_path + "_vecnorm.pkl")
    
    if vnorm_path.exists():
        vec_env = VecNormalize.load(str(vnorm_path), vec_env)
        vec_env.training = False
        vec_env.reward_norm = False
    else:
        # Se non c'è, usiamo il default per evitare crash, anche se le obs non saranno normalizzate identiche
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=False)
        
    return vec_env

def main():
    config_path = Path(__file__).resolve().parent.parent / "config" / "rl_params.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    spot_usd = config.get("paths", {}).get("spot_usd", "")
    
    print(f"[play_isaaclab] Creazione env con {args.num_envs} robot in parallelo...")
    vec_env = make_env(args.num_envs, spot_usd)
    
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
    ros2_ws = _PROJECT_ROOT / "ros2_ws"
    model_path = args.model if args.model.startswith("/") else str(ros2_ws / args.model)
    
    print(f"[play_isaaclab] Caricamento modello: {model_path}")
    model = PPO.load(model_path, env=vec_env, device="cuda")
    
    print("[play_isaaclab] Inizio simulazione in tempo reale...")
    obs = vec_env.reset()
    
    step_count = 0
    while simulation_app.is_running():
        actions, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(actions)
        
        step_count += 1
        if step_count % 50 == 0:
            print(f"[play_isaaclab] Step {step_count} | Azioni Modello (Robot 0): {actions[0]}")
        
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n=== ERRORE FATALE ===\n{traceback.format_exc()}")
    finally:
        print("[play_isaaclab] Chiudo la simulazione...")
        simulation_app.close()
