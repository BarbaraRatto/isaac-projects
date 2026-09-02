#!/usr/bin/env python3
"""
train_rl_isaaclab.py

Script di training PPO per Spot in Isaac Lab.

Questo script deve essere eseguito tramite il launcher di Isaac Lab,
non direttamente con python3.

Uso:
  # Dalla cartella my-projects:
  ./isaaclab.sh -p ros2_ws/src/spot_rl_controller/spot_rl_controller/train_rl_isaaclab.py \
      --num_envs 1024 --headless

  # Con meno robot (per test su macchine con meno VRAM):
  ./isaaclab.sh -p ... --num_envs 64 --headless

  # Smoke test con rendering:
  ./isaaclab.sh -p ... --num_envs 4

  # Riprendere da checkpoint:
  ./isaaclab.sh -p ... --resume models/best_model.zip

Dove si trova isaaclab.sh:
  Nella cartella di installazione di Isaac Lab, tipicamente:
    ~/isaac-lab/isaaclab.sh
  oppure impostare la variabile ISAACLAB_PATH.

Output:
  - models/best_model.zip  (SB3, compatibile con rl_controller_node.py)
  - models/final_model.zip
  - logs/spot_rl_*/        (TensorBoard)
  - models/vecnorm_*.pkl   (normalizzazione reward)
"""

from __future__ import annotations

# ─── ATTENZIONE: AppLauncher DEVE essere il primo import ─────────────────────
# Isaac Lab richiede che il motore di simulazione parta prima di tutto il resto.
import argparse
import sys
from pathlib import Path

# Parser argomenti (richiesto da AppLauncher)
parser = argparse.ArgumentParser(description="Training PPO Spot energy-aware (Isaac Lab)")
parser.add_argument("--num_envs",    type=int,   default=1024)
#parser.add_argument("--headless",    action="store_true")
parser.add_argument("--resume",      type=str,   default=None,
                    help="Percorso al checkpoint SB3 da cui riprendere (senza .zip)")
parser.add_argument("--total_steps", type=int,   default=5_000_000)
parser.add_argument("--config",      type=str,   default=None)
parser.add_argument("--dry_run",     action="store_true",
                    help="Esegui solo 1000 step (smoke test)")

# Isaac Lab AppLauncher aggiunge i suoi argomenti prima di tutto
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = args.headless  # già gestito
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
# ─────────────────────────────────────────────────────────────────────────────

import os
import time
import yaml
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import VecNormalize

# ─── Isaac Lab wrappers SB3 ───────────────────────────────────────────────────
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
# Nota: il nome esatto può variare tra versioni di Isaac Lab.

# ─── Nostro environment e rete ───────────────────────────────────────────────
# Aggiungi il pacchetto al path se necessario
_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from spot_rl_controller.spot_isaaclab_env import SpotEnergyEnv, SpotEnergyEnvCfg


# ===========================================================================
# Percorsi di default
# ===========================================================================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
# → my-projects/
SPOT_USD_PATH = str(_PROJECT_ROOT / "Graph.usd")
MODEL_DIR     = str(_PROJECT_ROOT / "ros2_ws" / "models")
LOG_DIR       = str(_PROJECT_ROOT / "ros2_ws" / "logs")


# ===========================================================================
# Caricamento configurazione PPO
# ===========================================================================
def load_config(config_path: str | None) -> dict:
    """Carica rl_params.yaml se esiste, altrimenti usa default."""
    if config_path is not None and Path(config_path).exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}

    # Cerca automaticamente
    candidates = [
        _PROJECT_ROOT / "ros2_ws" / "src" / "spot_rl_controller" / "config" / "rl_params.yaml",
        Path("config/rl_params.yaml"),
    ]
    for c in candidates:
        if c.exists():
            print(f"[train_isaaclab] Config: {c}")
            with open(c) as f:
                return yaml.safe_load(f) or {}

    print("[train_isaaclab] config/rl_params.yaml non trovato — uso default PPO.")
    return {}


# ===========================================================================
# Creazione environment
# ===========================================================================
def make_env(num_envs: int, spot_usd: str) -> Sb3VecEnvWrapper:
    """Crea l'Isaac Lab env e lo wrappa per SB3."""
    cfg = SpotEnergyEnvCfg()
    cfg.scene.num_envs   = num_envs
    cfg.robot_usd_path   = spot_usd
    cfg.sim.device       = "cuda:0"

    env     = SpotEnergyEnv(cfg)
    vec_env = Sb3VecEnvWrapper(env)   # VecEnv SB3-compatibile
    vec_env = VecNormalize(
        vec_env,
        norm_obs=False,      # la normalizzazione obs è già nell'env
        norm_reward=True,
        clip_reward=10.0,
    )
    return vec_env


# ===========================================================================
# Training
# ===========================================================================
def train(
    num_envs:    int,
    total_steps: int,
    resume_from: str | None,
    config:      dict,
    dry_run:     bool,
    spot_usd:    str,
) -> None:
    ppo_cfg = config.get("ppo", {})
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    total_steps = 1000 if dry_run else total_steps

    # ── Creazione env ──────────────────────────────────────────────────────
    print(f"[train_isaaclab] Creazione env con {num_envs} robot in parallelo...")
    vec_env = make_env(num_envs, spot_usd)

    # ── Modello PPO ────────────────────────────────────────────────────────
    if resume_from and Path(resume_from + ".zip").exists():
        print(f"[train_isaaclab] Riprendendo da: {resume_from}")
        model = PPO.load(resume_from, env=vec_env)
        vnorm = Path(resume_from + "_vecnorm.pkl")
        if vnorm.exists():
            vec_env = VecNormalize.load(str(vnorm), vec_env)
    else:
        print("[train_isaaclab] Nuovo modello PPO...")
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
            n_steps=int(ppo_cfg.get("n_steps",   2048)),
            batch_size=int(ppo_cfg.get("batch_size", 256)),
            n_epochs=int(ppo_cfg.get("n_epochs",   10)),
            gamma=float(ppo_cfg.get("gamma",       0.99)),
            gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
            clip_range=float(ppo_cfg.get("clip_range", 0.2)),
            ent_coef=float(ppo_cfg.get("ent_coef",  0.01)),
            vf_coef=float(ppo_cfg.get("vf_coef",   0.5)),
            max_grad_norm=float(ppo_cfg.get("max_grad_norm", 0.5)),
            # Architettura MLP: due hidden layer da 512 e 256
            # (input: 2510-d flat obs → hidden → azioni 3-d)
            policy_kwargs={"net_arch": [512, 256]},
            verbose=1,
            tensorboard_log=LOG_DIR,
            device="cuda",
        )

    print(
        f"\n[train_isaaclab] "
        f"{'DRY RUN' if dry_run else 'TRAINING'} — "
        f"{total_steps:,} step, {num_envs} robot\n"
    )

    # ── Callback: salva checkpoint ogni 100k step ──────────────────────────
    callbacks = []
    if not dry_run:
        callbacks.append(CheckpointCallback(
            save_freq=100_000 // num_envs,   # ogni N update (in base a n_envs)
            save_path=MODEL_DIR,
            name_prefix="spot_rl_isaaclab",
            save_vecnormalize=True,
        ))

    # ── Training ───────────────────────────────────────────────────────────
    t0 = time.time()
    model.learn(
        total_timesteps=total_steps,
        callback=CallbackList(callbacks),
        log_interval=10,
        reset_num_timesteps=(resume_from is None),
        progress_bar=True,
    )
    elapsed = time.time() - t0
    print(f"\n[train_isaaclab] Completato in {elapsed/60:.1f} min")

    # ── Salvataggio finale ─────────────────────────────────────────────────
    if not dry_run:
        out_path = os.path.join(MODEL_DIR, "final_model_isaaclab")
        model.save(out_path)
        vec_env.save(out_path + "_vecnorm.pkl")
        print(f"[train_isaaclab] Modello salvato: {out_path}.zip")
        print(
            "[train_isaaclab] Carica in rl_controller_node.py con:\n"
            f"  --ros-args -p model_path:={out_path}"
        )

    vec_env.close()


# ===========================================================================
# Entry point
# ===========================================================================
def main() -> None:
    config = load_config(args.config)

    # USD di Spot: usa quello configurato o cerca automaticamente
    spot_usd = config.get("paths", {}).get("spot_usd", SPOT_USD_PATH)
    if not Path(spot_usd).exists():
        print(f"[train_isaaclab] ERRORE: USD di Spot non trovato: {spot_usd}")
        print("  Configura 'paths.spot_usd' in rl_params.yaml")
        simulation_app.close()
        sys.exit(1)

    import traceback
    try:
        train(
            num_envs=args.num_envs,
            total_steps=args.total_steps,
            resume_from=args.resume,
            config=config,
            dry_run=args.dry_run,
            spot_usd=spot_usd,
        )
    except Exception as e:
        print("\n\n=== ERRORE FATALE IN PYTHON ===")
        traceback.print_exc()
        print("===============================\n")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
