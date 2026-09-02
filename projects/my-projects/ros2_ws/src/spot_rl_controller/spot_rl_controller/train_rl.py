#!/usr/bin/env python3
"""
train_rl.py

Script di training PPO per la policy RL di Spot.

Uso:
    python3 train_rl.py                         # training completo
    python3 train_rl.py --dry-run --n-steps 200  # smoke test (200 step)
    python3 train_rl.py --resume models/best_model  # riprende da checkpoint

Prerequisiti:
  - Isaac Sim in esecuzione (headless o meno)
  - ROS2 sourced (ros2 topic list deve mostrare /terrain_gridmap, /energy/...)
  - energy_costmap_node in esecuzione (ros2 run spot_rl_controller energy_costmap_node)
  - Spot che cammina sul terreno (teleop o CHAMP con /cmd_vel)

Monitoraggio:
    tensorboard --logdir ./logs/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import yaml
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Aggiungi il pacchetto al path se eseguito standalone
_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from spot_rl_controller.spot_env import SpotEnv
from spot_rl_controller.spot_policy_net import SpotFeaturesExtractor


# ===========================================================================
# Caricamento configurazione
# ===========================================================================
def load_config(config_path: str | Path) -> dict:
    """Carica il file YAML di configurazione."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ===========================================================================
# Creazione env wrappato per SB3
# ===========================================================================
def make_env(config: dict) -> SpotEnv:
    """Factory function per creare e wrappare l'env (Monitor + logging)."""
    log_dir = config.get("paths", {}).get("log_dir", "logs/")
    os.makedirs(log_dir, exist_ok=True)

    env = SpotEnv(config)
    env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
    return env


# ===========================================================================
# Configurazione policy_kwargs per SB3
# ===========================================================================
def build_policy_kwargs(config: dict, env: SpotEnv) -> dict:
    """
    Costruisce i kwargs per la policy SB3 con il features extractor custom.
    """
    net_cfg = config.get("network", {})
    return {
        "features_extractor_class": SpotFeaturesExtractor,
        "features_extractor_kwargs": {
            "features_dim":      net_cfg.get("features_dim",      256),
            "cnn_channels":      net_cfg.get("cnn_channels",      [32, 64, 128]),
            "cnn_kernel_size":   net_cfg.get("cnn_kernel_size",   3),
            "mlp_hidden_sizes":  net_cfg.get("mlp_hidden_sizes",  [512, 256]),
        },
        # Layer della policy MLP (dopo il features extractor)
        "net_arch": [256, 128],
    }


# ===========================================================================
# Training
# ===========================================================================
def train(config: dict, resume_from: str | None = None, dry_run: bool = False,
          dry_run_steps: int = 200) -> None:
    """
    Esegue il training PPO.

    Parametri
    ---------
    config       : configurazione caricata dal YAML.
    resume_from  : percorso a un checkpoint SB3 da cui riprendere (opzionale).
    dry_run      : se True, esegue solo dry_run_steps e non salva il modello.
    dry_run_steps: numero di step per il dry run.
    """
    ppo_cfg   = config.get("ppo", {})
    paths_cfg = config.get("paths", {})

    model_dir      = paths_cfg.get("model_dir",      "models/")
    log_dir        = paths_cfg.get("log_dir",         "logs/")
    best_model_name = paths_cfg.get("best_model_name", "best_model")
    final_model_name = paths_cfg.get("final_model_name", "final_model")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)

    total_timesteps = (
        dry_run_steps if dry_run
        else int(ppo_cfg.get("total_timesteps", 1_000_000))
    )

    # ── Creazione env ──────────────────────────────────────────────────────
    print("[train_rl] Creazione Gymnasium env...")
    env = make_env(config)

    # VecEnv con normalizzazione reward (consigliato per PPO)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=False,   # la normalizzazione obs avviene nell'env (clip)
        norm_reward=True,
        clip_reward=10.0,
    )

    # ── Policy kwargs ──────────────────────────────────────────────────────
    policy_kwargs = build_policy_kwargs(config, env)

    # ── Creazione modello PPO ─────────────────────────────────────────────
    if resume_from is not None and os.path.exists(resume_from + ".zip"):
        print(f"[train_rl] Riprendendo da checkpoint: {resume_from}")
        model = PPO.load(resume_from, env=vec_env)
        # Ricarica anche la VecNormalize se esiste
        vec_norm_path = resume_from + "_vecnorm.pkl"
        if os.path.exists(vec_norm_path):
            vec_env = VecNormalize.load(vec_norm_path, vec_env)
            print(f"[train_rl] VecNormalize caricata da {vec_norm_path}")
    else:
        print("[train_rl] Creazione nuovo modello PPO...")
        model = PPO(
            policy="MultiInputPolicy",
            env=vec_env,
            learning_rate=float(ppo_cfg.get("learning_rate",   3e-4)),
            n_steps=int(ppo_cfg.get("n_steps",                 2048)),
            batch_size=int(ppo_cfg.get("batch_size",           256)),
            n_epochs=int(ppo_cfg.get("n_epochs",               10)),
            gamma=float(ppo_cfg.get("gamma",                   0.99)),
            gae_lambda=float(ppo_cfg.get("gae_lambda",         0.95)),
            clip_range=float(ppo_cfg.get("clip_range",         0.2)),
            ent_coef=float(ppo_cfg.get("ent_coef",             0.01)),
            vf_coef=float(ppo_cfg.get("vf_coef",               0.5)),
            max_grad_norm=float(ppo_cfg.get("max_grad_norm",   0.5)),
            verbose=1,
            tensorboard_log=log_dir,
            policy_kwargs=policy_kwargs,
            device="cuda",
        )

    print(f"[train_rl] Parametri policy:\n{model.policy}")

    # ── Callback ──────────────────────────────────────────────────────────
    callbacks = []

    if not dry_run:
        # Salva checkpoint ogni 10.000 step
        checkpoint_cb = CheckpointCallback(
            save_freq=10_000,
            save_path=model_dir,
            name_prefix="spot_rl",
            save_vecnormalize=True,
        )
        callbacks.append(checkpoint_cb)

    callback_list = CallbackList(callbacks)

    # ── Training ───────────────────────────────────────────────────────────
    print(
        f"\n[train_rl] Inizio training "
        f"({'DRY RUN' if dry_run else 'FULL'}, "
        f"{total_timesteps} step)...\n"
    )
    t0 = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        log_interval=int(ppo_cfg.get("log_interval", 10)),
        reset_num_timesteps=(resume_from is None),
        progress_bar=True,
    )

    elapsed = time.time() - t0
    print(f"\n[train_rl] Training completato in {elapsed/60:.1f} min")

    # ── Salvataggio finale ─────────────────────────────────────────────────
    if not dry_run:
        final_path = os.path.join(model_dir, final_model_name)
        model.save(final_path)
        vec_env.save(final_path + "_vecnorm.pkl")
        print(f"[train_rl] Modello salvato in: {final_path}.zip")

    vec_env.close()


# ===========================================================================
# Entry point
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Training PPO per Spot energy-aware navigation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Percorso al file rl_params.yaml. "
             "Default: cerca config/rl_params.yaml vicino allo script.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Percorso al checkpoint SB3 da cui riprendere (senza .zip)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Esegui solo --n-steps step (smoke test, non salva)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=200,
        help="Numero di step per il dry run (default: 200)",
    )
    args = parser.parse_args()

    # Ricerca automatica del config
    if args.config is None:
        candidates = [
            Path(__file__).parent.parent / "config" / "rl_params.yaml",
            Path("config/rl_params.yaml"),
            Path("rl_params.yaml"),
        ]
        for c in candidates:
            if c.exists():
                args.config = str(c)
                break
        if args.config is None:
            print(
                "ERRORE: config/rl_params.yaml non trovato. "
                "Usa --config per specificarlo."
            )
            sys.exit(1)

    print(f"[train_rl] Config: {args.config}")
    config = load_config(args.config)

    train(
        config=config,
        resume_from=args.resume,
        dry_run=args.dry_run,
        dry_run_steps=args.n_steps,
    )


if __name__ == "__main__":
    main()
