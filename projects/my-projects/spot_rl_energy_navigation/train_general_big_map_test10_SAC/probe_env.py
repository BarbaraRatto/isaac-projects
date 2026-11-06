"""Two-robot, two-step cloud installation check. No learner or scaler fitting."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.dont_write_bytecode = True


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    from train_general_big_map_test10_SAC.assets import validate_assets
    from train_general_big_map_test10_SAC.config import DEFAULT_PCA, LOCAL_DINO, OBSERVATION_SIZE, CRITIC_SIZE
    from train_general_big_map_test10_SAC.artifacts import source_identity
    from train_general_big_map_test10_SAC.calibration import validate_calibration
    from train_general_big_map_test10_SAC.perception import FrozenPCA
    import torch
    validate_assets()
    pca = FrozenPCA.load(DEFAULT_PCA)
    validate_calibration(pca, source_identity(LOCAL_DINO))
    app = env = None
    try:
        app = AppLauncher(args).app
        from train_general_big_map_test10_SAC.env import GeneralMapEnv
        from train_general_big_map_test10_SAC.scene import GeneralMapCfg
        from train_general_big_map_test10_SAC.layouts import training_conditions
        cfg = GeneralMapCfg()
        cfg.scene.num_envs = 2
        cfg.sim.device = args.device
        env = GeneralMapEnv(cfg, training_conditions(2), pca=pca, model_path=LOCAL_DINO,
                            dino_batch_size=2, map_backend='cuda', rgb_backend='cuda')
        obs, _ = env.reset(seed=123)
        for _ in range(2):
            obs, _, _, _, _ = env.step(torch.tensor((-.5, 0.), device=env.device).repeat(2, 1))
            for key, size in (('policy', OBSERVATION_SIZE), ('critic', CRITIC_SIZE)):
                if obs[key].shape != (2, size) or not torch.isfinite(obs[key]).all():
                    raise RuntimeError(f'Invalid {key} sensor observations: {obs[key].shape}')
        print('[PROBE] PASS: two robots, two steps, RGBD/DINO/PCA and finite observations; '
              'no scaler fitting, replay or SAC updates.', flush=True)
    finally:
        if env is not None:
            env.close()
        if app is not None:
            app.close(skip_cleanup=args.headless)


if __name__ == '__main__':
    main()
