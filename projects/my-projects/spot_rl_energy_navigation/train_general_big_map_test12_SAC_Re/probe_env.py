"""Two-robot installation probe before PCA fitting; no learner, replay or training."""
import argparse
from pathlib import Path
import sys
import traceback

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))


def main():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    args.enable_cameras=True
    app=AppLauncher(args).app
    env=None
    try:
        import torch
        from train_general_big_map_test12_SAC_Re.assets import validate_assets
        from train_general_big_map_test12_SAC_Re.env import GeneralMapEnv
        from train_general_big_map_test12_SAC_Re.scene import GeneralMapCfg
        from train_general_big_map_test12_SAC_Re.layouts import training_conditions
        from train_general_big_map_test12_SAC_Re.config import LOCAL_DINO,OBSERVATION_SIZE,CRITIC_SIZE
        from train_general_big_map_test12_SAC_Re.fast_rgb import DeviceRGB
        validate_assets()
        cfg=GeneralMapCfg()
        cfg.scene.num_envs=2;cfg.sim.device=args.device
        # Capture mode verifies raw DINO384 without pretending to have a fitted PCA.
        env=GeneralMapEnv(cfg,training_conditions(2),pca=None,capture=True,
                          dino_batch_size=2,model_path=LOCAL_DINO,map_backend='cpu')
        obs,_=env.reset(seed=123)
        for _ in range(2):
            obs,_,_,_,_=env.step(torch.tensor((-.5,0.),device=env.device).repeat(2,1))
            for key,size in [('policy',OBSERVATION_SIZE),('critic',CRITIC_SIZE)]:
                if obs[key].shape!=(2,size) or not bool(torch.isfinite(obs[key]).all()):
                    raise RuntimeError(f'Invalid {key} observations')
        if torch.device(args.device).type=='cuda':
            check=DeviceRGB(env.vision.extractor.processor,args.device)
            print('[PROBE] RGB equivalence:',check.verify(env.camera.data.output['rgb'][...,:3]),flush=True)
        print('[PROBE] PASS: two robots, RGBD, raw DINO and finite observations; '
              'no SAC, replay, scaler fitting or PCA fitting.',flush=True)
    except Exception:
        traceback.print_exc();sys.stderr.flush()
        raise
    finally:
        if env is not None:env.close()
        app.close(skip_cleanup=args.headless)


if __name__=='__main__':main()
