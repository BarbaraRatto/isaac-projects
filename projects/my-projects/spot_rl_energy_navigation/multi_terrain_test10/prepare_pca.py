"""Scripted RGB-D collection on both small mirrored boards, followed by frozen PCA fitting."""
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from multi_terrain_test10.config import DEFAULT_PCA, LOCAL_DINO


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_PCA)
    parser.add_argument('--collect-steps', type=int, default=1500, help='vector steps; not RL training transitions')
    parser.add_argument('--sample-every', type=int, default=5)
    parser.add_argument('--samples-per-terrain', type=int, default=10000)
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--dino-batch-size', type=int, default=4)
    parser.add_argument('--dino-model', type=Path, default=LOCAL_DINO)
    parser.add_argument('--torch-threads', type=int, default=4)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new --output to preserve the fitted PCA')
    if args.num_envs < 2 or args.num_envs % 2 or min(args.collect_steps, args.sample_every, args.dino_batch_size,
                                                 args.samples_per_terrain, args.torch_threads) < 1:
        parser.error('Positive settings and an even --num-envs >= 2 are required')
    args.enable_cameras = True
    launcher = AppLauncher(args)
    app = launcher.app
    env = None
    try:
        import numpy as np
        import torch
        from multi_terrain_test10.assets import validate_assets
        from multi_terrain_test10.artifacts import source_identity, write_json
        from multi_terrain_test10.config import NAV_CFG, TERRAIN_CODES
        from multi_terrain_test10.env import Test10Env
        from multi_terrain_test10.layouts import training_conditions
        from multi_terrain_test10.perception import FrozenPCA
        from multi_terrain_test10.scene import Test10Cfg
        validate_assets()
        identity = source_identity(args.dino_model)
        torch.set_num_threads(args.torch_threads)
        cfg = Test10Cfg()
        cfg.scene.num_envs = args.num_envs
        cfg.seed, cfg.sim.device = args.seed, args.device
        conditions = training_conditions(args.num_envs)
        env = Test10Env(cfg, conditions, capture=True, dino_batch_size=args.dino_batch_size,
                        model_path=args.dino_model)
        env.reset(seed=args.seed)
        random = np.random.default_rng(args.seed)
        # Independent reservoir per terrain actually present in these two boards.
        codes = sorted({TERRAIN_CODES[name] for c in conditions for row in c.case.grid for name in row})
        samples = {code: [] for code in codes}
        seen = dict.fromkeys(codes, 0)
        collection_episodes = []
        waypoints = torch.zeros((args.num_envs, 2), device=env.device)
        stages = np.zeros(args.num_envs, dtype=int)
        episodes = np.zeros(args.num_envs, dtype=int)
        # Both border routes and a diagonal route; speed/yaw variation is seeded.
        routes = (((3.8, 0.0), (4.0, 3.8), (4.0, 4.0)),
                  ((0.0, 3.8), (3.8, 4.0), (4.0, 4.0)),
                  ((2.0, 2.0), (3.5, 3.5), (4.0, 4.0)))

        def collect():
            for i, (points, features) in enumerate(env.vision.samples):
                xy = points[:, :2]-env.scene.env_origins[i, :2].cpu().numpy()
                cols = np.floor((xy[:, 0]+1)/2).astype(int)
                rows = np.floor((xy[:, 1]+1)/2).astype(int)
                for row, col, feature in zip(rows, cols, features):
                    if not (0 <= row < 3 and 0 <= col < 3):
                        continue
                    code = TERRAIN_CODES[conditions[i].case.grid[row][col]]
                    seen[code] += 1
                    bucket = samples[code]
                    if len(bucket) < args.samples_per_terrain:
                        bucket.append(feature.copy())
                    else:
                        index = int(random.integers(seen[code]))
                        if index < len(bucket):
                            bucket[index] = feature.copy()

        collect()
        for step in range(args.collect_steps):
            if not app.is_running():
                raise RuntimeError('Collection stopped before completion; PCA not saved')
            xy = env.robot.data.root_pos_w[:, :2]-env.scene.env_origins[:, :2]
            for i in range(args.num_envs):
                route = routes[(i//2+episodes[i]) % len(routes)]
                if np.linalg.norm(xy[i].cpu().numpy()-route[stages[i]]) < 0.45:
                    stages[i] = min(stages[i]+1, len(route)-1)
                waypoints[i] = torch.tensor(route[stages[i]], device=env.device)
            delta = waypoints-xy
            error = torch.atan2(delta[:, 1], delta[:, 0])-env.yaw()
            error = torch.atan2(error.sin(), error.cos())
            speed = (0.45+0.15*torch.sin(torch.tensor(step*0.07, device=env.device))) * error.cos().clamp_min(0)
            actions = torch.stack((2*speed/NAV_CFG.max_forward_m_s-1,
                                   (1.8*error/NAV_CFG.max_yaw_rad_s).clamp(-1, 1)), dim=1)
            env.vision.capture = (step+1) % args.sample_every == 0
            _, _, _, _, info = env.step(actions)
            if env.vision.capture:
                collect()
            for record in info['episodes']:
                i = record['env_id']
                collection_episodes.append(dict(record, collection_step=step+1,
                                                route=int((i//2+episodes[i]) % len(routes))))
                stages[i] = 0
                episodes[i] += 1
            if (step+1) % 50 == 0:
                print(f'[PCA] {step+1}/{args.collect_steps} vector steps; samples={ {k: len(v) for k,v in samples.items()} }', flush=True)
        counts = {k: len(v) for k, v in samples.items()}
        if min(counts.values()) < 250:
            raise RuntimeError(f'Insufficient coverage of a terrain: {counts}; collect longer before fitting')
        # Equal terrain weight in fitting, without duplicating rare samples.
        per_class = min(counts.values())
        balanced = np.concatenate([np.asarray(samples[k])[random.choice(counts[k], per_class, replace=False)] for k in codes])
        metadata = dict(identity=identity, seed=args.seed, collect_steps=args.collect_steps,
                        samples_seen=seen, reservoir_counts=counts, fitted_per_terrain=per_class,
                        fit_samples=len(balanced), whitening=False, dataset='scripted_small_mirrored_boards')
        pca = FrozenPCA.fit(balanced, metadata)
        pca.save(args.output)
        np.savez_compressed(args.output.with_suffix('.samples.npz'), features=balanced,
                            terrain_codes=np.repeat(codes, per_class))
        write_json(args.output.with_suffix('.json'), dict(**metadata, explained_variance_32=float(pca.explained.sum())))
        write_json(args.output.with_suffix('.collection.json'), collection_episodes)
        print(f'[PCA] saved {args.output}; explained variance={pca.explained.sum():.3%}', flush=True)
    finally:
        if env is not None:
            env.close()
        app.close(skip_cleanup=args.headless)


if __name__ == '__main__':
    main()
