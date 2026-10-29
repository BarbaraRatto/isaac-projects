"""Collect current-map RGB-D views, fit PCA32, validate on separate views, then freeze."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from train_general_big_map_test5.config import DEFAULT_PCA, LOCAL_DINO, PCA_DATASET, TERRAIN_NAMES


def update_manifest(output, pca):
    if output.resolve() != DEFAULT_PCA.resolve():
        return
    from train_general_big_map_test5.config import experiment_signature
    from train_general_big_map_test5.perception import file_sha256
    path = HERE/'artifacts/manifest.json'
    manifest = json.loads(path.read_text())
    manifest.update(signature=experiment_signature(), pca_sha256=file_sha256(output),
                    pca_source=str(output.relative_to(HERE)), pca_status='ready',
                    pca_calibration_identity=pca.metadata['identity'])
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2)+'\n')
    temporary.replace(path)


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_PCA)
    parser.add_argument('--reuse', action='store_true', help='Reuse only a validated compatible artifact')
    parser.add_argument('--views-per-terrain', type=int, default=96)
    parser.add_argument('--validation-views', type=int, default=24)
    parser.add_argument('--samples-per-terrain', type=int, default=10000)
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--render-frames', type=int, default=3)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--dino-batch-size', type=int, default=4)
    parser.add_argument('--dino-model', type=Path, default=LOCAL_DINO)
    parser.add_argument('--torch-threads', type=int, default=4)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.views_per_terrain, args.validation_views, args.num_envs,
           args.render_frames, args.dino_batch_size, args.torch_threads) < 1 or args.samples_per_terrain < 1000:
        parser.error('Positive counts and at least 1000 samples per terrain are required')
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the lock alive until completion, including the atomic artifact commit.
    with args.output.with_suffix('.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another process is calibrating this PCA; wait for it to finish') from None
        run(args, AppLauncher)


def run(args, AppLauncher):
    import numpy as np
    import torch
    from train_general_big_map_test5.artifacts import source_identity, write_json
    from train_general_big_map_test5.assets import validate_assets
    from train_general_big_map_test5.calibration import (view_plan, point_labels, Reservoir,
                                                       validate_calibration, validation_metrics)
    from train_general_big_map_test5.layouts import map_metadata, training_conditions
    from train_general_big_map_test5.perception import FrozenPCA
    validate_assets()
    identity = source_identity(args.dino_model)
    if args.output.exists():
        if not args.reuse:
            raise FileExistsError('PCA already exists; use --reuse or choose a new --output')
        pca = FrozenPCA.load(args.output)
        validate_calibration(pca, identity)
        update_manifest(args.output, pca)
        print(f'[PCA] Compatible seven-terrain PCA already available: {args.output}', flush=True)
        return
    torch.set_num_threads(args.torch_threads)
    with np.load(HERE/'artifacts/spawn_bank.npz', allow_pickle=False) as bank:
        points, evaluation = bank['points'], bank['evaluation']
    meta = map_metadata()
    plan = view_plan(points, evaluation, meta, args.seed, args.views_per_terrain, args.validation_views)
    reservoirs = {'train': Reservoir(args.samples_per_terrain, args.seed),
                  'validation': Reservoir(min(2500, args.samples_per_terrain), args.seed+1)}
    args.enable_cameras = True
    launcher = AppLauncher(args)
    app, env = launcher.app, None
    started = time.monotonic()
    try:
        from train_general_big_map_test5.env import GeneralMapEnv
        from train_general_big_map_test5.scene import GeneralMapCfg
        cfg = GeneralMapCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.seed, cfg.sim.device = args.seed, args.device
        env = GeneralMapEnv(cfg, training_conditions(args.num_envs), capture=True,
                            dino_batch_size=args.dino_batch_size, model_path=args.dino_model)
        ids = torch.arange(env.num_envs, device=env.device)
        print(f'[PCA] Acquiring {len(plan)} static views on the CURRENT map, all 7 materials; '
              'no RL updates during calibration.', flush=True)
        for first in range(0, len(plan), env.num_envs):
            if not app.is_running():
                raise RuntimeError('PCA collection interrupted; training will not start')
            views = plan[first:first+env.num_envs]
            padded = views+[views[-1]]*(env.num_envs-len(views))
            root = env.settled_root.clone()
            for i, view in enumerate(padded):
                point = points[view['point_id']]
                root[i, :2] = env.scene.env_origins[i, :2]+torch.as_tensor(point[:2], device=env.device)
                root[i, 2] = env.scene.env_origins[i, 2]+float(point[3])+env.standing_clearance[i]+.02
                yaw = view['yaw']
                root[i, 3:7] = torch.tensor([np.cos(yaw/2), 0., 0., np.sin(yaw/2)], device=env.device)
                root[i, 7:] = 0
            # Deliberate static camera acquisition: do not walk, align, collect
            # rewards or use the RL curriculum to determine visual coverage.
            env.robot.write_root_state_to_sim(root)
            env.robot.write_joint_state_to_sim(env.settled_joint_pos, env.settled_joint_vel)
            env.robot.reset()
            env.robot.update(env.physics_dt)
            env.vision.reset(ids)
            env._sync_cameras()
            for _ in range(args.render_frames):
                env.sim.render()
            env.camera.update(0., force_recompute=True)
            d = env.robot.data
            env.vision.extract(env.camera, d.root_pos_w, env.yaw(), d.body_link_pos_w,
                               torch.zeros(env.num_envs, device=env.device))
            for i, view in enumerate(views):
                xyz, features = env.vision.samples[i]
                xy = xyz[:, :2]-env.scene.env_origins[i, :2].cpu().numpy()
                labels = point_labels(xy, meta)
                reservoirs[view['split']].add(features, labels)
                view['observed_patch_counts'] = np.bincount(labels[labels >= 0], minlength=7).tolist()
            done = first+len(views)
            if first == 0 or done % (env.num_envs*10) == 0 or done == len(plan):
                elapsed = time.monotonic()-started
                counts = {split: [len(x) for x in reservoir.samples] for split, reservoir in reservoirs.items()}
                print(f'[PCA] views {done}/{len(plan)}, {elapsed/60:.1f} min; '
                      f'ETA acquisition ~{elapsed/done*(len(plan)-done)/60:.1f} min; counts={counts}', flush=True)
        write_json(args.output.with_suffix('.collection.json'), dict(views=plan, camera_mode='static_settled_robot_pose',
                   num_envs=env.num_envs, render_frames=args.render_frames, elapsed_seconds=time.monotonic()-started))
        train_x, train_y = reservoirs['train'].balanced(1000)
        valid_x, valid_y = reservoirs['validation'].balanced(250)
        metadata = dict(identity=identity, dataset=PCA_DATASET, seed=args.seed,
                        materials=list(TERRAIN_NAMES), whitening=False,
                        fitted_per_terrain=len(train_x)//7, fit_samples=len(train_x),
                        validation_counts=np.bincount(valid_y, minlength=7).tolist(),
                        samples_seen={s: r.seen.tolist() for s, r in reservoirs.items()},
                        frames_per_material={s: r.frames.tolist() for s, r in reservoirs.items()},
                        camera_pose_split='disjoint subsets of training spawn-bank points; evaluation endpoints excluded',
                        views_per_terrain=args.views_per_terrain, validation_views=args.validation_views)
        pca = FrozenPCA.fit(train_x, metadata)
        report = validation_metrics(pca, valid_x, valid_y)
        validate_calibration(pca, identity)
        np.savez_compressed(args.output.with_suffix('.samples.npz'), features=train_x, terrain_codes=train_y,
                            validation_features=valid_x, validation_terrain_codes=valid_y)
        write_json(args.output.with_suffix('.json'), dict(**metadata, fit_variance_retained=float(pca.explained.sum()),
                                                         validation=report))
        temporary = args.output.with_suffix('.tmp')
        pca.save(temporary)
        temporary.replace(args.output)
        update_manifest(args.output, pca)
        print(f'[PCA] Saved {args.output}; fit variance={pca.explained.sum():.1%}, '
              f'separate-view variance={report["overall"]["variance_retained"]:.1%}. Calibration complete.', flush=True)
    finally:
        if env is not None:
            env.close()
        app.close(skip_cleanup=args.headless)


if __name__ == '__main__':
    main()
