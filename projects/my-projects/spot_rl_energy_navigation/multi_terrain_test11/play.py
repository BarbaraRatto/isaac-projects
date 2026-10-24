"""Evaluate one checkpoint on BOTH small mirrored boards; GUI unless --headless."""
import argparse
from datetime import datetime
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from multi_terrain_test11.config import LOCAL_DINO, experiment_signature


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=5, help='episodes per map')
    parser.add_argument('--layout', choices=('both', 'rocks_asphalt__right_rocks',
                                             'rocks_asphalt__right_asphalt'), default='both',
                        help='Evaluate both in parallel or one map in a single environment')
    parser.add_argument('--seed', type=int, default=20000)
    parser.add_argument('--agent-device', default='cpu')
    parser.add_argument('--dino-model', type=Path, default=LOCAL_DINO)
    parser.add_argument('--dino-batch-size', type=int, default=2)
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--output', type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.episodes, args.dino_batch_size, args.torch_threads) < 1:
        parser.error('Episodes, batch size and threads must be positive')
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint missing: {args.checkpoint}')
    args.enable_cameras = True
    app = AppLauncher(args).app
    env = trajectory = None
    try:
        import torch
        from multi_terrain_test11.artifacts import source_identity, CSVLog, write_json, EPISODE_FIELDS
        from multi_terrain_test11.assets import validate_assets
        from multi_terrain_test11.env import Test11Env
        from multi_terrain_test11.evaluation import evaluate
        from multi_terrain_test11.layouts import training_conditions, evaluation_condition
        from multi_terrain_test11.networks import ActorCritic
        from multi_terrain_test11.perception import FrozenPCA
        from multi_terrain_test11.scene import Test11Cfg
        validate_assets()
        torch.set_num_threads(args.torch_threads)
        state = torch.load(args.checkpoint, map_location=args.agent_device, weights_only=False)
        if state['signature'] != experiment_signature() or state['identity'] != source_identity(args.dino_model):
            raise ValueError('Checkpoint/configuration/asset mismatch')
        pca = FrozenPCA(**state['pca'])
        policy = ActorCritic().to(args.agent_device).eval()
        policy.load_state_dict(state['learner']['policy'])
        cfg = Test11Cfg()
        conditions = training_conditions(2) if args.layout == 'both' else (evaluation_condition(args.layout),)
        cfg.scene.num_envs = len(conditions)
        cfg.seed, cfg.sim.device = args.seed, args.device
        env = Test11Env(cfg, conditions, pca=pca, model_path=args.dino_model,
                        dino_batch_size=args.dino_batch_size)
        output = args.output or HERE/'results'/datetime.now().strftime('evaluation_%Y%m%d_%H%M%S_%f')
        output.mkdir(parents=True, exist_ok=False)
        trajectory = CSVLog(output/'trajectories.csv', ['env_id', 'episode', 'step', 'x', 'y',
                                                         'energy_j', 'action_forward', 'action_yaw'])
        summary, records, _ = evaluate(env, policy, args.agent_device, args.episodes, args.seed, app, trajectory)
        log = CSVLog(output/'episodes.csv', [*EPISODE_FIELDS, 'episode'])
        try:
            for record in records:
                log.write(dict(transitions=state['learner']['transitions'], **record))
        finally:
            log.close()
        write_json(output/'summary.json', dict(checkpoint=str(args.checkpoint.resolve()), seed=args.seed,
                                              layout=args.layout,
                                              summary=summary, episodes=records))
        print(f'[TEST11] {summary}\n[TEST11] results={output}', flush=True)
    finally:
        if trajectory is not None:
            trajectory.close()
        if env is not None:
            env.close()
        app.close(skip_cleanup=args.headless)


if __name__ == '__main__':
    main()
