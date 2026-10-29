"""Evaluate a checkpoint on random held-out big-map missions; one visible robot by default."""
import argparse
from datetime import datetime
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from train_general_big_map_test5.config import LOCAL_DINO, MISSION_CFG, experiment_signature


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=9, help='episodes per robot on the selected stage')
    parser.add_argument('--stage', type=int, choices=range(len(MISSION_CFG.distance_bands)), help='default: last evaluated training stage')
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20000)
    parser.add_argument('--agent-device', default='cpu')
    parser.add_argument('--dino-model', type=Path, default=LOCAL_DINO)
    parser.add_argument('--dino-batch-size', type=int, default=2)
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--output', type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.episodes, args.dino_batch_size, args.torch_threads, args.num_envs) < 1:
        parser.error('Episodes, batch size and threads must be positive')
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint missing: {args.checkpoint}')
    args.enable_cameras = True
    app = AppLauncher(args).app
    env = trajectory = diagnostics = None
    try:
        import torch
        from train_general_big_map_test5.artifacts import source_identity, CSVLog, write_json, EPISODE_FIELDS
        from train_general_big_map_test5.assets import validate_assets
        from train_general_big_map_test5.env import GeneralMapEnv
        from train_general_big_map_test5.evaluation import evaluate
        from train_general_big_map_test5.layouts import training_conditions
        from train_general_big_map_test5.networks import ActorCritic
        from train_general_big_map_test5.perception import FrozenPCA
        from train_general_big_map_test5.scene import GeneralMapCfg
        from train_general_big_map_test5.falls import DIAGNOSTIC_FIELDS
        validate_assets()
        torch.set_num_threads(args.torch_threads)
        state = torch.load(args.checkpoint, map_location=args.agent_device, weights_only=False)
        if state['signature'] != experiment_signature() or state['identity'] != source_identity(args.dino_model):
            raise ValueError('Checkpoint/configuration/asset mismatch')
        pca = FrozenPCA(**state['pca'])
        policy = ActorCritic().to(args.agent_device).eval()
        policy.load_state_dict(state['learner']['policy'])
        cfg = GeneralMapCfg()
        conditions = training_conditions(args.num_envs)
        cfg.scene.num_envs = len(conditions)
        cfg.seed, cfg.sim.device = args.seed, args.device
        env = GeneralMapEnv(cfg, conditions, pca=pca, model_path=args.dino_model,
                        dino_batch_size=args.dino_batch_size)
        env.capture_fall_diagnostics = True
        output = args.output or HERE/'results'/datetime.now().strftime('evaluation_%Y%m%d_%H%M%S_%f')
        output.mkdir(parents=True, exist_ok=False)
        diagnostics = CSVLog(output/'fall_diagnostics.csv', DIAGNOSTIC_FIELDS)
        trajectory = CSVLog(output/'trajectories.csv', ['env_id', 'episode', 'step', 'x', 'y',
                                                         'energy_j', 'phase', 'goal_x', 'goal_y', 'yaw', 'command_vx', 'command_wz',
                                                         'action_forward', 'action_yaw'])
        summary, records, _ = evaluate(env, policy, args.agent_device, args.episodes, args.seed, app, trajectory,
                                       stage=args.stage if args.stage is not None else state['training_state'].get('last_evaluated_stage', 0),
                                       diagnostics=diagnostics)
        log = CSVLog(output/'episodes.csv', [*EPISODE_FIELDS, 'episode'])
        try:
            for record in records:
                log.write(dict(transitions=state['learner']['transitions'], **record))
        finally:
            log.close()
        write_json(output/'summary.json', dict(checkpoint=str(args.checkpoint.resolve()), seed=args.seed,
                                              num_envs=args.num_envs,
                                              summary=summary, episodes=records))
        print(f'[GENERAL_MAP] {summary}\n[GENERAL_MAP] results={output}', flush=True)
    finally:
        if diagnostics is not None:
            diagnostics.close()
        if trajectory is not None:
            trajectory.close()
        if env is not None:
            env.close()
        app.close(skip_cleanup=args.headless)


if __name__ == '__main__':
    main()
