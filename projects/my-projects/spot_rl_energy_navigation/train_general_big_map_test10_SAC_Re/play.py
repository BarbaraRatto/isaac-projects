"""Evaluate a checkpoint on random held-out big-map missions; one visible robot by default."""
import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from train_general_big_map_test10_SAC_Re.config import LOCAL_DINO, MISSION_CFG, SAC_CFG

from train_general_big_map_test10_SAC_Re.runtime_config import playback_signature


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=9, help='episodes per robot on the selected stage')
    parser.add_argument('--stage', type=int, choices=range(len(MISSION_CFG.distance_bands)), help='default: last evaluated training stage')
    parser.add_argument('--mission-suite',type=Path,help='fixed 9- or 15-episode mixed-distance lab suite')
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20000)
    parser.add_argument('--agent-device', default='cpu')
    parser.add_argument('--dino-model', type=Path, default=LOCAL_DINO)
    parser.add_argument('--dino-batch-size', type=int, default=2)
    parser.add_argument('--map-backend', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--output', type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.episodes, args.dino_batch_size, args.torch_threads, args.num_envs) < 1:
        parser.error('Episodes, batch size and threads must be positive')
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint missing: {args.checkpoint}')
    if args.mission_suite:
        if not args.mission_suite.is_file():
            parser.error('Mission suite JSON not found')
        count=len(json.loads(args.mission_suite.read_text())['cases'])
        if args.num_envs!=1 or args.episodes!=count or count not in (9,15):
            parser.error(f'This mission suite requires --num-envs 1 --episodes {count}')
    args.enable_cameras = True
    app = AppLauncher(args).app
    env = trajectory = diagnostics = None
    try:
        import torch
        from train_general_big_map_test10_SAC_Re.artifacts import source_identity, CSVLog, write_json, EPISODE_FIELDS
        from train_general_big_map_test10_SAC_Re.assets import validate_assets
        from train_general_big_map_test10_SAC_Re.env import GeneralMapEnv
        from train_general_big_map_test10_SAC_Re.evaluation import evaluate
        from train_general_big_map_test10_SAC_Re.layouts import training_conditions
        from train_general_big_map_test10_SAC_Re.networks import SACPolicy
        from train_general_big_map_test10_SAC_Re.perception import FrozenPCA
        from train_general_big_map_test10_SAC_Re.scene import GeneralMapCfg
        from train_general_big_map_test10_SAC_Re.falls import DIAGNOSTIC_FIELDS
        validate_assets()
        torch.set_num_threads(args.torch_threads)
        state = torch.load(args.checkpoint, map_location=args.agent_device, weights_only=False)
        if state['signature'] != playback_signature(SAC_CFG, state['signature']) or state['identity'] != source_identity(args.dino_model):
            raise ValueError('Checkpoint/configuration/asset mismatch')
        pca = FrozenPCA(**state['pca'])
        policy = SACPolicy().to(args.agent_device).eval()
        policy.load_state_dict(state['learner']['policy'])
        cfg = GeneralMapCfg()
        conditions = training_conditions(args.num_envs)
        cfg.scene.num_envs = len(conditions)
        cfg.seed, cfg.sim.device = args.seed, args.device
        env = GeneralMapEnv(cfg, conditions, pca=pca, model_path=args.dino_model,
                        dino_batch_size=args.dino_batch_size, map_backend=args.map_backend)
        env.capture_fall_diagnostics = True
        suite=None
        if args.mission_suite:
            from train_general_big_map_test10_SAC_Re.missions_15 import SuiteSampler
            env.sampler=SuiteSampler(args.mission_suite)
            suite=env.sampler.document
        output = args.output or HERE/'results'/datetime.now().strftime('evaluation_%Y%m%d_%H%M%S_%f')
        output.mkdir(parents=True, exist_ok=False)
        if suite is not None:
            write_json(output/f'missions_{len(suite["cases"])}.json',suite)
        write_json(output/'evaluation_config.json', dict(checkpoint=str(args.checkpoint.resolve()),
                   seed=args.seed, requested_stage=args.stage, episodes=args.episodes,
                   num_envs=args.num_envs, map_backend=args.map_backend,
                   goal_scaling=dict(mode="fixed_scale", scale_m=10.0),mission_suite=suite))
        diagnostics = CSVLog(output/'fall_diagnostics.csv', DIAGNOSTIC_FIELDS)
        trajectory = CSVLog(output/'trajectories.csv', ['env_id', 'episode', 'step', 'x', 'y',
                                                         'energy_j', 'phase', 'goal_x', 'goal_y', 'yaw', 'command_vx', 'command_wz',
                                                         'action_forward', 'action_yaw'])
        summary, records, _ = evaluate(env, policy, args.agent_device, args.episodes, args.seed, app, trajectory,
                                       stage=args.stage if args.stage is not None else state['training_state'].get('last_evaluated_stage', 0),
                                       diagnostics=diagnostics)
        if suite is not None:
            summary['stage_name']=suite['name']
            summary['mission_suite']=suite['name']
            for record in records:
                case=suite['cases'][record['episode']-1]
                record['mission_category']=case['category']
                record['source_episode']=case.get('source_episode',case['episode'])
        log = CSVLog(output/'episodes.csv', [*EPISODE_FIELDS, 'episode']+(['mission_category','source_episode'] if suite else []))
        try:
            for record in records:
                log.write(dict(transitions=state['learner']['transitions'], **record))
        finally:
            log.close()
        write_json(output/'summary.json', dict(checkpoint=str(args.checkpoint.resolve()), seed=args.seed,
                                              num_envs=args.num_envs,
                                              goal_scaling=dict(mode="fixed_scale", scale_m=10.0),
                                              summary=summary, episodes=records,mission_suite=suite))
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
