"""Random A-B missions on frozen real_terrains_v2; exactly 200,000 transitions by default."""
import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from train_general_big_map_test2.config import DEFAULT_PCA, LOCAL_DINO, PPO_CFG, MISSION_CFG, experiment_signature, OBSERVATION_SIZE, CRITIC_SIZE


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timesteps', type=int, default=PPO_CFG.timesteps, help='total target, also when resuming')
    parser.add_argument('--pca', type=Path, default=DEFAULT_PCA)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--agent-device', default='cpu')
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--dino-batch-size', type=int, default=4)
    parser.add_argument('--dino-model', type=Path, default=LOCAL_DINO)
    parser.add_argument('--eval-every', type=int, default=PPO_CFG.eval_every, help='0 disables in-training evaluation')
    parser.add_argument('--checkpoint-every', type=int, default=PPO_CFG.checkpoint_every)
    parser.add_argument('--probe-steps', type=int, default=0, help='technical simulator check only, no PPO updates')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.timesteps < 32 or args.timesteps % 32:
        parser.error('--timesteps must be a positive multiple of 32')
    if min(args.torch_threads, args.dino_batch_size, args.checkpoint_every) < 1 or min(args.eval_every, args.probe_steps) < 0:
        parser.error('Invalid thread/batch/checkpoint/evaluation/probe settings')
    if not args.resume and not args.pca.is_file():
        parser.error(f'Frozen PCA artifact missing: {args.pca}')
    args.enable_cameras = True
    launcher = AppLauncher(args)
    app = launcher.app
    env = None
    logs = []
    try:
        import copy
        import numpy as np
        import torch
        from train_general_big_map_test2.artifacts import (source_identity, CSVLog, EPISODE_FIELDS, write_json,
                                                     save_checkpoint, restore_rng, snapshot_sources, pca_compatible)
        from train_general_big_map_test2.assets import validate_assets
        from train_general_big_map_test2.env import GeneralMapEnv
        from train_general_big_map_test2.evaluation import evaluate
        from train_general_big_map_test2.layouts import training_conditions
        from train_general_big_map_test2.networks import ExportedActor
        from train_general_big_map_test2.perception import FrozenPCA
        from train_general_big_map_test2.ppo import PPO, rollout_length
        from train_general_big_map_test2.scene import GeneralMapCfg
        validate_assets()
        torch.set_num_threads(args.torch_threads)
        identity = source_identity(args.dino_model)
        state = torch.load(args.resume, map_location=args.agent_device, weights_only=False) if args.resume else None
        if state is not None:
            if state['signature'] != experiment_signature() or state['identity'] != identity:
                raise ValueError('Checkpoint configuration, DINO weights or terrain assets differ')
            pca = FrozenPCA(**state['pca'])
        else:
            pca = FrozenPCA.load(args.pca)
        if not pca_compatible(pca.metadata.get('identity', {}), identity):
            raise ValueError('PCA DINO weights or preprocessing differ')
        cfg = GeneralMapCfg()
        cfg.seed, cfg.sim.device = args.seed, args.device
        cfg.scene.num_envs = PPO_CFG.num_envs
        conditions = training_conditions(cfg.scene.num_envs)
        env = GeneralMapEnv(cfg, conditions, pca=pca, dino_batch_size=args.dino_batch_size, model_path=args.dino_model)
        learner = PPO(args.agent_device)
        best_score = None
        best_by_stage = {}
        last_evaluated_stage = 0
        if state is not None:
            learner.load_state_dict(state['learner'])
            best_score = state['best_score']
            best_by_stage = state['training_state'].get('best_by_stage', {})
            last_evaluated_stage = state['training_state'].get('last_evaluated_stage', 0)
            restore_rng(state['rng'])
            env.curriculum.load_state_dict(state['training_state']['curriculum'])
        if learner.transitions >= args.timesteps:
            raise ValueError('Checkpoint has already reached the requested total --timesteps')
        obs, _ = env.reset(seed=None if state else args.seed)
        if args.probe_steps:
            for _ in range(args.probe_steps):
                action = torch.tensor((-0.5, 0.0), device=env.device).repeat(env.num_envs, 1)
                obs, _, _, _, info = env.step(action)
                print(f'[PROBE] actor={tuple(obs["policy"].shape)} critic={tuple(obs["critic"].shape)} '
                      f'cells={env.visual_valid.mean():.1f}/{env.geometry_valid.mean():.1f} '
                      f'energy_cost={info["reward_components"][:,1].mean():.4f}', flush=True)
            return
        run = args.run_dir or HERE/'runs'/datetime.now().strftime('general_map_test2_ppo_%Y%m%d_%H%M%S_%f')
        run.mkdir(parents=True, exist_ok=False)
        snapshot_sources(run)
        pca.save(run/'pca32.npz')
        write_json(run/'config.json', dict(signature=experiment_signature(), identity=identity,
                                          arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                                          training_conditions=[dict(env_id=c.env_id, layout=c.layout.name) for c in conditions],
                                          initial_transitions=learner.transitions,
                                          parameters=sum(p.numel() for p in learner.policy.parameters()),
                                          resume_resets_simulation=True, evaluation_in_training_buffer=False))
        (run/'architecture.txt').write_text(str(learner.policy)+'\n')
        episode_log = CSVLog(run/'episodes.csv', EPISODE_FIELDS)
        fields = ['transitions', 'rollout_steps', 'wall_seconds', 'transitions_per_second', 'actor_loss',
                  'nav_value_loss', 'energy_value_loss', 'entropy', 'kl', 'clip_fraction', 'early_stop',
                  'minibatch_updates', 'total_updates', 'visual_cells', 'geometry_cells',
                  'action_saturation', 'mean_nav_reward', 'mean_energy_cost',
                  'visual_remembered_cells', 'geometry_remembered_cells', 'curriculum_level']
        training_log = CSVLog(run/'training.csv', fields)
        eval_log = CSVLog(run/'evaluations.csv', ['transitions', 'stage', 'episodes', 'success_rate', 'mean_success_energy_j'])
        logs.extend((episode_log, training_log, eval_log))
        policy = learner.policy
        device = learner.device
        if state is None:
            policy.normalizer.update(obs['policy'].to(device), obs['critic'].to(device))
        next_checkpoint = (learner.transitions//args.checkpoint_every+1)*args.checkpoint_every
        next_eval = (learner.transitions//args.eval_every+1)*args.eval_every if args.eval_every else None
        started = time.perf_counter()
        initial_transitions = learner.transitions
        interrupted = False
        episode_count = success_count = 0
        print(f'[GENERAL_MAP] {run}\n[GENERAL_MAP] target={args.timesteps}; actor={OBSERVATION_SIZE} critic={CRITIC_SIZE}; PPO rollout=4096', flush=True)

        def save(name):
            save_checkpoint(run/name, learner, pca, identity, best_score,
                            training_state=dict(curriculum=env.curriculum.state_dict(), best_by_stage=best_by_stage,
                                                last_evaluated_stage=last_evaluated_stage))

        try:
            while learner.transitions < args.timesteps:
                steps = rollout_length(args.timesteps-learner.transitions, env.num_envs, PPO_CFG.rollout_steps)
                batch = {k: [] for k in ('actor', 'critic', 'latent', 'log_prob', 'values', 'rewards', 'terminals')}
                visual_counts, geometry_counts, saturation = [], [], []
                visual_remembered, geometry_remembered = [], []
                for _ in range(steps):
                    if not app.is_running():
                        raise KeyboardInterrupt
                    actor, critic = obs['policy'].to(device), obs['critic'].to(device)
                    with torch.no_grad():
                        action, latent, log_prob, values = policy.act(actor, critic)
                    obs, _, terminated, truncated, info = env.step(action)
                    if bool(truncated.any()):
                        raise RuntimeError('This task defines true terminals; unexpected truncation')
                    for key, value in dict(actor=actor, critic=critic, latent=latent, log_prob=log_prob,
                                           values=values, rewards=info['reward_components'].to(device),
                                           terminals=terminated.to(device)).items():
                        batch[key].append(value.detach().clone())
                    learner.transitions += env.num_envs
                    visual_counts.append(float(env.visual_valid.mean()))
                    geometry_counts.append(float(env.geometry_valid.mean()))
                    visual_remembered.append(float(env.visual_remembered.float().mean()))
                    geometry_remembered.append(float(env.geometry_remembered.float().mean()))
                    saturation.append(float((action.abs()>0.98).float().mean()))
                    for record in info['episodes']:
                        episode_log.write(dict(transitions=learner.transitions, **record))
                        episode_count += 1
                        success_count += record['success']
                batch = {k: torch.stack(v) for k,v in batch.items()}
                with torch.no_grad():
                    bootstrap = policy.critic(policy.normalizer.critic(obs['critic'].to(device)))
                metrics = learner.update(batch, bootstrap)
                # No change to preprocessing between rollout likelihoods and their PPO updates.
                policy.normalizer.update(batch['actor'].flatten(0, 1), batch['critic'].flatten(0, 1))
                elapsed = time.perf_counter()-started
                training_log.write(dict(transitions=learner.transitions, rollout_steps=steps, wall_seconds=elapsed,
                                         transitions_per_second=(learner.transitions-initial_transitions)/elapsed,
                                         **metrics, visual_cells=np.mean(visual_counts), geometry_cells=np.mean(geometry_counts),
                                         action_saturation=np.mean(saturation),
                                         visual_remembered_cells=np.mean(visual_remembered),
                                         geometry_remembered_cells=np.mean(geometry_remembered),
                                         mean_nav_reward=float(batch['rewards'][..., 0].mean()),
                                         mean_energy_cost=float(batch['rewards'][..., 1].mean()),
                                         curriculum_level=env.curriculum.level))
                print(f'[GENERAL_MAP] transitions={learner.transitions}/{args.timesteps} '
                      f'episodes={episode_count} successes={success_count} '
                      f'KL={metrics["kl"]:.4f} energy_cost={batch["rewards"][...,1].mean():.3f}', flush=True)
                if learner.transitions >= next_checkpoint:
                    save(f'policy_{learner.transitions}.pt')
                    next_checkpoint = (learner.transitions//args.checkpoint_every+1)*args.checkpoint_every
                if (next_eval is not None and learner.transitions >= next_eval) or learner.transitions == args.timesteps:
                    summary, records, score = evaluate(env, policy, device,
                                                       episodes_per_env=PPO_CFG.eval_episodes_per_env,
                                                       seed=MISSION_CFG.evaluation_seed, app=app)
                    write_json(run/f'evaluation_{learner.transitions}.json', dict(summary=summary, episodes=records))
                    eval_log.write(dict(transitions=learner.transitions, stage=summary['stage'], episodes=summary['episodes'],
                                        success_rate=summary['success_rate'], mean_success_energy_j=summary['mean_success_energy_j']))
                    last_evaluated_stage = summary['stage']
                    rate = summary['by_band'][str(env.curriculum.level)]['success_rate']
                    promoted = env.curriculum.update(rate)
                    write_json(run/'curriculum.json', dict(transitions=learner.transitions,
                               **env.curriculum.state_dict(), promoted=promoted))
                    if promoted:
                        print(f'[CURRICULUM] advanced to band {env.curriculum.level}', flush=True)
                    stage_key = str(last_evaluated_stage)
                    if stage_key not in best_by_stage or score > tuple(best_by_stage[stage_key]):
                        # No successes gives -inf; use finite sentinel for portable metadata.
                        best_by_stage[stage_key] = (score[0], score[1] if score[0] else -1e30)
                        save(f'policy_best_stage_{stage_key}.pt')
                    score = (last_evaluated_stage, *score)
                    if best_score is None or score > tuple(best_score):
                        best_score = score
                        save('policy_best.pt')
                    print(f'[EVAL] {summary}', flush=True)
                    # Fresh training episodes after evaluation; no evaluation observations enter the buffer.
                    obs, _ = env.reset()
                    save('policy_latest.pt')
                    next_eval = (learner.transitions//args.eval_every+1)*args.eval_every if args.eval_every else None
        except KeyboardInterrupt:
            interrupted = True
            print('[GENERAL_MAP] Interrupted; saving current learner. An incomplete rollout is discarded.', flush=True)
        name = 'policy_interrupted.pt' if interrupted else 'policy_final.pt'
        save(name)
        actor = ExportedActor(copy.deepcopy(policy).cpu().eval())
        torch.jit.trace(actor, torch.zeros((1, OBSERVATION_SIZE))).save(str(run/'actor.pt'))
        write_json(run/'summary.json', dict(interrupted=interrupted, transitions=learner.transitions,
                                            updates=learner.updates, episodes=episode_count, successes=success_count,
                                            wall_seconds=time.perf_counter()-started, checkpoint=str(run/name)))
        print(f'[GENERAL_MAP] saved {run/name}', flush=True)
    finally:
        for log in logs:
            log.close()
        if env is not None:
            env.close()
        app.close(skip_cleanup=args.headless)


if __name__ == '__main__':
    main()
