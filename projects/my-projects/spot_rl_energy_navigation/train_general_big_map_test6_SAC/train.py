"""Train asymmetric twin-Q SAC from scratch for 100,000 navigation transitions."""
import argparse
from datetime import datetime
from pathlib import Path
import random
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from train_general_big_map_test6_SAC.config import SAC_CFG, ALIGN_CFG, DEFAULT_PCA, LOCAL_DINO, MISSION_CFG, experiment_signature


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timesteps', type=int, default=SAC_CFG.timesteps, help='total target, including resumed transitions')
    parser.add_argument('--pca', type=Path, default=DEFAULT_PCA)
    parser.add_argument('--resume', type=Path, help='final/interrupted SAC checkpoint with its matching replay snapshot')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--agent-device', default='cpu')
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--dino-batch-size', type=int, default=4)
    parser.add_argument('--dino-model', type=Path, default=LOCAL_DINO)
    parser.add_argument('--eval-every', type=int, default=SAC_CFG.eval_every)
    parser.add_argument('--checkpoint-every', type=int, default=SAC_CFG.checkpoint_every)
    parser.add_argument('--probe-steps', type=int, default=0, help='technical simulator check; no SAC or replay updates')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.timesteps,args.torch_threads,args.dino_batch_size,args.checkpoint_every)<1 or min(args.eval_every,args.probe_steps)<0:
        parser.error('Invalid budget, thread, batch or interval settings')
    if not args.resume and not args.pca.is_file():
        parser.error(f'Frozen PCA missing: {args.pca}')
    args.enable_cameras = True
    app = AppLauncher(args).app
    env = None
    logs = []
    try:
        import copy
        import numpy as np
        import torch
        from train_general_big_map_test6_SAC.artifacts import (source_identity, CSVLog, EPISODE_FIELDS, write_json,
            save_checkpoint, restore_rng, snapshot_sources, pca_compatible)
        from train_general_big_map_test6_SAC.assets import validate_assets
        from train_general_big_map_test6_SAC.calibration import validate_calibration
        from train_general_big_map_test6_SAC.env import GeneralMapEnv
        from train_general_big_map_test6_SAC.evaluation import evaluate
        from train_general_big_map_test6_SAC.layouts import training_conditions
        from train_general_big_map_test6_SAC.networks import ExportedActor
        from train_general_big_map_test6_SAC.perception import FrozenPCA
        from train_general_big_map_test6_SAC.replay import ReplayBuffer, collect_step
        from train_general_big_map_test6_SAC.sac import SAC
        from train_general_big_map_test6_SAC.scene import GeneralMapCfg
        validate_assets()
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_num_threads(args.torch_threads)
        identity = source_identity(args.dino_model)
        state = torch.load(args.resume, map_location=args.agent_device, weights_only=False) if args.resume else None
        if state is not None and (state['signature'] != experiment_signature() or state['identity'] != identity):
            raise ValueError('Checkpoint configuration, model or assets differ')
        pca = FrozenPCA(**state['pca']) if state else FrozenPCA.load(args.pca)
        if not pca_compatible(pca.metadata.get('identity', {}), identity):
            raise ValueError('PCA DINO/preprocessing mismatch')
        validate_calibration(pca, identity)
        replay = ReplayBuffer(SAC_CFG.buffer_size)
        learner = SAC(args.agent_device)
        policy, device = learner.policy, learner.device
        cfg = GeneralMapCfg()
        cfg.seed, cfg.sim.device = args.seed, args.device
        cfg.scene.num_envs = SAC_CFG.num_envs
        env = GeneralMapEnv(cfg, training_conditions(SAC_CFG.num_envs), pca=pca,
                            dino_batch_size=args.dino_batch_size, model_path=args.dino_model)
        best_score, best_by_stage, last_evaluated_stage = None, {}, 0
        if state:
            replay_info = state['training_state'].get('replay_snapshot')
            if not replay_info:
                raise ValueError('This model checkpoint has no replay snapshot. Resume requires a final/interrupted checkpoint and matching replay archive.')
            replay.load(args.resume.parent/replay_info['file'])
            learner.load_state_dict(state['learner'])
            if replay.total != learner.transitions or replay_info['transitions'] != learner.transitions:
                raise ValueError('Replay and checkpoint transition counts differ')
            env.curriculum.load_state_dict(state['training_state']['curriculum'])
            best_score, best_by_stage = state['best_score'], state['training_state'].get('best_by_stage', {})
            last_evaluated_stage = state['training_state'].get('last_evaluated_stage', 0)
            restore_rng(state['rng'])
        if learner.transitions >= args.timesteps:
            raise ValueError('Requested --timesteps has already been reached')
        obs, _ = env.reset(seed=None if state else args.seed)
        if args.probe_steps:
            for _ in range(args.probe_steps):
                obs, _, _, _, info = env.step(torch.tensor((-.5,0.),device=env.device).repeat(env.num_envs,1))
                print(f'[PROBE] actor={tuple(obs["policy"].shape)} critic={tuple(obs["critic"].shape)} '
                      f'navigation={int(info["policy_mask"].sum())}',flush=True)
            return
        if not state:
            policy.normalizer.update(obs['policy'].to(device),obs['critic'].to(device))
        run = args.run_dir or HERE/'runs'/datetime.now().strftime('general_map_test6_sac_%Y%m%d_%H%M%S_%f')
        run.mkdir(parents=True, exist_ok=False)
        snapshot_sources(run)
        pca.save(run/'pca32.npz')
        write_json(run/'config.json',dict(signature=experiment_signature(),identity=identity,
            arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            initial_transitions=learner.transitions, replay_estimated_gib=replay.nbytes/1024**3,
            parameters=sum(p.numel() for p in policy.parameters()),
            training_conditions=[dict(env_id=c.env_id,layout=c.layout.name) for c in env.conditions],
            evaluation_in_training_buffer=False, resume_resets_simulation=True))
        (run/'architecture.txt').write_text(str(policy)+'\n')
        episode_log=CSVLog(run/'episodes.csv',EPISODE_FIELDS)
        metric_names=['actor_loss','q1_loss','q2_loss','alpha_loss','alpha','entropy','q_mean','target_q_mean']
        fields=['transitions','wall_seconds','transitions_per_second','replay_size','updates','vector_steps',
                'alignment_transitions','discarded_final_samples','visual_cells','geometry_cells',
                'visual_remembered_cells','geometry_remembered_cells','action_saturation','curriculum_level',*metric_names]
        train_log=CSVLog(run/'training.csv',fields)
        eval_log=CSVLog(run/'evaluations.csv',['transitions','stage','episodes','success_rate','mean_success_energy_j'])
        logs=[episode_log,train_log,eval_log]
        started=time.perf_counter()
        initial=learner.transitions
        episode_count=success_count=vector_steps=alignment_count=discarded_count=no_navigation=0
        next_log=(initial//SAC_CFG.log_every+1)*SAC_CFG.log_every
        next_checkpoint=(initial//args.checkpoint_every+1)*args.checkpoint_every
        next_eval=(initial//args.eval_every+1)*args.eval_every if args.eval_every else None
        last_metrics={key:None for key in metric_names}
        interrupted=False
        replay_snapshot=None

        def save(name):
            save_checkpoint(run/name,learner,pca,identity,best_score,training_state=dict(
                curriculum=env.curriculum.state_dict(),best_by_stage=best_by_stage,
                last_evaluated_stage=last_evaluated_stage,replay_snapshot=replay_snapshot))

        def log_progress():
            raw=obs['policy'].to(device)
            cells=raw[:,15:].reshape(len(raw),81,40)
            seconds=time.perf_counter()-started
            train_log.write(dict(transitions=learner.transitions,wall_seconds=seconds,
                transitions_per_second=(learner.transitions-initial)/max(seconds,1e-9),
                replay_size=replay.size,updates=learner.updates,vector_steps=vector_steps,
                alignment_transitions=alignment_count,discarded_final_samples=discarded_count,
                visual_cells=float(cells[...,36].sum(1).mean()),geometry_cells=float(cells[...,37].sum(1).mean()),
                visual_remembered_cells=float(((cells[...,36]>.5)&(cells[...,38]>1e-6)).float().sum(1).mean()),
                geometry_remembered_cells=float(((cells[...,37]>.5)&(cells[...,39]>1e-6)).float().sum(1).mean()),
                action_saturation=step_stats['action_saturation'],curriculum_level=env.curriculum.level,**last_metrics))
            print(f'[SAC] transitions={learner.transitions}/{args.timesteps} updates={learner.updates} '
                  f'replay={replay.size} episodes={episode_count} successes={success_count} '
                  f'alpha={last_metrics["alpha"]}',flush=True)

        print(f'[SAC] run={run}\n[SAC] replay RAM capacity={replay.nbytes/1024**3:.2f} GiB; '
              '1 SAC update per vector step containing navigation samples, after warmup.',flush=True)
        try:
            while learner.transitions < args.timesteps:
                if not app.is_running():
                    raise KeyboardInterrupt
                obs,step_stats=collect_step(env,policy,obs,replay,device,args.timesteps-learner.transitions,
                                           warmup=learner.transitions<SAC_CFG.learning_starts)
                count=step_stats['count']
                learner.transitions+=count
                vector_steps+=1
                alignment_count+=step_stats['alignment']
                discarded_count+=step_stats['discarded']
                no_navigation=0 if count else no_navigation+1
                if no_navigation>=ALIGN_CFG.max_steps_without_navigation:
                    raise RuntimeError('No navigation samples for 60 simulated seconds; inspect alignment')
                for record in step_stats['episodes']:
                    episode_log.write(dict(transitions=learner.transitions,**record))
                    episode_count+=1
                    success_count+=record['success']
                if count and learner.transitions>=SAC_CFG.learning_starts and replay.size>=SAC_CFG.batch_size:
                    for _ in range(SAC_CFG.gradient_steps):
                        last_metrics=learner.update(replay.sample(SAC_CFG.batch_size,device))
                if learner.transitions>=next_log or learner.transitions==args.timesteps:
                    log_progress()
                    next_log=(learner.transitions//SAC_CFG.log_every+1)*SAC_CFG.log_every
                if learner.transitions>=next_checkpoint:
                    save(f'policy_{learner.transitions}.pt')
                    next_checkpoint=(learner.transitions//args.checkpoint_every+1)*args.checkpoint_every
                if (next_eval is not None and learner.transitions>=next_eval) or learner.transitions==args.timesteps:
                    summary,records,score=evaluate(env,policy,device,episodes_per_env=SAC_CFG.eval_episodes_per_env,
                                                  seed=MISSION_CFG.evaluation_seed,app=app)
                    write_json(run/f'evaluation_{learner.transitions}.json',dict(summary=summary,episodes=records))
                    eval_log.write(dict(transitions=learner.transitions,stage=summary['stage'],episodes=summary['episodes'],
                                        success_rate=summary['success_rate'],mean_success_energy_j=summary['mean_success_energy_j']))
                    last_evaluated_stage=summary['stage']
                    rate=summary['by_band'][str(env.curriculum.level)]['success_rate']
                    promoted=env.curriculum.update(rate)
                    write_json(run/'curriculum.json',dict(transitions=learner.transitions,**env.curriculum.state_dict(),promoted=promoted))
                    finite_score=(score[0],score[1] if score[0] else -1e30)
                    stage_key=str(last_evaluated_stage)
                    if stage_key not in best_by_stage or finite_score>tuple(best_by_stage[stage_key]):
                        best_by_stage[stage_key]=finite_score
                        save(f'policy_best_stage_{stage_key}.pt')
                    ranked=(last_evaluated_stage,*finite_score)
                    if best_score is None or ranked>tuple(best_score):
                        best_score=ranked
                        save('policy_best.pt')
                    print(f'[EVAL] {summary}',flush=True)
                    if promoted:
                        print(f'[CURRICULUM] advanced to band {env.curriculum.level}',flush=True)
                    obs,_=env.reset()
                    save('policy_latest.pt')
                    next_eval=(learner.transitions//args.eval_every+1)*args.eval_every if args.eval_every else None
        except KeyboardInterrupt:
            interrupted=True
            print('[SAC] Interrupted; saving model and matching replay snapshot.',flush=True)
        replay_name=f'replay_{learner.transitions}.npz'
        print(f'[SAC] Saving replay ({replay.size} samples); this may take a little time.',flush=True)
        replay.save(run/replay_name)
        replay_snapshot=dict(file=replay_name,transitions=learner.transitions)
        name='policy_interrupted.pt' if interrupted else 'policy_final.pt'
        save(name)
        exported=ExportedActor(copy.deepcopy(policy).cpu().eval())
        torch.jit.trace(exported,torch.zeros(1,3255)).save(str(run/'actor.pt'))
        write_json(run/'summary.json',dict(interrupted=interrupted,transitions=learner.transitions,updates=learner.updates,
            episodes=episode_count,successes=success_count,wall_seconds=time.perf_counter()-started,checkpoint=str(run/name)))
        print(f'[SAC] Saved {run/name}',flush=True)
    finally:
        for log in logs:
            log.close()
        if env is not None:
            env.close()
        app.close(skip_cleanup=args.headless)


if __name__=='__main__':
    main()
