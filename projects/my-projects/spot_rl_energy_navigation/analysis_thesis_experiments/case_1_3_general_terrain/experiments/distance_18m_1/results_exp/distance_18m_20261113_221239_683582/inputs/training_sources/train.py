"""Train asymmetric twin-Q SAC from scratch for 1,500,000 navigation transitions with periodic complete recovery."""
import argparse
from datetime import datetime
from pathlib import Path
import random
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from train_general_big_map_test9_SAC.config import SAC_CFG, ALIGN_CFG, DEFAULT_PCA, LOCAL_DINO, MISSION_CFG

from train_general_big_map_test9_SAC.runtime_config import training_settings, runtime_signature


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timesteps', type=int, default=SAC_CFG.timesteps, help='total target, including resumed transitions')
    parser.add_argument('--num-envs', type=int, default=SAC_CFG.num_envs, help='parallel robot environments')
    parser.add_argument('--gradient-steps', type=int, default=SAC_CFG.gradient_steps,
                        help='SAC updates per vector step; not automatically scaled with num-envs')
    parser.add_argument('--pca', type=Path, default=DEFAULT_PCA)
    parser.add_argument('--resume', type=Path, help='run directory (verified fallback), complete snapshot, or final/interrupted checkpoint')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--recovery-minutes', type=float, default=30., help='wall-time interval between complete snapshots')
    parser.add_argument('--recovery-keep', type=int, default=2, help='completed recovery generations to retain (minimum 2)')
    parser.add_argument('--recovery-reserve-gib', type=float, default=1., help='free disk reserve after a new full snapshot')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--agent-device', default='cpu')
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--dino-batch-size', type=int, default=4)
    parser.add_argument('--map-backend', choices=('cpu', 'cuda'), default='cpu',
                        help='persistent map/geometry backend')
    parser.add_argument('--rgb-backend', choices=('opencv', 'cuda'), default='opencv',
                        help='CUDA equivalent of uint8 OpenCV resize + processor; checked on first real frames')
    parser.add_argument('--replay-device', default='cpu', help='cpu or cuda:0; storage backend does not change checkpoint schema')
    parser.add_argument('--profile-timing', action='store_true', help='synchronized phase timings in timing.json')
    parser.add_argument('--dino-model', type=Path, default=LOCAL_DINO)
    parser.add_argument('--eval-every', type=int, default=SAC_CFG.eval_every)
    parser.add_argument('--checkpoint-every', type=int, default=SAC_CFG.checkpoint_every)
    parser.add_argument('--probe-steps', type=int, default=0, help='technical simulator check; no SAC or replay updates')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.timesteps,args.torch_threads,args.dino_batch_size,args.checkpoint_every,args.num_envs,args.gradient_steps)<1 or min(args.eval_every,args.probe_steps)<0:
        parser.error('Invalid budget, thread, batch or interval settings')
    if not args.resume and not args.pca.is_file():
        parser.error(f'Frozen PCA missing: {args.pca}')
    if args.recovery_minutes <= 0 or args.recovery_keep < 2 or args.recovery_reserve_gib < 0:
        parser.error('Invalid recovery interval, retention or disk reserve')
    if args.rgb_backend == 'cuda' and (args.map_backend != 'cuda' or not args.device.startswith('cuda')):
        parser.error('--rgb-backend cuda requires --map-backend cuda and --device cuda:0')
    from train_general_big_map_test9_SAC.recovery import (resolve_resume, RecoveryManager,
        check_space, graceful_signals)
    if args.resume:
        args.resume = resolve_resume(args.resume)
        print(f'[RECOVERY] Verified checkpoint selected: {args.resume}', flush=True)
    settings = training_settings(SAC_CFG, args.num_envs, args.gradient_steps)
    args.enable_cameras = True
    app = None
    env = None
    logs = []
    try:
        import copy
        import numpy as np
        import torch
        from train_general_big_map_test9_SAC.artifacts import (source_identity, CSVLog, EPISODE_FIELDS, write_json,
            save_checkpoint, restore_rng, snapshot_sources, pca_compatible)
        from train_general_big_map_test9_SAC.assets import validate_assets
        from train_general_big_map_test9_SAC.calibration import validate_calibration
        from train_general_big_map_test9_SAC.evaluation import evaluate
        from train_general_big_map_test9_SAC.layouts import training_conditions
        from train_general_big_map_test9_SAC.networks import ExportedActor
        from train_general_big_map_test9_SAC.perception import FrozenPCA
        from train_general_big_map_test9_SAC.replay import ReplayBuffer, collect_step
        from train_general_big_map_test9_SAC.sac import SAC, materialize_metrics
        from train_general_big_map_test9_SAC.timing import PhaseTimers
        validate_assets()
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_num_threads(args.torch_threads)
        identity = source_identity(args.dino_model)
        state = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
        if state is not None and (state['signature'] != runtime_signature(settings) or state['identity'] != identity):
            raise ValueError('Checkpoint configuration, model or assets differ')
        pca = FrozenPCA(**state['pca']) if state else FrozenPCA.load(args.pca)
        if not pca_compatible(pca.metadata.get('identity', {}), identity):
            raise ValueError('PCA DINO/preprocessing mismatch')
        validate_calibration(pca, identity)
        run = args.run_dir or HERE/'runs'/datetime.now().strftime('general_map_test9_sac_%Y%m%d_%H%M%S_%f')
        if not args.probe_steps and run.exists():
            raise FileExistsError(f'Choose a NEW --run-dir; refusing to overwrite {run}')
        parent = run.parent
        while not parent.exists():
            parent = parent.parent
        from train_general_big_map_test9_SAC.config import OBSERVATION_SIZE, PRIVILEGED_SIZE
        replay_bytes = settings.buffer_size*((2*OBSERVATION_SIZE+2*PRIVILEGED_SIZE+3)*4+1)
        if not args.probe_steps:
            check_space(parent, replay_bytes, args.recovery_reserve_gib)
        replay = ReplayBuffer(settings.buffer_size, device=args.replay_device)
        learner = SAC(args.agent_device, settings=settings)
        policy, device = learner.policy, learner.device
        best_score, best_by_stage, last_evaluated_stage = None, {}, 0
        if state:
            replay_info = state['training_state'].get('replay_snapshot')
            if not replay_info:
                raise ValueError('This model checkpoint has no replay snapshot. Resume from a complete recovery snapshot or its run directory.')
            replay.load(args.resume.parent/replay_info['file'])
            learner.load_state_dict(state['learner'])
            if replay.total != learner.transitions or replay_info['transitions'] != learner.transitions:
                raise ValueError('Replay and checkpoint transition counts differ')
            best_score, best_by_stage = state['best_score'], state['training_state'].get('best_by_stage', {})
            last_evaluated_stage = state['training_state'].get('last_evaluated_stage', 0)
        if learner.transitions >= args.timesteps:
            raise ValueError('Requested --timesteps has already been reached')
        app = AppLauncher(args).app
        from train_general_big_map_test9_SAC.env import GeneralMapEnv
        from train_general_big_map_test9_SAC.scene import GeneralMapCfg
        cfg = GeneralMapCfg()
        cfg.seed, cfg.sim.device = args.seed, args.device
        cfg.scene.num_envs = settings.num_envs
        env = GeneralMapEnv(cfg, training_conditions(settings.num_envs), pca=pca,
                            dino_batch_size=args.dino_batch_size, model_path=args.dino_model,
                            map_backend=args.map_backend, rgb_backend=args.rgb_backend)
        timers = PhaseTimers(args.profile_timing, args.device)
        if hasattr(env, 'vision'):
            env.vision.timers = timers
        if state:
            env.curriculum.load_state_dict(state['training_state']['curriculum'])
            restore_rng(state['rng'])
        obs, _ = env.reset(seed=None if state else args.seed)
        if args.probe_steps:
            for _ in range(args.probe_steps):
                obs, _, _, _, info = env.step(torch.tensor((-.5,0.),device=env.device).repeat(env.num_envs,1))
                print(f'[PROBE] actor={tuple(obs["policy"].shape)} critic={tuple(obs["critic"].shape)} '
                      f'navigation={int(info["policy_mask"].sum())}',flush=True)
            return
        if not state:
            policy.normalizer.update(obs['policy'].to(device),obs['critic'].to(device))
        run.mkdir(parents=True, exist_ok=False)
        check_space(run, replay.nbytes, args.recovery_reserve_gib)
        snapshot_sources(run)
        pca.save(run/'pca32.npz')
        write_json(run/'config.json',dict(signature=runtime_signature(settings),identity=identity,
            arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            initial_transitions=learner.transitions, replay_estimated_gib=replay.nbytes/1024**3,
            parameters=sum(p.numel() for p in policy.parameters()),
            training_conditions=[dict(env_id=c.env_id,layout=c.layout.name) for c in env.conditions],
            evaluation_in_training_buffer=False, resume_resets_simulation=True,
            recovery_interval_seconds=args.recovery_minutes*60,
            software=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                          cuda=torch.version.cuda)))
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
        next_log=(initial//settings.log_every+1)*settings.log_every
        next_checkpoint=(initial//args.checkpoint_every+1)*args.checkpoint_every
        next_eval=(initial//args.eval_every+1)*args.eval_every if args.eval_every else None
        last_metrics={key:None for key in metric_names}
        interrupted=False
        recovery = RecoveryManager(run, args.recovery_minutes*60, args.recovery_keep, args.recovery_reserve_gib)

        def training_state():
            return dict(curriculum=env.curriculum.state_dict(), best_by_stage=best_by_stage,
                        last_evaluated_stage=last_evaluated_stage,
                        counters=dict(episode_count=episode_count, success_count=success_count,
                                      vector_steps=vector_steps, alignment_count=alignment_count,
                                      discarded_count=discarded_count),
                        parent_checkpoint=str(args.resume) if args.resume else None)

        def save(name, complete=None):
            state = training_state()
            state['replay_snapshot'] = None
            if complete is not None:
                relative = complete.relative_to(run)
                state['complete_snapshot'] = str(relative)
                state['replay_snapshot'] = dict(file=str(relative/'replay.npz'), transitions=learner.transitions)
            with timers.measure('model_checkpoint'):
                save_checkpoint(run/name, learner, pca, identity, best_score, training_state=state)

        def save_complete(reason):
            with timers.measure('complete_backup'):
                result = recovery.save(learner, replay, pca, identity, best_score, training_state(), reason)
            if timers.enabled:
                write_json(run/'timing.json', timers.report())
            return result

        def log_progress():
            metrics = materialize_metrics(last_metrics)
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
                action_saturation=step_stats['action_saturation'],curriculum_level=env.curriculum.level,**metrics))
            if timers.enabled:
                write_json(run/'timing.json', timers.report())
            print(f'[SAC] transitions={learner.transitions}/{args.timesteps} updates={learner.updates} '
                  f'replay={replay.size} episodes={episode_count} successes={success_count} '
                  f'alpha={metrics["alpha"]}',flush=True)

        print(f'[SAC] run={run}\n[SAC] replay device={replay.device}, capacity={replay.nbytes/1024**3:.2f} GiB; '
              f'{settings.gradient_steps} SAC updates per vector step containing navigation samples, after warmup.',flush=True)
        save_complete('initial_or_resume')
        with graceful_signals() as stop_request:
            try:
                while learner.transitions < args.timesteps:
                    if stop_request.signal is not None or not app.is_running():
                        raise KeyboardInterrupt
                    with timers.measure('collect'):
                        obs,step_stats=collect_step(env,policy,obs,replay,device,args.timesteps-learner.transitions,
                                                   warmup=learner.transitions<settings.learning_starts)
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
                    if count and learner.transitions>=settings.learning_starts and replay.size>=settings.batch_size:
                        with timers.measure('sac_updates'):
                            for _ in range(settings.gradient_steps):
                                last_metrics=learner.update(replay.sample(settings.batch_size,device), defer_metrics=True)
                    if learner.transitions>=next_log or learner.transitions==args.timesteps:
                        log_progress()
                        next_log=(learner.transitions//settings.log_every+1)*settings.log_every
                    if learner.transitions>=next_checkpoint:
                        save(f'policy_{learner.transitions}.pt')
                        next_checkpoint=(learner.transitions//args.checkpoint_every+1)*args.checkpoint_every
                    if recovery.due():
                        save_complete('periodic')
                    if stop_request.signal is not None:
                        raise KeyboardInterrupt
                    if (next_eval is not None and learner.transitions>=next_eval) or learner.transitions==args.timesteps:
                        with timers.measure('evaluation'):
                            summary,records,score=evaluate(env,policy,device,episodes_per_env=settings.eval_episodes_per_env,
                                                          seed=MISSION_CFG.evaluation_seed,app=app,
                                                          stop_requested=lambda: stop_request.signal is not None)
                        if timers.enabled:
                            write_json(run/'timing.json', timers.report())
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
                        if promoted:
                            save_complete('curriculum_promotion')
                        save('policy_latest.pt')
                        next_eval=(learner.transitions//args.eval_every+1)*args.eval_every if args.eval_every else None
            except KeyboardInterrupt:
                interrupted=True
                print('[SAC] Interrupted; saving model and matching replay snapshot.',flush=True)
            except Exception as exc:
                write_json(run/'failure.json', dict(error=repr(exc), transitions=learner.transitions,
                           recovery_status='Use the last COMPLETE snapshot; current in-memory update may be partial.'))
                print(f'[RECOVERY] Training failed. Previous complete snapshots remain in {run / "recovery"}.', flush=True)
                raise
            complete = save_complete('interrupted' if interrupted else 'final')
            name='policy_interrupted.pt' if interrupted else 'policy_final.pt'
            save(name, complete=complete)
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
        if app is not None:
            app.close(skip_cleanup=args.headless)


if __name__=='__main__':
    main()
