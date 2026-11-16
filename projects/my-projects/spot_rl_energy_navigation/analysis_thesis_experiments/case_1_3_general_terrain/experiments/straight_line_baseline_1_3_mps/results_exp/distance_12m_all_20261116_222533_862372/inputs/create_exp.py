"""Repeat the selected thesis missions with a constant-speed line controller."""
import argparse
from datetime import datetime
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode=True
from missions import HERE, TRAINING, MAP, METADATA, load_suite, suite_path, sha, write_json
from logging_utils import CSVLog, validate_logs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distance',type=int,choices=(12,18),required=True)
    parser.add_argument('--category',choices=('all','regular','ice','stairs','surface_choice','multiple_choices'),default='all')
    parser.add_argument('--episodes',type=int,nargs='+',help='Run only these thesis IDs, preserving their numbering')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--torch-threads',type=int,default=4)
    parser.add_argument('--no-plots',action='store_true')
    parser.add_argument('--check',action='store_true',help='Validate inputs without launching Isaac Sim')
    offline=any(flag in sys.argv for flag in ('--check','--help','-h'))
    if offline:
        parser.add_argument('--device',default='cuda:0');parser.add_argument('--headless',action='store_true')
    else:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    if args.torch_threads<1:
        parser.error('Use a positive thread count')
    document=load_suite(args.distance)
    if args.category!='all':
        if args.category not in document['category_order']:
            parser.error('Category is not present at this distance')
        document['cases']=[c for c in document['cases'] if c['category']==args.category]
    if args.episodes is not None:
        if len(args.episodes)!=len(set(args.episodes)) or not set(args.episodes)<={c['episode'] for c in document['cases']}:
            parser.error('Episode IDs must be distinct and present in the requested category')
        document['cases']=[c for c in document['cases'] if c['episode'] in args.episodes]
    total=len(document['cases'])
    if not total:
        parser.error('No missions requested')
    if args.output is not None and args.output.exists():
        parser.error('--output must be a new directory')
    from runtime import validate_map
    from train_general_big_map_test7_SAC.assets import validate_assets, ROBOT_USD, POLICY_WEIGHTS, ZED_USD
    import torch
    torch.set_num_threads(args.torch_threads)
    # Validate the frozen locomotion model without constructing any navigation actor.
    locomotion=torch.jit.load(str(POLICY_WEIGHTS),map_location='cpu').eval()
    with torch.inference_mode():
        probe=locomotion(torch.zeros((1,48)))
    if probe.numel()!=12 or not bool(torch.isfinite(probe).all()):
        raise ValueError('Invalid frozen locomotion model')
    del locomotion
    if args.check:
        validate_assets();validate_map()
        speeds=[c['reference_speed_m_s'] for c in document['cases']]
        print(f'Inputs verified: {total} missions at {args.distance} m; speeds {min(speeds):.6f}–{max(speeds):.6f} m/s; no navigation actor or simulator started.')
        return
    args.enable_cameras=True
    app=AppLauncher(args).app
    env=None;logs=[];created=False
    output=args.output or HERE/'results_exp'/datetime.now().strftime(f'distance_{args.distance}m_{args.category}_%Y%m%d_%H%M%S_%f')
    try:
        from train_general_big_map_test7_SAC.scene import GeneralMapCfg
        from runtime import create_environment
        validate_assets();validate_map()
        output.mkdir(parents=True,exist_ok=False);created=True
        frozen=output/'inputs';frozen.mkdir()
        for path in [MAP,METADATA,HERE/'README.md',*sorted(HERE.glob('*.py')),HERE/'run_exp.sh',suite_path(args.distance),suite_path(args.distance).with_suffix('.csv')]:
            shutil.copy2(path,frozen/path.name)
        training_sources=frozen/'training_sources';training_sources.mkdir()
        for path in sorted(TRAINING.glob('*.py')):
            shutil.copy2(path,training_sources/path.name)
        write_json(frozen/'selected_missions.json',document)
        config=dict(method='Fixed 1.3 m/s start-goal line tracking; no learned navigation policy',fixed_forward_speed_m_s=1.3,
            requested_distance=args.distance,requested_category=args.category,selected_episodes=[c['episode'] for c in document['cases']],
            mission_file='selected_missions.json',frozen_panel_sha256=sha(suite_path(args.distance)),
            map=str(MAP),map_sha256=sha(MAP),map_metadata_sha256=sha(METADATA),
            source_selection=document['source_selection'],source_selection_sha256=document['source_selection_sha256'],
            policy_sources=document['sources'],controller=document['controller'],
            assets={str(p):sha(p) for p in (ROBOT_USD,POLICY_WEIGHTS,ZED_USD)},
            command=sys.argv,device=args.device,simulation_seed=20000,
            energy='Absolute joint mechanical work; physics-step trapezoidal integration',
            time='Simulated time including initial alignment; excludes one-time standing calibration',
            perception='Camera synchronization retained; navigation actor and DINO/PCA feature extraction disabled',
            calibration_point=document['calibration_point'])
        write_json(output/'evaluation_config.json',config)
        cfg=GeneralMapCfg();cfg.scene.num_envs=1;cfg.seed=20000;cfg.sim.device=args.device
        env=create_environment(cfg,document,None,None,'cpu')
        config.update(goal_radius_m=env.nav.goal_radius_m,physics_dt_s=env.physics_dt,navigation_dt_s=env.step_dt)
        write_json(output/'evaluation_config.json',config)
        env.power_log=CSVLog(output/'power_samples.csv',['episode','physics_step','time_s','power_w']);logs.append(env.power_log)
        env.reset(seed=20000)
        initial=env.sample();trajectory=CSVLog(output/'trajectories.csv',list(initial));logs.append(trajectory);trajectory.write(initial)
        episode_log=diagnostic_log=None;records=[]
        while app.is_running() and len(records)<total:
            env.terminal_sample=None
            # LoggedEnv computes line commands and keeps the original alignment state machine.
            with torch.no_grad():
                _,_,_,_,extras=env.step(torch.zeros((1,2),device=env.device))
            trajectory.write(env.terminal_sample if env.terminal_sample is not None else env.sample())
            for sample in extras['fall_diagnostics']:
                episode=document['cases'][len(records)]['episode']
                if diagnostic_log is None:
                    diagnostic_log=CSVLog(output/'fall_diagnostics.csv',['episode',*sample]);logs.append(diagnostic_log)
                diagnostic_log.write(dict(episode=episode,**sample))
            for record in extras['episodes']:
                if episode_log is None:
                    episode_log=CSVLog(output/'episodes.csv',list(record));logs.append(episode_log)
                episode_log.write(record);records.append(record)
                for log in logs:log.flush()
                write_json(output/'summary.json',dict(completed_missions=len(records),episodes=records))
                print(f"[RESULT] E{record['episode']} ({len(records)}/{total}): {record['reason']}; "
                      f"command {record['reference_speed_m_s']:.4f} m/s; actual mean {record['mean_speed_m_s']:.4f} m/s; "
                      f"{record['energy_j']:.1f} J; {record['path_m']:.2f} m; {record['time_s']:.1f} s",flush=True)
                if len(records)<total:trajectory.write(env.sample())
        if len(records)!=total:
            raise RuntimeError(f'Interrupted after {len(records)}/{total} completed missions')
        for log in logs:log.flush()
        checks=validate_logs(output,document,env.physics_dt,cfg.decimation)
        if not args.no_plots:
            from plots import plot_results
            plot_results(output,document)
        write_json(output/'complete.json',checks)
        print(f'[RESULTS] {output.resolve()}',flush=True)
    except BaseException as error:
        if created:write_json(output/'interrupted.json',dict(error=repr(error)))
        raise
    finally:
        for log in logs:log.close()
        if env is not None:env.close()
        app.close(skip_cleanup=args.headless)


if __name__=='__main__':
    main()
