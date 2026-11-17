"""Run the original binary routes and forward commands on the opposite map."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode=True
from references import HERE,PROJECT,TRAINING,PANEL,GROUPS,load
from binary_logging import CSVWriter,file_hash,write_json,validate_results


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('--group',choices=('both',*GROUPS),default='both')
    result.add_argument('--missions',nargs='+',help='Original IDs, for example A1 or B1')
    result.add_argument('--output',type=Path)
    result.add_argument('--lookahead',type=float,default=.35)
    result.add_argument('--torch-threads',type=int,default=4)
    result.add_argument('--no-plots',action='store_true')
    result.add_argument('--check',action='store_true')
    result.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    return result


def worker(args,document,cases):
    from isaaclab.app import AppLauncher
    args.enable_cameras=True
    app=AppLauncher(args).app
    env=None;logs=[];output=args.output/args.group
    try:
        import torch
        torch.set_num_threads(args.torch_threads)
        from multi_terrain_test13.assets import validate_assets,ROBOT_USD,POLICY_WEIGHTS,ZED_USD
        from multi_terrain_test13.layouts import ROCKS,MIRRORED
        from multi_terrain_test13.scene import Test13Cfg
        from runtime import create_environment
        validate_assets()
        selected=[c for c in cases if c['group']==args.group]
        if not selected:raise ValueError('No missions in this group')
        target=selected[0]['mission']['target_map'];layout=ROCKS if target=='rocks_right' else MIRRORED
        output.mkdir(exist_ok=False)
        cfg=Test13Cfg();cfg.scene.num_envs=1;cfg.seed=0;cfg.sim.device=args.device
        env=create_environment(cfg,layout,selected,args.lookahead)
        config=dict(group=args.group,source_map=selected[0]['mission']['source_map'],target_map=target,
            method='Recorded geometric route with original forward commands indexed by simulated time',
            steering='Pure-pursuit route correction; original yaw commands saved as reference only',
            controller=dict(lookahead_m=args.lookahead,max_yaw_rad_s=env.nav.max_yaw_rad_s),
            physics_dt_s=env.physics_dt,action_dt_s=env.step_dt,
            timeout_s=env.nav.max_episode_seconds,goal_radius_m=env.nav.goal_radius_m,
            energy='Absolute joint mechanical work; physics-step trapezoidal integration',
            device=args.device,simulation_seed=0,
            maps=document['maps'],reference_panel_sha256=file_hash(PANEL),
            assets={str(p):file_hash(p) for p in (ROBOT_USD,POLICY_WEIGHTS,ZED_USD)},
            selected_missions=[c['mission']['mission_id'] for c in selected])
        write_json(output/'run_config.json',config)
        env.reset(seed=0)
        initial=env.sample();trajectory=CSVWriter(output/'trajectories.csv',list(initial));logs.append(trajectory)
        trajectory.write(initial)
        env.effort_fields=[f'{name}_effort_nm' for name in env.robot.joint_names]
        env.velocity_fields=[f'{name}_velocity_rad_s' for name in env.robot.joint_names]
        env.joint_log=CSVWriter(output/'joint_samples.csv',
            ['mission_id','physics_step','time_s','power_w',*env.effort_fields,*env.velocity_fields]);logs.append(env.joint_log)
        episode_log=None;records=[]
        while app.is_running() and len(records)<len(selected):
            env.terminal_sample=None
            with torch.inference_mode():
                _,_,_,_,extras=env.step(torch.zeros((1,2),device=env.device))
            trajectory.write(env.terminal_sample if env.terminal_sample is not None else env.sample())
            for record in extras['episodes']:
                if episode_log is None:
                    episode_log=CSVWriter(output/'episodes.csv',list(record));logs.append(episode_log)
                episode_log.write(record);records.append(record)
                for log in logs:log.flush()
                write_json(output/'summary.json',dict(completed_missions=len(records),episodes=records))
                print(f"[RESULT] {record['mission_id']}: {record['reason']}; {record['energy_j']:.1f} J; "
                    f"{record['path_m']:.2f} m; {record['time_s']:.1f} s; max route error "
                    f"{record['max_tracking_error_m']:.3f} m",flush=True)
                if len(records)<len(selected):trajectory.write(env.sample())
        if len(records)!=len(selected):raise RuntimeError('Simulation interrupted before all missions finished')
        for log in logs:log.flush()
        checks=validate_results(output,[c['mission'] for c in selected],cfg.decimation,env.physics_dt)
        # Check the applied command time series before marking the session complete.
        from checks import validate_forward_commands
        validate_forward_commands(output,selected)
        if not args.no_plots:
            from plots import plot_results
            plot_results(output,layout,selected)
        write_json(output/'complete.json',checks)
    except BaseException as error:
        if output.exists():write_json(output/'interrupted.json',dict(error=repr(error)))
        raise
    finally:
        for log in logs:log.close()
        if env is not None:env.close()
        app.close(skip_cleanup=args.headless)


def main():
    argparser=parser()
    if '--worker' in sys.argv:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(argparser)
    else:
        argparser.add_argument('--device',default='cuda:0')
        argparser.add_argument('--headless',action='store_true')
    args=argparser.parse_args()
    if args.lookahead<=0 or args.torch_threads<1:argparser.error('Use positive lookahead and thread count')
    document=load();cases=document['cases']
    if args.group!='both':cases=[c for c in cases if c['group']==args.group]
    if args.missions:
        valid={c['mission']['mission_id'] for c in cases}
        if len(set(args.missions))!=len(args.missions) or not set(args.missions)<=valid:
            argparser.error('Mission IDs must be distinct and belong to the selected group')
        cases=[c for c in cases if c['mission']['mission_id'] in args.missions]
    if args.output is not None and args.output.exists() and not args.worker:
        argparser.error('--output must be a new directory')
    if args.check:
        from multi_terrain_test13.assets import validate_assets,POLICY_WEIGHTS
        import torch
        validate_assets()
        model=torch.jit.load(str(POLICY_WEIGHTS),map_location='cpu').eval()
        with torch.inference_mode():probe=model(torch.zeros((1,48)))
        if probe.numel()!=12 or not bool(torch.isfinite(probe).all()):raise ValueError('Invalid locomotion model')
        print(f'Checked {len(cases)} missions, opposite maps, frozen references and locomotion assets. No simulator started.')
        return
    if args.worker:
        if args.group=='both' or args.output is None:argparser.error('Worker requires a group and output')
        worker(args,document,cases);return
    output=(args.output or HERE/'results_exp'/datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')).resolve()
    output.mkdir(parents=True,exist_ok=False)
    frozen=output/'inputs';shutil.copytree(HERE/'inputs',frozen)
    for path in sorted(HERE.glob('*.py')):shutil.copy2(path,frozen/path.name)
    for name in ('README.md','run_exp.sh'):shutil.copy2(HERE/name,frozen/name)
    training_sources=frozen/'training_sources';training_sources.mkdir()
    for path in sorted(TRAINING.glob('*.py')):shutil.copy2(path,training_sources/path.name)
    write_json(output/'run_config.json',dict(command=sys.argv,reference_panel_sha256=file_hash(PANEL),
        groups=list(dict.fromkeys(c['group'] for c in cases)),selected_missions=[c['mission']['mission_id'] for c in cases]))
    print(f'[RESULTS] {output}',flush=True)
    try:
        for group in dict.fromkeys(c['group'] for c in cases):
            command=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],
                     '--group',group,'--output',str(output),'--worker']
            subprocess.run(command,check=True)
        write_json(output/'complete.json',dict(groups=list(dict.fromkeys(c['group'] for c in cases))))
    except BaseException as error:
        write_json(output/'interrupted.json',dict(error=repr(error)));raise


if __name__=='__main__':main()
