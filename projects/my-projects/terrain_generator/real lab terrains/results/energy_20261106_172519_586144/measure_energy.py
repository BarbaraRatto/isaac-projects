"""Matched straight-traversal mechanical-energy comparison; no RL or camera model."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(PROJECT/'spot_rl_energy_navigation'))

from energy_metrics import TraversalMeter,write_report


def load_test_areas(create_prim):
    """Let Isaac Sim's prim helper author its own transform, exactly once."""
    centres={'grey':-5.,'green':5.}
    for material,y in centres.items():
        print(f'[ENERGY] Loading {material} terrain at Y={y:g} m...',flush=True)
        create_prim('/World/Lab/'+material,'Xform',
                    usd_path=str(HERE/'maps'/f'{material}_flat.usda'),
                    translation=(0.,y,0.))
    return centres


def main():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--speeds',type=float,nargs='+',default=[.6,.9,1.2])
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--distance',type=float,default=6.,help='measured forward distance, excluding approach')
    parser.add_argument('--approach',type=float,default=1.,help='unmeasured initial acceleration distance')
    parser.add_argument('--settle-seconds',type=float,default=2.)
    parser.add_argument('--output',type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    if (args.repeats<1 or not args.speeds or not all(math.isfinite(v) and 0<v<=1.6 for v in args.speeds)
            or not math.isfinite(args.distance) or not math.isfinite(args.approach)
            or not args.distance>0 or args.approach<.5 or args.approach+args.distance>8.
            or not math.isfinite(args.settle_seconds) or args.settle_seconds<1):
        parser.error('Use speeds in (0,1.6], repeats>=1, approach>=0.5 m, distance>0, approach+distance<=8 m, settle>=1 s')
    manifest=json.loads((HERE/'maps/manifest.json').read_text())
    for material in ('grey','green'):
        path=HERE/'maps'/f'{material}_flat.usda'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['maps_sha256'][path.name]:
            parser.error('Terrain asset changed; regenerate with generate.sh')
    if json.loads((HERE/'materials.json').read_text())!=manifest['materials']:
        parser.error('Material settings changed; regenerate terrains with generate.sh')
    output=args.output or HERE/'results'/datetime.now().strftime('energy_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    cfg=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
             terrain=manifest,measurement=dict(physics_dt_s=.005,approach_m=args.approach,
             forward_distance_m=args.distance,power='sum(abs(measured_joint_effort * joint_velocity))',
             excludes='settling and initial approach; includes every physics step within measurement gates'),
             robot_sha256=hashlib.sha256((PROJECT/'IsaacRobotics/assets/spot.usd').read_bytes()).hexdigest(),
             locomotion_sha256=hashlib.sha256((PROJECT/'IsaacRobotics/policies/spot/models/spot_policy.pt').read_bytes()).hexdigest())
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    for name in ('measure_energy.py','energy_metrics.py','materials.json','create_terrains.py'):
        (output/name).write_bytes((HERE/name).read_bytes())
    app=None;streams=[];count=0;stopped=False;failed=False;phase='starting Isaac Sim'
    try:
        app=AppLauncher(args).app
        import torch
        import isaaclab.sim as sim_utils
        from isaacsim.core.utils.prims import create_prim
        from locomotion import create_spot,reset_spot,SpotVelocityController
        from rl_config import PHYSICS_DT
        if PHYSICS_DT!=.005:raise ValueError('Unexpected physics timestep')
        phase='creating simulation context'
        print('[ENERGY] Creating simulation context...',flush=True)
        sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT,device=args.device))
        phase='loading terrain areas'
        centres=load_test_areas(create_prim)
        light=sim_utils.DistantLightCfg(intensity=3000.,angle=1.)
        light.func('/World/defaultLight',light)
        phase='spawning Spot and loading locomotion controller'
        print('[ENERGY] Spawning Spot and loading locomotion controller...',flush=True)
        robot=create_spot();controller=SpotVelocityController(robot,sim.device)
        phase='initializing physics'
        print('[ENERGY] Initializing physics...',flush=True)
        sim.reset()
        trial_fields=['material','repeat','command_speed_m_s','reason','measurement_seconds','energy_j',
                      'path_m','forward_m','j_per_forward_m','j_per_path_m','mean_forward_speed_m_s',
                      'elapsed_walking_seconds','max_tilt_deg','max_lateral_error_m']
        sample_fields=['material','repeat','command_speed_m_s','phase','walking_time_s','x','y','z',
                       'actual_vx_m_s','actual_vy_m_s','tilt_deg','power_w','measured_energy_j','measured_path_m']
        trial_stream=(output/'trials.csv').open('w',newline='',buffering=1);streams.append(trial_stream)
        sample_stream=(output/'samples.csv').open('w',newline='',buffering=1);streams.append(sample_stream)
        trials=csv.DictWriter(trial_stream,fieldnames=trial_fields);trials.writeheader()
        samples=csv.DictWriter(sample_stream,fieldnames=sample_fields);samples.writeheader()
        command=torch.zeros((1,3),device=sim.device)
        physics_step=0

        def advance(vx):
            nonlocal physics_step
            while app.is_running() and not sim.is_playing() and not sim.is_stopped():sim.render()
            if not app.is_running() or sim.is_stopped():raise KeyboardInterrupt
            command.zero_();command[:,0]=vx
            controller.forward(command)
            sim.step(render=not args.headless and physics_step%4==0)
            robot.update(PHYSICS_DT);physics_step+=1

        def state():
            torque=robot.root_physx_view.get_dof_projected_joint_forces()
            power=(torque*robot.data.joint_vel).abs().sum()
            position=robot.data.root_pos_w[0].detach().cpu().tolist()
            velocity=robot.data.root_lin_vel_w[0].detach().cpu().tolist()
            if not bool(torch.isfinite(power)) or not all(math.isfinite(v) for v in position+velocity):
                raise RuntimeError('Non-finite physics measurement')
            tilt=math.degrees(math.acos(max(-1.,min(1.,-float(robot.data.projected_gravity_b[0,2])))))
            return position,velocity,float(power),tilt

        print('[ENERGY] Output: '+str(output),flush=True)
        for speed in args.speeds:
            if stopped:break
            for repeat in range(1,args.repeats+1):
                if stopped:break
                order=('grey','green') if repeat%2 else ('green','grey')
                for material in order:
                    phase=f'measuring {material}, speed={speed:g}, repeat={repeat}'
                    y=centres[material]
                    sim.set_camera_view((4,y-9,7),(4,y,0))
                    reset_spot(robot,(0.,y,.75));controller.reset();robot.update(PHYSICS_DT)
                    meter=TraversalMeter(args.approach,args.approach+args.distance)
                    elapsed=tilted=low=0.;max_tilt=max_lateral=0.;reason='timeout'
                    try:
                        for _ in range(round(args.settle_seconds/PHYSICS_DT)):advance(0.)
                        previous,_,previous_power,tilt=state()
                        if previous[2]<.3 or tilt>60:
                            reason='bad_initial_stance'
                        else:
                            timeout=max(20.,4*(args.approach+args.distance)/speed)
                            for _ in range(math.ceil(timeout/PHYSICS_DT)):
                                advance(speed);position,velocity,power,tilt=state()
                                elapsed+=PHYSICS_DT
                                meter.update(previous[:2],position[:2],previous_power,power,PHYSICS_DT)
                                max_tilt=max(max_tilt,tilt)
                                lateral=abs(position[1]-y);max_lateral=max(max_lateral,lateral)
                                tilted=tilted+PHYSICS_DT if tilt>=90 else 0.
                                low=low+PHYSICS_DT if position[2]<.25 else 0.
                                samples.writerow(dict(material=material,repeat=repeat,command_speed_m_s=speed,
                                    phase='measurement' if meter.active else 'approach',walking_time_s=elapsed,
                                    x=position[0],y=position[1],z=position[2],actual_vx_m_s=velocity[0],
                                    actual_vy_m_s=velocity[1],tilt_deg=tilt,power_w=power,
                                    measured_energy_j=meter.energy_j,measured_path_m=meter.path_m))
                                previous,previous_power=position,power
                                if tilted>=1.:reason='persistent_tilt';break
                                if low>=1.:reason='persistent_low_clearance';break
                                if lateral>2. or not -1.2<position[0]<9.2:reason='left_test_area';break
                                if meter.completed:reason='completed';break
                    except KeyboardInterrupt:
                        reason='interrupted';stopped=True
                    good=reason=='completed'
                    row=dict(material=material,repeat=repeat,command_speed_m_s=speed,reason=reason,
                             measurement_seconds=meter.seconds,energy_j=meter.energy_j,path_m=meter.path_m,
                             forward_m=meter.forward_m,j_per_forward_m=meter.energy_j/meter.forward_m if good else '',
                             j_per_path_m=meter.energy_j/meter.path_m if good else '',
                             mean_forward_speed_m_s=meter.forward_m/meter.seconds if good else '',
                             elapsed_walking_seconds=elapsed,max_tilt_deg=max_tilt,max_lateral_error_m=max_lateral)
                    trials.writerow(row);count+=1
                    value=f'{row["j_per_forward_m"]:.1f} J/m' if good else 'not used in energy averages'
                    print(f'[ENERGY] {material} speed={speed:g} repeat={repeat}: {reason}; {value}',flush=True)
                    if stopped:break
    except Exception as error:
        failed=True
        # Kit's final shutdown can terminate the process before Python prints
        # an unhandled exception. Save and flush diagnostics BEFORE app.close.
        failure=dict(phase=phase,error=repr(error),traceback=traceback.format_exc())
        (output/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
        print(f'[ENERGY ERROR] {phase}: {error}',file=sys.stderr,flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        print('[ENERGY ERROR] Details saved in '+str(output/'failure.json'),flush=True)
        raise
    finally:
        for stream in streams:stream.close()
        try:
            if count:
                write_report(output)
                expected=2*len(args.speeds)*args.repeats
                status='failed' if failed else 'interrupted' if stopped else 'completed' if count==expected else 'incomplete'
                (output/'run_status.json').write_text(json.dumps(dict(status=status,trials_recorded=count,
                    trials_expected=expected,data_files_closed=True),indent=2)+'\n')
                print(f'[ENERGY] FINISHED: {count}/{expected} attempts recorded; status={status}. '
                      'Data saved. Closing Isaac Sim...',flush=True)
        finally:
            # No sensors/Replicator jobs or unsaved USD edits. All result files
            # are already closed; bypass the stage/Replicator teardown that can
            # hang in GUI mode after the final trial.
            if app is not None:app.close(wait_for_replicator=False,skip_cleanup=True)


if __name__=='__main__':main()
