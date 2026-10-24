"""Paired forced-route benchmark on the NEW manual maps, no RL/DINO training."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def guided_command(xy, yaw, target, speed):
    error = math.atan2(math.sin(math.atan2(target[1]-xy[1], target[0]-xy[0])-yaw),
                       math.cos(math.atan2(target[1]-xy[1], target[0]-xy[0])-yaw))
    vx = speed*max(0., math.cos(error)) if abs(error) < .6 else 0.
    return vx, max(-1.5, min(1.5, 2.*error))


def branch(xy):
    col, row = (math.floor((float(v)+1)/2) for v in xy)
    if (row == 0 and col in (1, 2)) or (row == 1 and col == 2):
        return 'right'
    if (col == 0 and row in (1, 2)) or (row == 2 and col == 1):
        return 'left'
    if row == 1 and col == 1:
        return 'ice'
    return 'other'


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--speed', type=float, default=.8)
    parser.add_argument('--seed', type=int, default=230023)
    parser.add_argument('--output', type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.repeats < 1 or not 0 < args.speed <= 1.6:
        parser.error('Positive repeats and speed in (0,1.6] required')
    output = args.output or HERE/'results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    source = HERE.parents[1]/'terrain_generator/manual_asphalt_rocks_10cm'
    hashes = {}
    for side in ('right', 'left'):
        p = source/f'rocks_{side}.usd'
        shutil.copy2(p, output/p.name)
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    shutil.copy2(HERE/'compare.py', output/'benchmark_source.py')
    shutil.copy2(HERE.parents[1]/'terrain_generator/create_real_terrains_rocks_10cm.py', output/'generator_source.py')
    (output/'config.json').write_text(json.dumps(dict(speed=args.speed, repeats=args.repeats,
        seed=args.seed, source=str(source), sha256=hashes, dt=.005, action_dt=.2,
        gamma=.999, goal_radius=.55, deadline=40., initial_yaw=math.pi/4,
        waypoint_radius=.25, settle_s=2., energy='PhysX sum(abs(torque*joint_speed)), trapezoid',
        note='Geometry fixed; matched initial XY jitter +/-5 cm; same route on both maps.'), indent=2))
    app = AppLauncher(args).app
    try:
        import numpy as np
        import torch
        import carb
        import isaaclab.sim as sim_utils
        from isaacsim.core.utils.prims import create_prim
        from omni.physx import get_physx_scene_query_interface
        from locomotion import create_spot, reset_spot, SpotVelocityController
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=.005, device=args.device, enable_scene_query_support=True))
        for index, side in enumerate(('right', 'left')):
            create_prim(f'/World/Map{index}', 'Xform', usd_path=str((output/f'rocks_{side}.usd').resolve()),
                        translation=(20.*index, 0., 0.))
        light = sim_utils.DistantLightCfg(intensity=3000., angle=1.)
        light.func('/World/Light', light)
        robot = create_spot()
        controller = SpotVelocityController(robot, sim.device)
        sim.reset()
        query = get_physx_scene_query_interface()
        zero = torch.zeros((1, 3), device=sim.device)
        def step(command):
            if not app.is_running():
                raise KeyboardInterrupt
            controller.forward(command)
            sim.step(render=not args.headless)
            robot.update(.005)
        def power():
            value = float((robot.root_physx_view.get_dof_projected_joint_forces()*robot.data.joint_vel).abs().sum())
            if not math.isfinite(value):
                raise RuntimeError('Non-finite power')
            return value
        def ground_height(index):
            pos = robot.data.root_pos_w[0].cpu().numpy()
            hits = []
            def hit_callback(hit):
                if str(hit.collision).startswith(f'/World/Map{index}/'):
                    hits.append(float(hit.position[2]))
                return True
            query.raycast_all(carb.Float3(float(pos[0]), float(pos[1]), float(pos[2]+.05)),
                              carb.Float3(0, 0, -1), 5., hit_callback)
            return max(hits) if hits else float('-inf')
        fields = ['trial', 'map', 'route', 'material', 'speed', 'success', 'reason', 'route_followed',
                  'energy_j', 'discounted_energy_j', 'nav_return', 'discounted_nav_return',
                  'path_m', 'seconds', 'final_distance_m', 'right_m', 'left_m', 'ice_m',
                  'start_x', 'start_y', 'settle_speed', 'waypoint_index']
        detail_fields = ['trial', 'map', 'route', 'step', 'x', 'y', 'vx', 'wz', 'energy_j',
                         'distance_m', 'nav_reward']
        rng = np.random.default_rng(args.seed)
        rows = []
        with (output/'episodes.csv').open('w', buffering=1) as ef, (output/'steps.csv').open('w', buffering=1) as sf:
            log = csv.DictWriter(ef, fieldnames=fields); log.writeheader()
            details = csv.DictWriter(sf, fieldnames=detail_fields); details.writeheader()
            for trial in range(args.repeats):
                jitter = rng.uniform(-.05, .05, 2)
                order = [(i, side) for i in (0, 1) for side in ('right', 'left')]
                rng.shuffle(order)
                for index, route in order:
                    offset = np.array([20.*index, 0.])
                    reset_spot(robot, (offset[0]+jitter[0], jitter[1], .75))
                    root = robot.data.default_root_state.clone()
                    root[0, :3] = torch.tensor([offset[0]+jitter[0], jitter[1], .75], device=sim.device)
                    root[0, 3:7] = torch.tensor([math.cos(math.pi/8), 0., 0., math.sin(math.pi/8)], device=sim.device)
                    root[0, 7:] = 0
                    robot.write_root_state_to_sim(root)
                    robot.reset(); controller.reset(); robot.update(.005)
                    if not args.headless:
                        sim.set_camera_view((offset[0]+8., 9., 11.), (offset[0]+2., 2., 0.))
                    for _ in range(400): step(zero)
                    settle_speed = float(robot.data.root_lin_vel_b[0, :2].norm())
                    previous = robot.data.root_pos_w[0, :2].cpu().numpy().copy()-offset
                    start = previous.copy()
                    distance = float(np.linalg.norm(previous-[4.,4.]))
                    waypoints = [(2.,0.),(4.,0.),(4.,2.),(4.,4.)] if route == 'right' else [(0.,2.),(0.,4.),(2.,4.),(4.,4.)]
                    waypoint = 0
                    energy = discounted_energy = nav_total = discounted_nav = path = 0.
                    route_m = dict(right=0., left=0., ice=0., other=0.)
                    reason = 'timeout'; success = False
                    prev_power = power()
                    for t in range(200):
                        xy = robot.data.root_pos_w[0, :2].cpu().numpy()-offset
                        if waypoint < len(waypoints)-1 and np.linalg.norm(xy-waypoints[waypoint]) < .25:
                            waypoint += 1
                        q = robot.data.root_quat_w[0].cpu().numpy()
                        yaw = math.atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2))
                        vx,wz = guided_command(xy,yaw,waypoints[waypoint],args.speed)
                        command = torch.tensor([[vx,0.,wz]], device=sim.device)
                        step_energy = 0.
                        for _ in range(40):
                            step(command)
                            current_power = power()
                            step_energy += .5*(prev_power+current_power)*.005
                            prev_power = current_power
                        current = robot.data.root_pos_w[0, :2].cpu().numpy().copy()-offset
                        segment = float(np.linalg.norm(current-previous))
                        path += segment
                        route_m[branch((current+previous)/2)] += segment
                        previous = current
                        new_distance = float(np.linalg.norm(current-[4.,4.]))
                        height = ground_height(index)
                        outside = bool((current < -.8).any() or (current > 4.8).any())
                        fallen = float(robot.data.projected_gravity_b[0,2]) > -.5 or float(robot.data.root_pos_w[0,2])-height < .25 or not math.isfinite(height)
                        success = new_distance <= .55 and not outside and not fallen
                        failure = outside or fallen or (t == 199 and not success)
                        nav = 3*(distance-new_distance)-.02+100*success-100*failure
                        distance = new_distance
                        energy += step_energy; discounted_energy += .999**t*step_energy
                        nav_total += nav; discounted_nav += .999**t*nav
                        details.writerow(dict(trial=trial+1,map=index,route=route,step=t+1,x=current[0],y=current[1],
                                              vx=vx,wz=wz,energy_j=step_energy,distance_m=distance,nav_reward=nav))
                        if success or failure:
                            reason = 'goal' if success else ('out_of_bounds' if outside else ('fallen' if fallen else 'timeout'))
                            break
                    opposite = 'left' if route == 'right' else 'right'
                    followed = waypoint == 3 and route_m[route] > 3. and route_m[opposite] < .15 and route_m['ice'] < .15
                    material = 'rocks' if route == ('right' if index == 0 else 'left') else 'asphalt'
                    record = dict(trial=trial+1,map=index,route=route,material=material,speed=args.speed,
                        success=int(success),reason=reason,route_followed=int(followed),energy_j=energy,
                        discounted_energy_j=discounted_energy,nav_return=nav_total,discounted_nav_return=discounted_nav,
                        path_m=path,seconds=(t+1)*.2,final_distance_m=distance,right_m=route_m['right'],
                        left_m=route_m['left'],ice_m=route_m['ice'],start_x=start[0],start_y=start[1],
                        settle_speed=settle_speed,waypoint_index=waypoint)
                    rows.append(record); log.writerow(record)
                    print(f'[BENCH] trial={trial+1} map={index} route={route} material={material} '
                          f'{reason} followed={followed} E={energy:.1f}J path={path:.2f}m t={(t+1)*.2:.1f}s', flush=True)
        (output/'complete.json').write_text(json.dumps(dict(episodes=len(rows),output=str(output)),indent=2))
        print(f'[BENCH] Finished: {output}', flush=True)
    finally:
        app.close(skip_cleanup=args.headless)


if __name__ == '__main__':
    main()
