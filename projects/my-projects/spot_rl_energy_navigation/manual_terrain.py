"""Keyboard driving with the frozen Spot locomotion controller; no navigation policy."""
import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--terrain', type=Path, default=Path(__file__).resolve().parents[1] /
                    'terrain_generator/manual_asphalt_rocks/rocks_right.usd')
parser.add_argument('--spawn', type=float, nargs=3, default=(0., 0., .75))
parser.add_argument('--speed', type=float, default=.9)
parser.add_argument('--yaw-speed', type=float, default=1.)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.headless:
    parser.error('Keyboard driving requires a graphical window; omit --headless.')
if not args.terrain.is_file():
    parser.error(f'Missing terrain: {args.terrain}. Run create_manual_asphalt_rocks.py first.')
if not 0 < args.speed <= 1.6 or not 0 < args.yaw_speed <= 1.5:
    parser.error('Use 0 < speed <= 1.6 and 0 < yaw-speed <= 1.5.')
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import carb
import omni.appwindow
import torch
import isaaclab.sim as sim_utils
from isaacsim.core.utils.prims import create_prim
from locomotion import create_spot, reset_spot, SpotVelocityController
from rl_config import PHYSICS_DT


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args.device))
    create_prim('/World/Terrain', 'Xform', usd_path=str(args.terrain.resolve()))
    light = sim_utils.DistantLightCfg(intensity=3000., angle=1.)
    light.func('/World/defaultLight', light)
    x, y, _ = args.spawn
    sim.set_camera_view((x + 9., y - 9., 10.), (x + 2., y + 2., 0.))
    robot = create_spot()
    controller = SpotVelocityController(robot, sim.device)
    sim.reset()
    reset_spot(robot, tuple(args.spawn))
    controller.reset()
    robot.update(PHYSICS_DT)
    pressed = set()
    requests = set()

    def keyboard_event(event, *unused):
        name = event.input.name
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if name in ('R', 'SPACE', 'ESCAPE'):
                requests.add(name)
                pressed.clear()
            else:
                pressed.add(name)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            pressed.discard(name)
        return True

    inputs = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    subscription = inputs.subscribe_to_keyboard_events(keyboard, keyboard_event)
    measuring = False
    energy = distance = elapsed = 0.
    previous_power = None
    previous_xy = robot.data.root_pos_w[0, :2].clone()
    settle = round(2. / PHYSICS_DT)
    step = 0
    print('[MANUAL] W/S or up/down: forward/back; A/D or left/right: turn. '
          'Release keys: stop. SPACE: start a fresh measurement / stop and print it. '
          'R: respawn. ESC: quit. Click the viewport to focus the keyboard.', flush=True)

    def report():
        per_m = f'{energy / distance:.1f}' if distance > .01 else 'n/a'
        print(f'[MEASURE] t={elapsed:.2f}s E_abs={energy:.2f}J path={distance:.3f}m '
              f'J/m={per_m}', flush=True)

    try:
        while simulation_app.is_running():
            if 'ESCAPE' in requests:
                break
            if sim.is_stopped():
                break
            if not sim.is_playing():
                pressed.clear()
                sim.render()
                continue
            if 'R' in requests:
                if measuring:
                    report()
                measuring = False
                reset_spot(robot, tuple(args.spawn))
                controller.reset()
                robot.update(PHYSICS_DT)
                settle = round(2. / PHYSICS_DT)
            if 'SPACE' in requests and settle == 0:
                if measuring:
                    report()
                    measuring = False
                else:
                    energy = distance = elapsed = 0.
                    previous_power = None
                    previous_xy = robot.data.root_pos_w[0, :2].clone()
                    measuring = True
                    print('[MEASURE] Started (includes standing time until SPACE is pressed again).', flush=True)
            requests.clear()
            vx = args.speed * (int(bool(pressed & {'W', 'UP'})) - int(bool(pressed & {'S', 'DOWN'})))
            wz = args.yaw_speed * (int(bool(pressed & {'A', 'LEFT'})) - int(bool(pressed & {'D', 'RIGHT'})))
            if settle:
                vx = wz = 0.
                settle -= 1
                if settle == 0:
                    print('[MANUAL] Ready.', flush=True)
            command = torch.tensor([[vx, 0., wz]], device=sim.device)
            controller.forward(command)
            sim.step(render=(step % 4 == 0))
            robot.update(PHYSICS_DT)
            if measuring:
                torque = robot.root_physx_view.get_dof_projected_joint_forces()
                power = torch.sum(torch.abs(torque * robot.data.joint_vel)).item()
                xy = robot.data.root_pos_w[0, :2].clone()
                if previous_power is not None:
                    energy += .5 * (previous_power + power) * PHYSICS_DT
                    distance += torch.linalg.vector_norm(xy - previous_xy).item()
                    elapsed += PHYSICS_DT
                previous_power, previous_xy = power, xy
                if step % 200 == 0:
                    report()
            step += 1
    finally:
        if measuring:
            report()
        inputs.unsubscribe_to_keyboard_events(keyboard, subscription)


if __name__ == '__main__':
    try:
        main()
    finally:
        simulation_app.close()
