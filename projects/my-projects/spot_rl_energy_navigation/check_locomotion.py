"""One-robot Isaac Lab check. This script does not train an RL policy."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Check Spot locomotion and joint measurements in Isaac Lab.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402

from rl_config import (  # noqa: E402
    PHYSICS_DT, STAND_SECONDS, STOP_SECONDS, WALK_COMMAND, WALK_SECONDS, validate_assets,
)
from locomotion import SpotVelocityController, create_spot, reset_spot  # noqa: E402
from scene import build_scene  # noqa: E402


def command_at(time_s: float, device: str) -> torch.Tensor:
    if STAND_SECONDS <= time_s < STAND_SECONDS + WALK_SECONDS:
        return torch.tensor([WALK_COMMAND], device=device, dtype=torch.float32)
    return torch.zeros((1, 3), device=device)


def main() -> None:
    validate_assets()
    print("[CHECK] Creating Isaac Lab simulation", flush=True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args.device))
    sim.set_camera_view((3.0, 12.0, 4.0), (0.0, 9.0, 0.0))
    print("[CHECK] Loading terrain USD", flush=True)
    build_scene()
    print("[CHECK] Creating Spot and loading controller", flush=True)
    robot = create_spot()
    controller = SpotVelocityController(robot, sim.device)
    print("[CHECK] Starting physics", flush=True)
    sim.reset()
    reset_spot(robot)
    robot.update(PHYSICS_DT)

    total_steps = round((STAND_SECONDS + WALK_SECONDS + STOP_SECONDS) / PHYSICS_DT)
    measured_steps = 0
    energy_j = 0.0
    start_position = robot.data.root_pos_w[0].detach().cpu().numpy().copy()
    print(f"[CHECK] Starting at {start_position.tolist()}; steps={total_steps}, dt={PHYSICS_DT}", flush=True)

    for step in range(total_steps):
        if not simulation_app.is_running():
            print("[CHECK] Simulation closed before the check finished.", flush=True)
            return
        command = command_at(step * PHYSICS_DT, sim.device)
        controller.forward(command)
        sim.step(render=not args.headless)
        robot.update(PHYSICS_DT)

        efforts = robot.data.applied_torque
        velocities = robot.data.joint_vel
        if efforts is not None and velocities is not None:
            if efforts.shape != velocities.shape or not torch.isfinite(efforts).all():
                raise RuntimeError("Invalid joint efforts or velocity measurements.")
            energy_j += torch.sum(torch.abs(efforts * velocities)).item() * PHYSICS_DT
            measured_steps += 1

        if (step + 1) % round(1.0 / PHYSICS_DT) == 0 or step + 1 == total_steps:
            position = robot.data.root_pos_w[0].detach().cpu().numpy()
            print(
                f"[CHECK] t={(step + 1) * PHYSICS_DT:.1f}s "
                f"command={command[0].tolist()} xyz={position.round(3).tolist()} "
                f"joint_samples={measured_steps} energy={energy_j:.2f} J",
                flush=True,
            )

    end_position = robot.data.root_pos_w[0].detach().cpu().numpy()
    displacement = float(np.linalg.norm(end_position[:2] - start_position[:2]))
    if measured_steps == 0:
        raise RuntimeError("No joint efforts were available; energy cannot be estimated.")
    print(
        f"[CHECK] Finished: horizontal displacement={displacement:.3f} m, "
        f"final height={end_position[2]:.3f} m, energy={energy_j:.2f} J.",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(skip_cleanup=args.headless)
