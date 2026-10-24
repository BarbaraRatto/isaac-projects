"""Technical 32-robot simulator probe; never trains or saves a policy."""

import time
import numpy as np
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from multi_terrain_test6.env import Test6Env
from multi_terrain_test6.config import VISUAL_SIZE
from multi_terrain_test6.utils import stop_actions

def probe(env: Test6Env, steps: int) -> None:
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    for condition in env.conditions:
        base = f"/World/envs/env_{condition.env_id}"
        for path in (f"{base}/Warehouse", f"{base}/Spot",
                     f"{base}/NavigationZEDCamera"):
            if not stage.GetPrimAtPath(path).IsValid():
                raise RuntimeError(f"Missing prim in {condition.name}: {path}")
        grid_path = f"{base}/Warehouse/Grid"
        actual = {child.GetName() for child in stage.GetPrimAtPath(grid_path).GetChildren()}
        expected = {f"Cell_{row}_{col}_{terrain}"
                    for row, cells in enumerate(condition.layout.grid)
                    for col, terrain in enumerate(cells)}
        if actual != expected:
            raise RuntimeError(f"Wrong terrain in {condition.name}: {actual ^ expected}")
    wrapped = Sb3VecEnvWrapper(env, fast_variant=False)
    observation = wrapped.reset()
    print(f"[PROBE] verified_conditions={len(env.conditions)} "
          f"envs={wrapped.num_envs} observation={observation.shape} "
          f"camera_rgb={tuple(env.camera.data.output['rgb'].shape)} "
          f"camera_depth={tuple(env.camera.data.output['distance_to_image_plane'].shape)} "
          f"valid_visual_cells={env.visual_valid.tolist()}", flush=True)
    if wrapped.action_space.shape != (2,):
        raise RuntimeError("Test6 must expose exactly two navigation actions")
    if observation.shape != (env.num_envs, 11 + VISUAL_SIZE) or not np.isfinite(observation).all():
        raise RuntimeError("Invalid binary-choice DINO observation")
    if not bool((env.visual_valid > 0).all()):
        raise RuntimeError("At least one Spot has no valid visible terrain sample")
    started = time.perf_counter()
    for i in range(steps):
        action = stop_actions(env.num_envs)
        if 0 < i < steps - 1:
            action[:, 0] = -0.6
            action[:, 1] = 0.2 if i % 2 else -0.2
        obs, rewards, dones, _ = wrapped.step(action)
        print(f"[PROBE] action={i + 1}/{steps} reward={rewards.round(3).tolist()} "
              f"done={dones.tolist()} visual_valid={env.visual_valid.tolist()}", flush=True)
        if not bool((env.command[:, 1] == 0).all()) or not bool((env.command[:, 0] >= 0).all()):
            raise RuntimeError("Forbidden lateral or reverse navigation command")
        if not np.isfinite(obs).all():
            raise RuntimeError("Non-finite observation during probe")
    elapsed = time.perf_counter() - started
    print(f"[PROBE] completed={steps} vector_actions, {steps * env.num_envs} "
          f"robot_transitions in {elapsed:.1f}s; no training was run", flush=True)

