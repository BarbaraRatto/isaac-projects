"""Measure camera and DINOv2 overhead in the single-Spot Isaac Lab loop.

This is a timing benchmark, not PPO training. It uses identical fixed velocity
commands and action timing in all modes. The existing ZED X right camera is mounted on Spot. Cached features are
replayed by index; this does not implement pose-aware lookup for RL training.
"""

import argparse
import csv
import importlib.util
import statistics
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_CSV = RESULTS_DIR / "dino_timing_zed.csv"
DEFAULT_CACHE = Path(__file__).resolve().parent / "runs" / "zed_feature_samples.npy"
ZED_USD = Path("/home/isaac/isaac-sim/assets/Assets/Isaac/5.1/Isaac/Sensors/Stereolabs/ZED_X/ZED_X.usdc")

parser = argparse.ArgumentParser(description="Benchmark DINOv2 in Spot's Isaac Lab action loop.")
parser.add_argument("--mode", choices=("baseline", "camera", "online", "cached"), required=True)
parser.add_argument("--actions", type=int, default=40, help="Measured 0.2 s velocity actions.")
parser.add_argument("--warmup-actions", type=int, default=4)
parser.add_argument("--image-every", type=int, default=1, help="Camera/feature update every N actions.")
parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.actions < 1 or args.warmup_actions < 0 or args.image_every < 1:
    parser.error("--actions and --image-every must be positive; --warmup-actions cannot be negative")
if args.mode == "cached" and not args.cache.is_file():
    parser.error(f"Feature cache is missing: {args.cache}. Run --mode online first.")
# Camera sensors also render in headless mode when this option is enabled.
if args.mode in ("camera", "online", "cached"):
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import omni.usd  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402

from locomotion import SpotVelocityController, create_spot, reset_spot  # noqa: E402
from navigation_config import NAV_CFG  # noqa: E402
from rl_config import PHYSICS_DT, validate_assets  # noqa: E402
from scene import build_scene  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "IsaacRobotics" / "applications"))


FIELDS = (
    "camera_asset", "mode", "actions", "image_every", "image_updates", "simulation_seconds",
    "wall_seconds", "actions_per_wall_second", "model_load_seconds",
    "action_mean_ms", "action_p95_ms", "physics_mean_ms", "physics_p95_ms",
    "camera_mean_ms", "camera_p95_ms", "feature_mean_ms", "feature_p95_ms",
    "gpu_memory_start_mb", "gpu_memory_peak_mb", "feature_shape", "checksum",
)
ACTION_STEPS = NAV_CFG.physics_steps_per_action
BLOCK_ACTIONS = 10
START = (10.5, 0.0, 0.75)  # Same asphalt cell as the step-3 navigation task.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360
CACHE_SAMPLES = 16


def percentile_95(values: list[float]) -> float:
    return float(np.percentile(values, 95)) if values else 0.0


def mean_ms(values: list[float]) -> float:
    return 1000.0 * statistics.mean(values) if values else 0.0


def p95_ms(values: list[float]) -> float:
    return 1000.0 * percentile_95(values)


def gpu_memory_mb() -> float | None:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 * 1024)
    except Exception:
        return None


def create_camera() -> Camera:
    """Mount the real ZED X USD asset and read its right RGB camera."""
    if not ZED_USD.is_file():
        raise FileNotFoundError(f"ZED X USD asset not found: {ZED_USD}")
    from spot_camera import attach_zed_camera

    stage = omni.usd.get_context().get_stage()
    camera_path = attach_zed_camera(stage, str(ZED_USD.parents[4]))
    camera_cfg = CameraCfg(
        prim_path=camera_path,
        update_period=0.0,
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=None,
    )
    return Camera(cfg=camera_cfg)


def main() -> None:
    validate_assets()
    has_camera = args.mode in ("camera", "online", "cached")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args.device))
    build_scene()
    robot = create_spot()
    camera = create_camera() if has_camera else None
    controller = SpotVelocityController(robot, sim.device)
    sim.reset()
    forward = torch.tensor([[0.9, 0.0, 0.0]], device=sim.device)
    zero = torch.zeros((1, 3), device=sim.device)
    cache = np.load(args.cache, mmap_mode="r") if args.mode == "cached" else None
    if cache is not None and (cache.ndim != 4 or cache.shape[1:] != (37, 37, 384)):
        raise RuntimeError(f"Unexpected feature cache shape: {cache.shape}")

    def physics_step(command: torch.Tensor, render: bool = False) -> None:
        if not simulation_app.is_running():
            raise RuntimeError("Isaac Sim closed during DINO benchmark")
        controller.forward(command)
        sim.step(render=render)
        robot.update(PHYSICS_DT)

    def settle() -> None:
        reset_spot(robot, START)
        controller.reset()
        robot.update(PHYSICS_DT)
        if camera is not None:
            camera.reset()
        n_steps = round(NAV_CFG.settle_seconds / PHYSICS_DT)
        for step in range(n_steps):
            physics_step(zero, render=has_camera and step == n_steps - 1)
        if camera is not None:
            camera.update(NAV_CFG.settle_seconds, force_recompute=True)

    def action_step() -> tuple[float, float, np.ndarray | None]:
        start = time.perf_counter()
        for step in range(ACTION_STEPS):
            physics_step(forward, render=has_camera and step == ACTION_STEPS - 1)
        physics_seconds = time.perf_counter() - start
        camera_seconds = 0.0
        if camera is not None:
            start = time.perf_counter()
            camera.update(ACTION_STEPS * PHYSICS_DT, force_recompute=True)
            rgb = camera.data.output["rgb"]
            if rgb.shape != (1, CAMERA_HEIGHT, CAMERA_WIDTH, 3):
                raise RuntimeError(f"Unexpected camera image shape: {rgb.shape}")
            frame = rgb[0].cpu().numpy().copy()
            camera_seconds = time.perf_counter() - start
            return physics_seconds, camera_seconds, frame
        return physics_seconds, camera_seconds, None

    settle()
    model_load_seconds = 0.0
    dino = None
    if args.mode == "online":
        package_root = PROJECT_ROOT / "ros2_ws" / "src" / "spot_terrain_gridmap"
        sys.path.insert(0, str(package_root))
        if importlib.util.find_spec("transformers") is None:
            raise RuntimeError("transformers is unavailable in Isaac Lab's Python environment")
        from spot_terrain_gridmap.dino_feature_extractor import DinoFeatureExtractor
        load_start = time.perf_counter()
        dino = DinoFeatureExtractor(model_name="facebook/dinov2-small", device="cuda", input_size=518)
        model_load_seconds = time.perf_counter() - load_start
        # Warm up the CUDA kernels before timing inference.
        _physics, _camera, frame = action_step()
        for _ in range(2):
            dino.extract(frame)
        settle()

    for step in range(args.warmup_actions):
        if step and step % BLOCK_ACTIONS == 0:
            settle()
        _physics, _camera, frame = action_step()
        if dino is not None and step % args.image_every == 0:
            dino.extract(frame)
        if cache is not None and step % args.image_every == 0:
            _ = float(cache[step % len(cache), 0, 0, 0])
    settle()

    action_times = []
    physics_times = []
    camera_times = []
    feature_times = []
    samples = []
    checksum = 0.0
    updates = 0
    memory_start = gpu_memory_mb()
    memory_peak = memory_start
    wall_start = time.perf_counter()
    for action in range(args.actions):
        action_start = time.perf_counter()
        if action and action % BLOCK_ACTIONS == 0:
            settle()
        physics_seconds, camera_seconds, frame = action_step()
        physics_times.append(physics_seconds)
        if has_camera:
            camera_times.append(camera_seconds)
        if action == 0 and frame is not None:
            from PIL import Image
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame).save(RESULTS_DIR / "dino_sample_rgb.png")
        if (has_camera or cache is not None) and action % args.image_every == 0:
            if dino is not None:
                start = time.perf_counter()
                features = dino.extract(frame)
                feature_times.append(time.perf_counter() - start)
                if features.shape != (37, 37, 384):
                    raise RuntimeError(f"Unexpected DINO feature shape: {features.shape}")
                checksum += float(features[0, 0, 0])
                if len(samples) < CACHE_SAMPLES:
                    samples.append(features)
            elif cache is not None:
                start = time.perf_counter()
                features = np.array(cache[updates % len(cache)], copy=True)
                checksum += float(features[0, 0, 0])
                feature_times.append(time.perf_counter() - start)
            elif frame is not None:
                checksum += float(frame[0, 0, 0])
            updates += 1
        action_times.append(time.perf_counter() - action_start)
        current_memory = gpu_memory_mb()
        if current_memory is not None:
            memory_peak = max(memory_peak or current_memory, current_memory)
        if (action + 1) % 10 == 0 or action + 1 == args.actions:
            print(f"[DINO] {args.mode}: {action + 1}/{args.actions} actions", flush=True)
    wall_seconds = time.perf_counter() - wall_start
    if samples:
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.cache, np.stack(samples, axis=0))
        print(f"[DINO] Feature samples saved: {args.cache}", flush=True)

    row = {
        "camera_asset": "ZED_X/CameraRight" if has_camera else "none",
        "mode": args.mode,
        "actions": args.actions,
        "image_every": args.image_every,
        "image_updates": updates,
        "simulation_seconds": round(args.actions * ACTION_STEPS * PHYSICS_DT, 3),
        "wall_seconds": round(wall_seconds, 3),
        "actions_per_wall_second": round(args.actions / wall_seconds, 3),
        "model_load_seconds": round(model_load_seconds, 3),
        "action_mean_ms": round(mean_ms(action_times), 2),
        "action_p95_ms": round(p95_ms(action_times), 2),
        "physics_mean_ms": round(mean_ms(physics_times), 2),
        "physics_p95_ms": round(p95_ms(physics_times), 2),
        "camera_mean_ms": round(mean_ms(camera_times), 2),
        "camera_p95_ms": round(p95_ms(camera_times), 2),
        "feature_mean_ms": round(mean_ms(feature_times), 2),
        "feature_p95_ms": round(p95_ms(feature_times), 2),
        "gpu_memory_start_mb": round(memory_start, 1) if memory_start is not None else "",
        "gpu_memory_peak_mb": round(memory_peak, 1) if memory_peak is not None else "",
        "feature_shape": "37x37x384" if dino is not None or cache is not None else "",
        "checksum": round(checksum, 4),
    }
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
    print(f"[DINO] Results: {row}", flush=True)
    print(f"[DINO] CSV: {args.csv}", flush=True)
    sim.clear_instance()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(skip_cleanup=args.headless)
