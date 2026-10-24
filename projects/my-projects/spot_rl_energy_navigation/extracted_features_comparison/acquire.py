"""Capture RGB-D only; never train RL. This is the ONLY simulator entry point."""
from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
from common import HERE, local_path, read_config, verify_local_assets, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--output", help="new directory inside this folder; default data/raw/<timestamp>")
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    config = read_config(args.config)
    manifest = verify_local_assets()
    output = local_path(args.output or f"data/raw/{datetime.now():%Y%m%d_%H%M%S_%f}")
    if output.exists():
        parser.error(f"Refusing to overwrite {output}")
    args.enable_cameras = True
    launcher = AppLauncher(args)
    app = launcher.app
    try:
        import cv2
        import numpy as np
        import torch
        from scene import build_scene, quaternion_from_euler, select_terrain_and_light

        print("[CAPTURE] Creating scene and RGB-D camera...", flush=True)
        sim, camera, paths = build_scene(config, manifest, args.device)
        print("[CAPTURE] Camera initialized; starting acquisition.", flush=True)
        output.mkdir(parents=True)
        write_json(output / "config.json", config)
        write_json(output / "assets_manifest.json", manifest)
        variants = list(enumerate(manifest["terrains"]))
        if args.mode == "pilot":
            seen = set()
            variants = [(i, v) for i, v in variants if not (v["label"] in seen or seen.add(v["label"]))]
        lights = config["lights"][:1] if args.mode == "pilot" else config["lights"]
        azimuths = config["views"]["pilot_azimuths_deg" if args.mode == "pilot" else "azimuths_deg"]
        index = 0
        with (output / "frames.jsonl").open("w") as records:
            import json
            for selected, variant in variants:
                for lighting in lights:
                    select_terrain_and_light(sim, paths, selected, lighting)
                    for distance in config["views"]["distances_m"]:
                        for azimuth in azimuths:
                            if not app.is_running():
                                raise KeyboardInterrupt
                            yaw = math.radians(azimuth)
                            position = [-distance*math.cos(yaw), -distance*math.sin(yaw), config["camera"]["height_m"]]
                            orientation = quaternion_from_euler(0, config["camera"]["pitch_down_deg"], azimuth)
                            camera.set_world_poses(
                                positions=torch.tensor([position], device=sim.device),
                                orientations=torch.tensor([orientation], device=sim.device),
                                convention="world",
                            )
                            # Render without advancing terrain physics. Force refreshed sensor buffers.
                            for _ in range(config["camera"]["settle_render_frames"]):
                                sim.render()
                                camera.update(1/60, force_recompute=True)
                            data = camera.data
                            frame_id = f"frame_{index:05d}"
                            rgb = data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
                            depth = data.output["distance_to_image_plane"][0].cpu().numpy().squeeze()
                            if not cv2.imwrite(str(output / f"{frame_id}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                                raise IOError("Could not write RGB")
                            np.save(output / f"{frame_id}_depth.npy", depth.astype(np.float32), allow_pickle=False)
                            record = {
                                "frame_id": frame_id, "label": variant["label"], "variant": variant["name"],
                                "rgb": f"{frame_id}.png", "depth": f"{frame_id}_depth.npy",
                                "pose_group": f"d{distance}_a{azimuth}", "lighting": lighting,
                                "distance_to_tile_center_xy_m": distance, "yaw_deg": azimuth,
                                "camera_position_w": data.pos_w[0].cpu().tolist(),
                                "camera_quaternion_ros_wxyz": data.quat_w_ros[0].cpu().tolist(),
                                "intrinsics": data.intrinsic_matrices[0].cpu().tolist(),
                                "terrain_bounds_xy": [-1.0, 1.0, -1.0, 1.0],
                            }
                            records.write(json.dumps(record) + "\n")
                            records.flush()
                            index += 1
                            print(f"[CAPTURE] {index}: {variant['name']} {record['pose_group']} {lighting['name']}", flush=True)
        write_json(output / "complete.json", {"frames": index, "mode": args.mode})
        print(f"Dataset: {output}", flush=True)
        print("[CAPTURE] All frames saved; closing Isaac Sim.", flush=True)
    except Exception:
        # SimulationApp shutdown may terminate Python before exception propagation.
        # Print/save the traceback BEFORE close(), so the actual cause is visible.
        detail = traceback.format_exc()
        print(f"[CAPTURE-ERROR]\n{detail}", file=sys.stderr, flush=True)
        diagnostic = HERE / "logs" / f"acquire_error_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
        try:
            write_json(diagnostic, {"output": str(output), "mode": args.mode, "traceback": detail})
            print(f"[CAPTURE-ERROR] Diagnostic saved: {diagnostic}", file=sys.stderr, flush=True)
        except OSError as error:
            print(f"[CAPTURE-ERROR] Could not save diagnostic: {error}", file=sys.stderr, flush=True)
        raise
    finally:
        # RGB/depth and metadata are written synchronously above. No Replicator
        # writer/job was started; its default wait can hang with Camera annotators.
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
