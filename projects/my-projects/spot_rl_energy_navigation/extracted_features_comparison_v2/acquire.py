"""Record actual Spot camera frames on seven terrains. ONLY simulator entry point."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import math
import sys
import traceback
sys.dont_write_bytecode = True
from common import HERE, LABELS, local_path, read_config, verify_local_assets, write_json
from routes import acquisition_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--output", help="new local output directory")
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    config, manifest = read_config(args.config), verify_local_assets()
    output = local_path(args.output or f"data/raw/{datetime.now():%Y%m%d_%H%M%S_%f}")
    if output.exists():
        parser.error(f"Refusing to overwrite {output}")
    tasks = acquisition_plan(config, manifest, args.mode)
    args.enable_cameras = True
    launcher = AppLauncher(args)
    app = launcher.app
    try:
        import cv2
        import numpy as np
        import torch
        from locomotion import SpotController
        from scene import build_scene, sync_camera
        from terrain_labels import decode_semantic_labels

        print("[V2] Creating board, Spot and RGB-D/semantic camera...", flush=True)
        sim, robot, camera = build_scene(config, manifest, args.device)
        controller = SpotController(robot, sim.device)
        dt = config["physics"]["dt"]
        zero = torch.zeros((1, 3), device=sim.device)
        output.mkdir(parents=True)
        write_json(output / "config.json", config)
        write_json(output / "assets_manifest.json", manifest)
        write_json(output / "plan.json", tasks)
        count, failures = 0, []
        coverage = {label: 0 for label in LABELS}

        def upright():
            return (float(robot.data.root_pos_w[0, 2]) >= config["physics"]["minimum_body_height_m"]
                    and float(robot.data.projected_gravity_b[0, 2]) < -.5)

        def advance(command, steps):
            for _ in range(steps):
                if not app.is_running():
                    raise KeyboardInterrupt
                controller.apply(command)
                robot.write_data_to_sim()
                sim.step(render=False)
                robot.update(dt)
                if not upright():
                    return False
            return True

        def reset(task):
            state = robot.data.default_root_state.clone()
            state[0, :3] = torch.tensor([*task["start_xy"], config["physics"]["spawn_height_m"]], device=sim.device)
            half = task["yaw_rad"]/2
            state[0, 3:7] = torch.tensor([math.cos(half), 0., 0., math.sin(half)], device=sim.device)
            state[:, 7:] = 0.
            robot.write_root_state_to_sim(state)
            robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
            robot.reset()
            controller.reset()
            # First integrate a few milliseconds before checking settled body height.
            return advance(zero, round(config["physics"]["settle_seconds"]/dt))

        with (output / "frames.jsonl").open("w") as stream:
            def capture(task, elapsed):
                nonlocal count
                # Physics remains active during the episode; these repeated renders
                # warm up annotators at one measured robot pose without extra time steps.
                sync_camera(robot, camera, config)
                for _ in range(config["camera"]["settle_render_frames"]):
                    sim.render()
                    camera.update(dt, force_recompute=True)
                data = camera.data
                rgb = data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
                depth = data.output["distance_to_image_plane"][0].cpu().numpy().squeeze().astype(np.float32)
                semantic_ids = data.output["semantic_segmentation"][0].cpu().numpy().squeeze()
                info = data.info[0]["semantic_segmentation"]
                labels = decode_semantic_labels(semantic_ids, info)
                if not (labels >= 0).any():
                    raise RuntimeError("No labelled terrain pixels: inspect semantic annotator/labels before continuing")
                frame_id = f"frame_{count:05d}"
                if not cv2.imwrite(str(output/f"{frame_id}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                    raise IOError("Could not save RGB")
                np.save(output/f"{frame_id}_depth.npy", depth, allow_pickle=False)
                np.save(output/f"{frame_id}_labels.npy", labels, allow_pickle=False)
                for i, label in enumerate(LABELS):
                    coverage[label] += int((labels == i).any())
                root = robot.data.root_state_w[0].cpu().tolist()
                row = {**task, "frame_id": frame_id, "rgb": f"{frame_id}.png",
                       "depth": f"{frame_id}_depth.npy", "semantic_labels": f"{frame_id}_labels.npy",
                       "time_in_episode_s": elapsed, "mode": args.mode,
                       "camera_position_w": data.pos_w[0].cpu().tolist(),
                       "camera_quaternion_ros_wxyz": data.quat_w_ros[0].cpu().tolist(),
                       "intrinsics": data.intrinsic_matrices[0].cpu().tolist(),
                       "robot_root_state_w": root}
                stream.write(json.dumps(row)+"\n")
                stream.flush()
                count += 1
                print(f"[V2-CAPTURE] {count}: {task['episode_id']} t={elapsed:.1f}", flush=True)

            for task in tasks:
                if not reset(task):
                    failures.append({"episode": task["episode_id"], "reason": "fallen_during_settle"})
                    continue
                if task["kind"] == "stance":
                    if advance(zero, round(config["stance"]["hold_seconds"]/dt)):
                        capture(task, 0.0)
                    else:
                        failures.append({"episode": task["episode_id"], "reason": "fallen_during_stance"})
                else:
                    steps = round(config["walking"]["duration_seconds"]/dt)
                    period = round(config["walking"]["capture_every_seconds"]/dt)
                    for step in range(steps):
                        q = robot.data.root_quat_w[0]
                        yaw = float(torch.atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2)))
                        error = math.atan2(math.sin(task["yaw_rad"]-yaw), math.cos(task["yaw_rad"]-yaw))
                        rate = max(-config["walking"]["max_yaw_rate_rad_s"], min(config["walking"]["max_yaw_rate_rad_s"], config["walking"]["yaw_gain"]*error))
                        command = torch.tensor([[config["walking"]["forward_speed_m_s"], 0., rate]], device=sim.device)
                        if not advance(command, 1):
                            failures.append({"episode": task["episode_id"], "reason": "fallen_during_walk", "step": step})
                            break
                        if (step+1) % period == 0:
                            capture(task, (step+1)*dt)
        write_json(output/"episodes_failed.json", failures)
        write_json(output/"complete.json", {"frames": count, "mode": args.mode, "failed_episodes": len(failures),
                                           "frames_with_label_pixels": coverage})
        print(f"[V2] Saved {count} frames, failed episodes={len(failures)}. Dataset: {output}", flush=True)
    except Exception:
        detail = traceback.format_exc()
        print(f"[V2-ERROR]\n{detail}", file=sys.stderr, flush=True)
        write_json(HERE/"logs"/f"acquire_error_{datetime.now():%Y%m%d_%H%M%S_%f}.json", {"traceback": detail})
        raise
    finally:
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
