"""Deterministic observation poses and short walking episodes; pure Python."""
import math


def acquisition_plan(config, manifest, mode):
    tasks = []
    for station in manifest["stations"]:
        if mode == "pilot" and station["variant"] != 0:
            continue
        for kind in ("stance", "walking"):
            section = config[kind]
            angles = section["pilot_azimuths_deg" if mode == "pilot" else "azimuths_deg"]
            distances = (section["pilot_distances_m" if mode == "pilot" else "distances_m"]
                         if kind == "stance" else [section["start_distance_m"]])
            for distance in distances:
                for angle in angles:
                    yaw = math.radians(angle)
                    center = station["center_xy"]
                    tasks.append({
                        "station": station["name"], "target_label": station["label"], "variant": station["variant"],
                        "kind": kind, "yaw_rad": yaw,
                        "start_xy": [center[0]-distance*math.cos(yaw), center[1]-distance*math.sin(yaw)],
                        "pose_group": f"{kind}_d{distance}_a{angle}",
                        "episode_id": f"{station['name']}_{kind}_d{distance}_a{angle}",
                    })
    return tasks
