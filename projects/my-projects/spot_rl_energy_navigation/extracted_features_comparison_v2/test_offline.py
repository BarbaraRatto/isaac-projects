"""Fixtures only: no SimulationApp, controller/model loading or real dataset."""
import ast
import copy
import json
from pathlib import Path
import sys
import unittest
sys.dont_write_bytecode = True
import numpy as np
from common import HERE, LABELS, camera_optics, local_path, read_config, verify_local_assets
from projection import project_patches, aggregate_cells, front_ids
from routes import acquisition_plan
from terrain_labels import decode_semantic_labels


class OfflineChecks(unittest.TestCase):
    def test_self_contained_payload_and_route_coverage(self):
        manifest = verify_local_assets()
        self.assertEqual(len(LABELS), 7)
        self.assertEqual(len(manifest["stations"]), 14)
        for label in LABELS:
            self.assertEqual({s["variant"] for s in manifest["stations"] if s["label"] == label}, {0, 1})
        for name in manifest["files"]:
            self.assertFalse(local_path(name).is_symlink())
        config = read_config()
        pilot = acquisition_plan(config, manifest, "pilot")
        full = acquisition_plan(config, manifest, "full")
        self.assertEqual(len(pilot), 35)
        self.assertEqual(len(full), 196)
        self.assertEqual({t["target_label"] for t in pilot}, set(LABELS))
        self.assertEqual(len({t["pose_group"] for t in pilot}), 5)
        for task in full:
            # Starting poses are on access corridors, outside all terrain tiles.
            x, y = task["start_xy"]
            self.assertFalse(any(s["bounds_xy"][0] < x < s["bounds_xy"][1]
                                 and s["bounds_xy"][2] < y < s["bounds_xy"][3] for s in manifest["stations"]))
        with self.assertRaises(ValueError):
            local_path("../extracted_features_comparison/config.json")

    def test_no_project_import_dependencies(self):
        for file in HERE.glob("*.py"):
            for node in ast.walk(ast.parse(file.read_text())):
                names = [node.module or ""] if isinstance(node, ast.ImportFrom) else (
                    [n.name for n in node.names] if isinstance(node, ast.Import) else [])
                self.assertFalse(any(n.startswith(("multi_terrain_", "terrain_generator", "extracted_features_comparison")) for n in names))

    def test_semantic_unknown_robot_and_ambiguous_labels_excluded(self):
        ids = np.array([[0, 1, 2, 3]])
        info = {"idToLabels": {"0": {"class": "BACKGROUND"}, "1": {"class": "ice"},
                               "2": {"class": "Spot"}, "3": {"class": ["ice", "rocks"]}}}
        np.testing.assert_array_equal(decode_semantic_labels(ids, info), [[-1, 1, -1, -1]])

    def test_high_surface_and_mixed_patch_projection(self):
        config = copy.deepcopy(read_config())
        config["dino"]["input_size"] = 70
        manifest = {"stations": [{"label": "asphalt", "bounds_xy": [-1, 1, -1, 1]}]}
        record = {"intrinsics": [[100, 0, 50], [0, 100, 50], [0, 0, 1]],
                  "camera_quaternion_ros_wxyz": [0, 1, 0, 0], "camera_position_w": [0, 0, 2]}
        depth = np.ones((100, 100), dtype=np.float32)
        semantic = np.zeros((100, 100), dtype=np.int8)
        points, labels, pure, instances, valid = project_patches(depth, semantic, record, config, manifest)
        self.assertTrue(valid.all())
        self.assertTrue((pure == 0).all())
        np.testing.assert_allclose(points[:, 2], 1.0)
        # Elevated surfaces survive semantic analysis, while the old Z cutoff would reject them.
        self.assertTrue((points[:, 2] > config["projection"]["rl_z_max_m"]).all())
        semantic[21, 21] = 1
        _, _, pure, _, _ = project_patches(depth, semantic, record, config, manifest)
        self.assertEqual(pure[6], -1)
        semantic[:] = -1
        self.assertFalse(project_patches(depth, semantic, record, config, manifest)[-1].any())

    def test_cell_average_mixed_purity_and_legacy_height(self):
        config = read_config()["projection"]
        features = np.stack([np.arange(384)-100, np.arange(384)+100]).astype(np.float32)
        points = np.array([[.1, .1, 1.], [.2, .2, 1.]])
        record = {"robot_root_state_w": [-.5, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
        cells = aggregate_cells(features, points, np.array([0, 0]), np.array([0, 0]), np.array([True, True]), record, config)
        np.testing.assert_array_equal(cells["vectors"], np.arange(384).reshape(1, 384))
        self.assertEqual(len(cells["rl_vectors"]), 0)
        self.assertTrue(cells["front"].any())
        mixed = aggregate_cells(features, points, np.array([0, 1]), np.array([0, 1]), np.array([True, True]), record, config)
        self.assertEqual(len(mixed["vectors"]), 0)

    def test_probe_holds_out_whole_frames_and_detects_leakage(self):
        from compare import linear_probe
        xs, ys, gs, splits, frames, episodes = [], [], [], [], [], []
        for partition in ("train", "test"):
            for group in range(3):
                for label_id, label in enumerate(LABELS):
                    vector = np.zeros(384)
                    vector[label_id] = 1
                    xs.append(vector)
                    ys.append(label)
                    gs.append(f"pose{group}")
                    splits.append(partition)
                    frames.append(f"{partition}_{group}_{label}")
                    episodes.append(f"{partition}_{group}_{label}")
        settings = {"test_fraction": .3, "bootstrap_repeats": 5}
        args = [np.asarray(v) for v in (xs, ys, gs, splits, frames, episodes)]
        result = linear_probe(*args, settings, 23)
        self.assertEqual(result["balanced_accuracy"], 1.)
        # Reuse one source frame on both sides; this must be rejected.
        args[4][-1] = args[4][0]
        with self.assertRaisesRegex(ValueError, "leakage"):
            linear_probe(*args, settings, 23)

    def test_usd_optics_and_local_board(self):
        try:
            from pxr import Usd, UsdGeom, Sdf
        except ImportError:
            self.skipTest("Use Isaac standalone Python for USD checks")
        stage = Usd.Stage.Open(str(local_path("assets/terrains/board.usdc")))
        manifest = verify_local_assets()
        for station in manifest["stations"]:
            bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
                stage.GetPrimAtPath(station["prim_path"])).ComputeAlignedRange()
            np.testing.assert_allclose([bounds.GetMin()[0], bounds.GetMax()[0], bounds.GetMin()[1], bounds.GetMax()[1]], station["bounds_xy"], atol=1e-4)
        floor = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
            stage.GetPrimAtPath("/World/AccessFloor")).ComputeAlignedRange()
        self.assertAlmostEqual(floor.GetMax()[2], -.03, places=5)
        # Keep the stage alive while setting the USD camera attribute.
        memory = Usd.Stage.CreateInMemory()
        camera = UsdGeom.Camera.Define(memory, "/Camera")
        camera.GetClippingRangeAttr().Set(camera_optics(read_config())["clipping_range"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
