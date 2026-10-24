"""Small numerical fixtures: no Isaac, no model loading, no actual dataset."""
import ast
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
import numpy as np
from common import HERE, camera_optics, local_path, read_config, verify_local_assets
from projection import aggregate_cells, front_cell_ids, project_patches, rotation_matrix


class OfflineChecks(unittest.TestCase):
    def test_camera_json_parameters_accepted_by_usd(self):
        # This check needs only USD's Python bindings, never SimulationApp.
        try:
            from pxr import Usd, UsdGeom
        except ImportError:
            self.skipTest("USD bindings unavailable; use Isaac's standalone python.sh")
        stage = Usd.Stage.CreateInMemory()
        camera = UsdGeom.Camera.Define(stage, "/Camera")
        optics = camera_optics(read_config())
        camera.GetFocalLengthAttr().Set(optics["focal_length"])
        camera.GetHorizontalApertureAttr().Set(optics["horizontal_aperture"])
        camera.GetVerticalApertureAttr().Set(optics["vertical_aperture"])
        camera.GetClippingRangeAttr().Set(optics["clipping_range"])
        np.testing.assert_allclose(camera.GetClippingRangeAttr().Get(), [.01, 100000])

    def test_local_payload(self):
        manifest = verify_local_assets()
        self.assertEqual({t["label"] for t in manifest["terrains"]}, {"ice", "gravel", "rocks"})
        self.assertEqual(len(manifest["terrains"]), 7)
        for filename in manifest["files"]:
            self.assertFalse(local_path(filename).is_symlink())
        with self.assertRaises(ValueError):
            local_path("../multi_terrain_test8/config.py")

    def test_no_experiment_imports(self):
        for path in HERE.glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                names = [node.module or ""] if isinstance(node, ast.ImportFrom) else (
                    [n.name for n in node.names] if isinstance(node, ast.Import) else [])
                for name in names:
                    self.assertFalse(name.startswith(("multi_terrain_", "terrain_generator", "spot_rl_energy_navigation")))

    def test_rotation_and_depth_purity(self):
        np.testing.assert_allclose(rotation_matrix([1, 0, 0, 0]), np.eye(3))
        config = read_config()["projection"]
        # Synthetic optical camera points down at a flat tile from z=1.
        record = {"intrinsics": [[100, 0, 50], [0, 100, 50], [0, 0, 1]],
                  "camera_quaternion_ros_wxyz": [0, 1, 0, 0],
                  "camera_position_w": [0, 0, 1], "terrain_bounds_xy": [-1, 1, -1, 1]}
        depth = np.ones((100, 100), dtype=np.float32)
        points, center, pure = project_patches(depth, record, config, patches=5)
        self.assertEqual(points.shape, (25, 3))
        self.assertTrue(center.all() and pure.all())
        np.testing.assert_allclose(points[:, 2], 0)
        depth[:] = np.inf
        _, center, pure = project_patches(depth, record, config, patches=5)
        self.assertFalse(center.any() or pure.any())

    def test_cell_mean_preserves_raw_features_and_front_mask(self):
        config = read_config()["projection"]
        raw = np.stack([np.arange(384)-100, np.arange(384)+100, np.ones(384)*999]).astype(np.float32)
        points = np.array([[.1, .1, 0], [.2, .2, 0], [.7, .2, 0]])
        means, counts = aggregate_cells(raw, points, np.array([True, True, False]), config)
        self.assertEqual(counts.sum(), 2)
        np.testing.assert_array_equal(means[2, 2], np.arange(384))
        record = {"yaw_deg": 0, "camera_position_w": [-.08, 0, .7]}
        # Virtual robot at -0.5: nearest central front sample hits cell (2,2).
        ids = front_cell_ids(record, counts, config)
        np.testing.assert_array_equal(ids, [2*12+2])
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
