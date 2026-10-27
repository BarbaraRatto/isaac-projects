"""Offline checks for packaging/integrity; no simulator or GPU imports."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import package
import verify


class CloudInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.nav = self.root / "spot_rl_energy_navigation"
        self.exp = self.nav / verify.EXPERIMENT
        self.cloud = self.nav / "cloud"
        self.zed = self.root / ".cloud_assets/ZED_X.usdc"
        self.write(self.zed)
        for path in (self.root / "IsaacRobotics/assets/spot.usd",
                     self.root / "IsaacRobotics/policies/spot/models/spot_policy.pt",
                     self.exp / "train.py", self.exp / "artifacts/pca32.npz",
                     self.cloud / "setup.sh"):
            self.write(path)
        identity = {"dino": {}, "terrain": {}, "missions": {}}
        for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
            path = self.nav / "extracted_features_comparison_v2/models/dinov2" / name
            self.write(path)
            identity["dino"][name] = verify.sha256(path)
        self.write(self.exp / "maps/real_terrains_v2.usd")
        identity["terrain"]["real_terrains_v2"] = verify.sha256(self.exp / "maps/real_terrains_v2.usd")
        for name in ("spawn_bank.npz", "map_metadata.json"):
            self.write(self.exp / "artifacts" / name)
            identity["missions"][name] = verify.sha256(self.exp / "artifacts" / name)
        (self.exp / "artifacts/manifest.json").write_text(json.dumps({"identity": identity}))

    def write(self, path, content=b"fixture"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def collect(self):
        with patch.multiple(package, HERE=self.cloud, PROJECT=self.root, NAV=self.nav):
            return package.collect_files([verify.EXPERIMENT], self.zed)

    def test_missing_ignored_policy_is_reported(self):
        verify.verify_files(self.root, self.zed)
        (self.root / "IsaacRobotics/policies/spot/models/spot_policy.pt").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "spot_policy.pt"):
            verify.verify_files(self.root, self.zed)

    def test_dino_or_terrain_change_is_rejected(self):
        for relative in ("extracted_features_comparison_v2/models/dinov2/model.safetensors",
                         verify.EXPERIMENT + "/maps/real_terrains_v2.usd"):
            with self.subTest(relative=relative):
                path = self.nav / relative
                old = path.read_bytes()
                path.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "changed frozen input"):
                    verify.verify_files(self.root, self.zed)
                path.write_bytes(old)

    def test_package_includes_ignored_inputs_excludes_runs_and_env(self):
        self.write(self.exp / "runs/run/policy_final.pt")
        self.write(self.exp / "results/results.csv")
        self.write(self.cloud / "env.sh", b"local machine settings")
        self.write(self.root / ".env", b"secret")
        files = self.collect()
        self.assertIn("IsaacRobotics/policies/spot/models/spot_policy.pt", files)
        self.assertIn(".cloud_assets/ZED_X.usdc", files)
        self.assertIn(f"spot_rl_energy_navigation/{verify.EXPERIMENT}/artifacts/pca32.npz", files)
        self.assertFalse(any("runs" in Path(name).parts or "results" in Path(name).parts for name in files))
        self.assertFalse(any(Path(name).name in {"env.sh", ".env"} for name in files))

    def test_symlink_and_invalid_experiment_rejected(self):
        (self.exp / "outside").symlink_to("/etc/passwd")
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.collect()
        with patch.multiple(package, HERE=self.cloud, PROJECT=self.root, NAV=self.nav):
            with self.assertRaisesRegex(ValueError, "Invalid experiment"):
                package.collect_files(["../../etc"], self.zed)

    def test_bundle_corruption_and_path_escape_rejected(self):
        self.cloud.mkdir(parents=True, exist_ok=True)
        manifest = self.cloud / "bundle-manifest.json"
        manifest.write_text(json.dumps({"files": {".cloud_assets/ZED_X.usdc": verify.sha256(self.zed)}}))
        verify.verify_bundle(self.root)
        self.zed.write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "missing/changed"):
            verify.verify_bundle(self.root)
        manifest.write_text(json.dumps({"files": {"../../etc/passwd": hashlib.sha256(b"").hexdigest()}}))
        with self.assertRaisesRegex(ValueError, "Invalid manifest"):
            verify.verify_bundle(self.root)


if __name__ == "__main__":
    unittest.main()
