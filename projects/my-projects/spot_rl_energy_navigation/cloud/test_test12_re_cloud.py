"""Offline launcher tests; no installation, CUDA or simulator is started."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


class Test12ReLaunchers(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='test12_re cloud ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'new project'
        self.cloud = self.root/'spot_rl_energy_navigation/cloud'
        self.cloud.mkdir(parents=True)
        for name in ('start_test12_re.sh', 'monitor_test12_re.sh', 'setup_test12_re.sh', 'run.sh'):
            shutil.copy2(HERE/name, self.cloud/name)
        self.env = dict(os.environ)
        for key in ('GENERAL_MAP_ISAACLAB', 'SPOT_RUNTIME_DIR', 'SPOT_ZED_USD'):
            self.env.pop(key, None)

    def call(self, script, *args):
        return subprocess.run(['bash', str(self.cloud/script), *map(str, args)],
                              env=self.env, text=True, capture_output=True)

    def test_fresh_setup_and_file_check_do_not_enter_training(self):
        (self.cloud/'setup.sh').write_text('[[ "$SPOT_RUNTIME_DIR" == "$EXPECTED_RUNTIME" ]] || exit 9\n'
                                         '[[ "$SPOT_ZED_USD" == "$EXPECTED_CAMERA" ]] || exit 10\n'
                                         'printf "%s\\n" "$@"\n')
        self.env.update(SPOT_RUNTIME_DIR='/old/runtime', SPOT_ZED_USD='/old/camera',
                        EXPECTED_RUNTIME=str(self.root/'.spot-runtime'),
                        EXPECTED_CAMERA=str(self.root/'.cloud_assets/ZED_X.usdc'))
        for option in ('--check-only', '--skip-probe'):
            result = self.call('setup_test12_re.sh', option)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(),
                             ['--experiment', 'train_general_big_map_test12_SAC_Re', option])
        self.assertFalse((self.root/'.spot-runtime').exists())

    def test_start_creates_new_resume_destination_and_preserves_arguments(self):
        (self.cloud/'env.sh').write_text('# fixture\n')
        (self.cloud/'run.sh').write_text('printf "%s\\n" "$@"\n')
        runs = self.root/'persistent runs'
        result = self.call('start_test12_re.sh', runs, '--resume', '/old run', '--timesteps', '1500000')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('train_general_big_map_test12_SAC_Re\n--run-dir\n', result.stdout)
        self.assertIn('--resume\n/old run\n--timesteps\n1500000\n', result.stdout)
        target = Path((runs/'latest_run.txt').read_text().strip())
        self.assertEqual(target.parent, runs)
        self.assertTrue(target.name.startswith('general_map_test12_sac_re_'))
        self.assertFalse(target.exists())  # the real train.py creates it after calibration
        invalid = self.call('start_test12_re.sh', runs, '--run-dir=/do/not/use')
        self.assertEqual(invalid.returncode, 2)

    def test_reuse_checks_software_and_uses_new_assets_without_touching_test9(self):
        old = self.root.parent/'old project'
        old_cloud = old/'spot_rl_energy_navigation/cloud'
        old_cloud.mkdir(parents=True)
        runtime = old/'runtime'
        (runtime/'venv/bin').mkdir(parents=True)
        private = runtime/'venv/bin/python'
        private.write_text('#!/usr/bin/env bash\nexec python3 "$@"\n')
        private.chmod(0o755)
        wrapper = runtime/'isaaclab-python.sh'
        wrapper.write_text('#!/usr/bin/env bash\nprintf "PROBE %s\\n" "$@"\n')
        wrapper.chmod(0o755)
        old_env = old_cloud/'env.sh'
        old_env.write_text('export SPOT_RUNTIME_DIR='+shlex.quote(str(runtime))+'\n'
                           'export GENERAL_MAP_ISAACLAB='+shlex.quote(str(wrapper))+'\n'
                           'export SPOT_ZED_USD=/old/camera.usdc\n')
        before = old_env.read_bytes()
        # Fake only runtime/package validation here; real bundle integrity is
        # exercised by the packager on every staged and archived payload.
        (self.cloud/'verify.py').write_text(
            'import os, sys\nprint("VERIFY", *sys.argv[1:])\n'
            'if "--runtime" in sys.argv:\n'
            ' assert os.environ["SPOT_ZED_USD"].endswith("new project/.cloud_assets/ZED_X.usdc")\n')
        result = self.call('setup_test12_re.sh', '--reuse-existing', old)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn('--check-bundle --experiment train_general_big_map_test12_SAC_Re', result.stdout)
        self.assertIn('--runtime --experiment train_general_big_map_test12_SAC_Re', result.stdout)
        self.assertIn('probe_env.py', result.stdout)
        self.assertNotIn('train.sh', result.stdout)
        self.assertEqual(old_env.read_bytes(), before)
        resolved = subprocess.run(['bash', '-c', 'source "$1"; printf "%s\\n" "$SPOT_ZED_USD" "$SPOT_RUNTIME_DIR"',
                                   '_', str(self.cloud/'env.sh')], capture_output=True, text=True, check=True)
        self.assertEqual(resolved.stdout.splitlines(), [str(self.root/'.cloud_assets/ZED_X.usdc'), str(runtime)])
        skipped = self.call('setup_test12_re.sh', '--reuse-existing', old, '--skip-probe')
        self.assertEqual(skipped.returncode, 0, skipped.stderr)
        self.assertNotIn('PROBE', skipped.stdout)

    def test_monitor_waits_while_first_calibration_has_no_run_log(self):
        runs = self.root/'runs'
        runs.mkdir()
        result = self.call('monitor_test12_re.sh', runs)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('No run started', result.stdout)



if __name__ == '__main__':
    unittest.main()
