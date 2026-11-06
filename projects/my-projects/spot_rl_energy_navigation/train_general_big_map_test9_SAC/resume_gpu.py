"""Validate the complete old run, then launch a NEW accelerated run with its settings."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def assert_no_training():
    active = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = path.read_bytes().split(b'\0')
            if any(arg.endswith(b'train_general_big_map_test9_SAC/train.py') for arg in argv):
                active.append(path.parent.name)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    if active:
        raise RuntimeError('Old test9 training still running (PIDs '+', '.join(active)+
                           '). Use Ctrl+C inside its tmux and wait for the saved checkpoint and shell prompt.')


def resume_command(project, root, run, config):
    old = config['arguments']
    device = old.get('device', 'cuda:0')
    if not device.startswith('cuda'):
        raise ValueError('This accelerated launcher requires the original CUDA simulator')
    command = ['bash', str(project/'start_test9.sh'), str(root), '--resume', str(run)]
    defaults = dict(timesteps=1500000, num_envs=128, gradient_steps=16, seed=0,
                    torch_threads=4, dino_batch_size=128, eval_every=10000,
                    checkpoint_every=10000, recovery_minutes=30., recovery_keep=2,
                    recovery_reserve_gib=1.)
    for key, default in defaults.items():
        command.extend(('--'+key.replace('_', '-'), str(old.get(key, default))))
    if old.get('dino_model'):
        command.extend(('--dino-model', old['dino_model']))
    command.extend(('--device', device, '--agent-device', device, '--map-backend', 'cuda',
                    '--rgb-backend', 'cuda', '--replay-device', device, '--profile-timing'))
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs_root', type=Path)
    parser.add_argument('--resume', type=Path, help='default: path recorded in latest_run.txt')
    parser.add_argument('--dry-run', action='store_true', help='print commands only')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    project = here.parents[1]
    root = args.runs_root.resolve()
    run = (args.resume or Path((root/'latest_run.txt').read_text().strip())).resolve()
    config = json.loads((run/'config.json').read_text())
    command = resume_command(project, root, run, config)
    device = config['arguments'].get('device', 'cuda:0')
    check = ['bash', str(project/'spot_rl_energy_navigation/cloud/run.sh'), '--python',
             str(here/'validate_acceleration.py'), '--device', device, '--resume', str(run),
             '--report', str(root/'acceleration_validation.json')]
    if config['arguments'].get('dino_model'):
        check.extend(('--dino-model', config['arguments']['dino_model']))
    import shlex
    print('Validate (no simulation or training): '+shlex.join(check), flush=True)
    print('Resume only after validation PASS: '+shlex.join(command), flush=True)
    if not args.dry_run:
        assert_no_training()
        subprocess.run(check, check=True)
        subprocess.run(command, check=True)


if __name__ == '__main__':
    # This launcher deliberately uses standard Python: no torch/simulator imports.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
