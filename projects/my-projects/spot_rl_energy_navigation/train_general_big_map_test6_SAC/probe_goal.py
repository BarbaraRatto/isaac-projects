"""CPU-only paired goal intervention on raw, saved SAC replay observations."""
import argparse
import csv
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_general_big_map_test6_SAC.networks import SACPolicy
from train_general_big_map_test6_SAC.actions import navigation_commands


ROTATION_CONDITIONS = {
    'original': (),
    'yaw_rate_zero': (7,),  # goal2 + linear velocity3 + angular velocity XYZ3
    'previous_yaw_action_zero': (12,),  # gravity3 + previous action (vx, yaw)
    'both_zero': (7, 12),
}


def intervene(raw, angle, zero_columns=()):
    changed = raw.clone()
    if angle is not None:
        distance = torch.linalg.vector_norm(raw[:, :2], dim=1)
        rad = np.deg2rad(angle)
        changed[:, 0] = distance * np.cos(rad)
        changed[:, 1] = distance * np.sin(rad)
    for column in zero_columns:
        changed[:, column] = 0.0  # physical/raw zero, BEFORE normalization
    unchanged = [i for i in range(raw.shape[1]) if i not in (0, 1, *zero_columns)]
    assert torch.equal(changed[:, unchanged], raw[:, unchanged])
    return changed


def sample_near(path, count, seed):
    """Uniform priority sample; stream the archive rather than load 2.5 GiB."""
    rng = np.random.default_rng(seed)
    selected, priorities, indices = [], np.empty(0), np.empty(0, dtype=int)
    eligible = 0
    with zipfile.ZipFile(path) as archive, archive.open('actor.npy') as stream:
        version = np.lib.format.read_magic(stream)
        if version != (1, 0):
            raise ValueError(f'Unsupported NPY header: {version}')
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        if fortran or shape[1:] != (3255,) or dtype != np.dtype('float32'):
            raise ValueError(f'Unexpected replay actor format: {shape}, {dtype}')
        selected = np.empty((0, shape[1]), dtype=dtype)
        for start in range(0, shape[0], 1024):
            n = min(1024, shape[0] - start)
            raw = stream.read(n * shape[1] * dtype.itemsize)
            batch = np.frombuffer(raw, dtype=dtype).reshape(n, shape[1])
            distance = np.linalg.norm(batch[:, :2], axis=1)
            ids = np.flatnonzero((distance > .55) & (distance <= 2.5))
            eligible += len(ids)
            selected = np.concatenate((selected, batch[ids]))
            indices = np.concatenate((indices, start + ids))
            priorities = np.concatenate((priorities, rng.random(len(ids))))
            keep = np.argsort(priorities)[:count]
            selected, indices, priorities = selected[keep], indices[keep], priorities[keep]
    if not len(selected):
        raise ValueError('No observations in the distance interval (0.55, 2.5] m')
    return selected, indices, eligible


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=here / 'runs/general_map_test6_sac_20261029_001023_953670/policy_final.pt')
    parser.add_argument('--samples', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--rotation-ablation', action='store_true',
                        help='Compare original, zero measured yaw rate, zero previous yaw action, and both')
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    info = checkpoint['training_state']['replay_snapshot']
    if info['transitions'] != checkpoint['learner']['transitions']:
        raise ValueError('Replay/checkpoint transition mismatch')
    replay = args.checkpoint.parent / info['file']
    print(f'Reading real training observations from {replay}', flush=True)
    raw, ids, eligible = sample_near(replay, args.samples, args.seed)
    policy = SACPolicy().eval()
    policy.load_state_dict(checkpoint['learner']['policy'])
    del checkpoint
    obs = torch.from_numpy(raw.copy())
    frozen_normalizer = {k: v.clone() for k, v in policy.normalizer.state_dict().items()}

    @torch.inference_mode()
    def commands(x):
        return navigation_commands(policy.act(x, None, deterministic=True)[0]).numpy()

    baseline = commands(obs)
    distance = torch.linalg.vector_norm(obs[:, :2], dim=1)
    rows, summaries, curves = [], [], {}
    # The original goal is the baseline; only raw goal x/y change in interventions.
    conditions = ROTATION_CONDITIONS if args.rotation_ablation else {'original': ()}
    angles = [None, -150, -110, -90, -60, -30, -10, 0, 10, 30, 60, 90, 110, 150, 180]
    for condition, angle in [(c, a) for c in conditions for a in angles]:
        changed = intervene(obs, angle, conditions[condition])
        cmd = commands(changed)
        label = 'original' if angle is None else str(angle)
        curves[condition, label] = cmd
        paired_delta = cmd[:, 2] - curves['original', label][:, 2]
        summaries.append(dict(condition=condition, goal_angle_deg=label, samples=len(obs),
                              vx_mean=float(cmd[:, 0].mean()), wz_mean=float(cmd[:, 2].mean()),
                              wz_p10=float(np.quantile(cmd[:, 2], .1)), wz_p90=float(np.quantile(cmd[:, 2], .9)),
                              left_fraction=float((cmd[:, 2] > .1).mean()),
                              right_fraction=float((cmd[:, 2] < -.1).mean()),
                              paired_delta_wz_vs_original_condition=float(paired_delta.mean()),
                              paired_abs_delta_wz_vs_original_condition=float(np.abs(paired_delta).mean()),
                              delta_wz_mean=float((cmd[:, 2]-baseline[:, 2]).mean())))
        for i, replay_id in enumerate(ids):
            rows.append(dict(replay_index=int(replay_id), condition=condition, intervention_deg=label,
                             distance_m=float(distance[i]), original_goal_x=float(obs[i, 0]),
                             original_goal_y=float(obs[i, 1]), original_yaw_rate=float(obs[i, 7]),
                             original_previous_yaw_action=float(obs[i, 12]),
                             paired_delta_wz=float(paired_delta[i]), vx=float(cmd[i, 0]), wz=float(cmd[i, 2])))
    for k, v in frozen_normalizer.items():
        assert torch.equal(v, policy.normalizer.state_dict()[k]), 'Normalizer changed'
    assert np.array_equal(commands(obs), baseline), 'Nondeterministic baseline'
    prefix = 'goal_rotation_probe_' if args.rotation_ablation else 'goal_probe_'
    output = args.output or here / 'analysis' / (prefix + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    for name, data in [('samples.csv', rows), ('summary.csv', summaries)]:
        with (output/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    metadata = dict(checkpoint=str(args.checkpoint.resolve()), replay=str(replay.resolve()),
                    seed=args.seed, eligible=eligible, sampled=len(obs), conditions=conditions,
                    paired_left_minus_right_by_condition={c: dict(
                        mean=float((curves[c, '60'][:, 2]-curves[c, '-60'][:, 2]).mean()),
                        positive_fraction=float((curves[c, '60'][:, 2]>curves[c, '-60'][:, 2]).mean()))
                        for c in conditions})
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    report = ['# Sensibilità al goal — SAC', '',
              'Osservazioni reali del replay di training, NON dei nove episodi di test.',
              'Il goal XY cambia direzione conservando la distanza; stessi campioni in tutte le condizioni.',
              'Condizioni: original = stato originale; yaw_rate_zero = sola velocità angolare Z azzerata;',
              'previous_yaw_action_zero = solo comando precedente di rotazione azzerato; both_zero = entrambi.',
              'Azzeramenti sugli input grezzi, prima della normalizzazione. Mappa e altri input invariati.',
              'Non viene imposto alcun limite al comando prodotto dalla rete.',
              'Policy deterministica, normalizzazione congelata, nessuna simulazione o aggiornamento.', '',
              f'Campioni: {len(obs)} su {eligible} disponibili entro 0.55–2.5 m; seed {args.seed}.', '',
              '| Condizione | Goal (gradi; positivo=sinistra) | vx medio (m/s) | wz medio (rad/s) | Sinistra >0.1 rad/s | Delta wz vs stato originale, stesso goal |',
              '|---|---|---:|---:|---:|---:|']
    for row in summaries:
        report.append(f"| {row['condition']} | {row['goal_angle_deg']} | {row['vx_mean']:.3f} | {row['wz_mean']:.3f} | {100*row['left_fraction']:.1f}% | {row['paired_delta_wz_vs_original_condition']:+.3f} |")
    report += ['', '## Interpretazione e limiti', '',
               'Una risposta diversa a destra e sinistra dimostra sensibilità al goal, non una navigazione corretta.',
               'Una risposta quasi invariata suggerisce scarso uso del goal o predominanza degli altri input.',
               'La rotazione verso il goal non è sempre la scelta corretta in presenza di ostacoli.',
               'I goal modificati possono essere fuori dalla distribuzione di training o su zone non percorribili.',
               'Anche azzerare velocità o azione precedente può creare stati fuori distribuzione o incoerenti con gli altri input.',
               'Una variazione dopo gli azzeramenti isola una dipendenza funzionale, non prova che tali input vadano eliminati.',
               'Il test isola l’effetto immediato dell’input goal sui comandi; non identifica da solo la causa di apprendimento,',
               'non misura la dinamica futura e non dimostra un difetto della reward. I campioni temporali sono correlati.']
    (output/'REPORT.md').write_text('\n'.join(report)+'\n')
    print('\n'.join(report))
    print(f'\nSaved to: {output}')


if __name__ == '__main__':
    main()
