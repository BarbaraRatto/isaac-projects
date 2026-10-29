"""Paired offline comparison: two SAC checkpoints on both saved replay samples."""
import argparse
import copy
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_general_big_map_test6_SAC.probe_goal import sample_near, intervene
from train_general_big_map_test6_SAC.networks import SACPolicy
from train_general_big_map_test6_SAC.actions import navigation_commands


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clipping_stats(normalizer, raw):
    """Measure pre-clamp values; exclude unobserved map features from denominators."""
    cells = raw[:, 15:].reshape(-1, 81, 40)
    groups = {
        'goal_x': (raw[:, :1], normalizer.state, slice(0, 1), None),
        'goal_y': (raw[:, 1:2], normalizer.state, slice(1, 2), None),
        'state_normalized': (torch.cat((raw[:, :8], raw[:, 13:14]), 1), normalizer.state, slice(None), None),
        'visual_valid': (cells[..., :32], normalizer.visual, slice(None), cells[..., 36:37] > .5),
        'geometry_valid': (cells[..., 32:36], normalizer.geometry, slice(None), cells[..., 37:38] > .5),
    }
    result = []
    for name, (values, moments, selection, mask) in groups.items():
        z = (values-moments.mean.float()[selection])/torch.sqrt(moments.var.float()[selection]+1e-6)
        z = z.flatten() if mask is None else z[mask.expand_as(z)]
        if not torch.isfinite(z).all():
            raise ValueError(f'Non-finite normalization: {name}')
        result.append(dict(group=name, values=z.numel(),
                           clipped_fraction=float((z.abs()>10).float().mean()) if z.numel() else None,
                           max_abs_z=float(z.abs().max()) if z.numel() else None))
    return result


@torch.inference_mode()
def compare_normalizers(models, datasets, provenance, args, here):
    snapshots = {name: {k: v.clone() for k, v in model.state_dict().items()}
                 for name, model in models.items()}
    summary, details, clipping, moments = [], [], [], []
    names = ['goal_x', 'goal_y', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz', 'power_ema']
    for label, model in models.items():
        state = model.normalizer.state
        for i, name in enumerate(names):
            moments.append(dict(normalizer=label, feature=name, mean=float(state.mean[i]),
                                std=float(state.var[i].sqrt()), count=float(state.count[i])))
    for source, (raw, ids) in datasets.items():
        for angle in [None, -150, -110, -90, -60, -30, -10, 0, 10, 30, 60, 90, 110, 150, 180]:
            changed = intervene(raw, angle)
            goal = 'original' if angle is None else str(angle)
            outputs = {}
            for norm_label, norm_model in models.items():
                normalized = norm_model.normalizer.actor(changed)
                for row in clipping_stats(norm_model.normalizer, changed):
                    clipping.append(dict(replay=source, goal=goal, normalizer=norm_label, **row))
                for actor_label, actor_model in models.items():
                    # Exactly the deterministic actor path: tanh(mean), no critic or sampling.
                    action = actor_model.actor(normalized)[:, :2].tanh()
                    cmd = navigation_commands(action)
                    if actor_label == norm_label:
                        expected = navigation_commands(actor_model.act(changed, None, deterministic=True)[0])
                        torch.testing.assert_close(cmd, expected, rtol=0, atol=0)
                    outputs[actor_label, norm_label] = cmd.numpy()
            for (actor, norm), cmd in outputs.items():
                delta = cmd[:, 2]-outputs[actor, actor][:, 2]
                summary.append(dict(replay=source, goal=goal, actor=actor, normalizer=norm,
                                    vx=float(cmd[:, 0].mean()), wz=float(cmd[:, 2].mean()),
                                    left_fraction=float((cmd[:, 2]>.1).mean()),
                                    right_fraction=float((cmd[:, 2]<-.1).mean()),
                                    delta_wz_vs_own_norm=float(delta.mean()),
                                    abs_delta_wz_vs_own_norm=float(np.abs(delta).mean())))
                for i, index in enumerate(ids):
                    details.append(dict(replay=source, index=int(index), goal=goal, actor=actor,
                                        normalizer=norm, vx=float(cmd[i, 0]), wz=float(cmd[i, 2]),
                                        delta_wz_vs_own_norm=float(delta[i])))
    for name, model in models.items():
        for key, value in model.state_dict().items():
            assert torch.equal(value, snapshots[name][key]), f'Model changed: {name}/{key}'
    output = args.output or here/'analysis'/('normalizer_comparison_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    for filename, rows in [('summary.csv', summary), ('samples.csv', details),
                           ('clipping.csv', clipping), ('state_statistics.csv', moments)]:
        write_csv(output/filename, rows)
    (output/'metadata.json').write_text(json.dumps(dict(seed=args.seed, sources=provenance,
                                                       mode='crossed_actor_normalizer'), indent=2)+'\n')
    report = ['# Pesi actor e normalizzazione: confronto incrociato', '',
              'Stessi campioni e goal per tutte le combinazioni; nessuna modifica ai modelli.',
              'Le combinazioni originali sono verificate equivalenti al normale percorso di inferenza.', '',
              '## Statistiche salvate del goal (unità: metri)', '',
              '| Normalizzazione | Input | Media | Deviazione standard |', '|---|---|---:|---:|']
    for r in moments:
        if r['feature'].startswith('goal'):
            report.append(f"| {r['normalizer']} | {r['feature']} | {r['mean']:.4f} | {r['std']:.4f} |")
    report += ['', '## Risposte (selezione; tutte le direzioni nei CSV)', '',
               '| Replay | Goal ° | Actor | Norm | vx m/s | wz rad/s | Sinistra | Δ wz vs norm propria |',
               '|---|---|---|---|---:|---:|---:|---:|']
    for r in summary:
        if r['goal'] in ('original', '-90', '-30', '0', '30', '90'):
            report.append(f"| {r['replay']} | {r['goal']} | {r['actor']} | {r['normalizer']} | {r['vx']:.3f} | {r['wz']:.3f} | {100*r['left_fraction']:.1f}% | {r['delta_wz_vs_own_norm']:+.3f} |")
    report += ['', '## Clipping sugli stati originali', '',
               '| Replay | Norm | Gruppo | Valori validi | % oltre ±10 prima del clipping |',
               '|---|---|---|---:|---:|']
    for r in clipping:
        if r['goal']=='original':
            fraction = 'n/a' if r['clipped_fraction'] is None else f"{100*r['clipped_fraction']:.4f}%"
            report.append(f"| {r['replay']} | {r['normalizer']} | {r['group']} | {r['values']} | {fraction} |")
    report += ['', '## Limiti', '',
               'Lo scambio riguarda l’intera normalizzazione dell’actor, non soltanto il goal.',
               'Le combinazioni incrociate non sono policy addestrate: pesi e statistiche si sono adattati insieme.',
               'Una grande variazione dimostra sensibilità allo scambio, non un bug o una causa unica del comportamento.',
               'Assenza di clipping non implica assenza di effetti della normalizzazione.',
               'Goal artificiali e combinazioni incrociate possono essere fuori distribuzione.',
               'Campioni correlati da replay, non dai nove episodi visualizzati; nessuna misura di successo futuro.',
               'Non usare le combinazioni incrociate come correzione pronta per il robot.']
    (output/'REPORT.md').write_text('\n'.join(report)+'\n')
    print('\n'.join(report))
    print(f'\nSaved to: {output}')


def grouped_normalizers(models):
    """Independent copies: change only the selected running statistics."""
    old, current = models['100k'].normalizer, models['300k'].normalizer
    result = {'baseline_300k': copy.deepcopy(current)}
    for name, selection in [('goal_100k', slice(0, 2)), ('other_state_100k', slice(2, 9))]:
        norm = copy.deepcopy(current)
        for field in ('mean', 'var', 'count'):
            getattr(norm.state, field)[selection].copy_(getattr(old.state, field)[selection])
        result[name] = norm
    norm = copy.deepcopy(current)
    norm.visual.load_state_dict(old.visual.state_dict())
    norm.geometry.load_state_dict(old.geometry.state_dict())
    result['gridmap_100k'] = norm
    result['all_100k'] = copy.deepcopy(old)
    return result


@torch.inference_mode()
def compare_groups(models, datasets, provenance, args, here):
    norms = grouped_normalizers(models)
    actor = models['300k'].actor
    snapshots = {name: {k: v.clone() for k, v in model.state_dict().items()} for name, model in models.items()}
    summary, details = [], []
    for source, (raw, ids) in datasets.items():
        for angle in [None, -150, -110, -90, -60, -30, -10, 0, 10, 30, 60, 90, 110, 150, 180]:
            changed = intervene(raw, angle)
            label = 'original' if angle is None else str(angle)
            baseline = navigation_commands(models['300k'].act(changed, None, deterministic=True)[0])
            for condition, norm in norms.items():
                command = navigation_commands(actor(norm.actor(changed))[:, :2].tanh())
                if condition == 'baseline_300k':
                    torch.testing.assert_close(command, baseline, rtol=0, atol=0)
                delta = command[:, 2]-baseline[:, 2]
                summary.append(dict(replay=source, goal=label, condition=condition,
                                    vx=float(command[:, 0].mean()), wz=float(command[:, 2].mean()),
                                    left_fraction=float((command[:, 2]>.1).float().mean()),
                                    right_fraction=float((command[:, 2]<-.1).float().mean()),
                                    delta_wz=float(delta.mean()), abs_delta_wz=float(delta.abs().mean())))
                for i, index in enumerate(ids):
                    details.append(dict(replay=source, index=int(index), goal=label, condition=condition,
                                        vx=float(command[i, 0]), wz=float(command[i, 2]), delta_wz=float(delta[i])))
    for name, model in models.items():
        for key, value in model.state_dict().items():
            assert torch.equal(value, snapshots[name][key]), f'Model changed: {name}/{key}'
    output = args.output or here/'analysis'/('normalizer_groups_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output/'summary.csv', summary)
    write_csv(output/'samples.csv', details)
    (output/'metadata.json').write_text(json.dumps(dict(seed=args.seed, sources=provenance,
                                                       actor='300k', conditions=list(norms)), indent=2)+'\n')
    report = ['# Normalizzazione per gruppi — actor fisso a 300k', '',
              'baseline_300k: tutta la normalizzazione a 300k.',
              'goal_100k: solo media/varianza del goal XY da 100k.',
              'other_state_100k: solo velocità lineari, angolari e potenza media da 100k.',
              'gridmap_100k: solo feature visive PCA e geometriche da 100k.',
              'all_100k: tutta la normalizzazione da 100k, controllo del precedente esperimento.',
              'Mappa grezza, goal, altri input e pesi sono identici fra condizioni; cambiano solo le statistiche indicate.', '',
              '| Replay | Goal ° | Condizione | vx m/s | wz rad/s | Sinistra | Δ wz vs baseline |',
              '|---|---|---|---:|---:|---:|---:|']
    for r in summary:
        if r['goal'] in ('original', '-90', '-30', '0', '30', '90'):
            report.append(f"| {r['replay']} | {r['goal']} | {r['condition']} | {r['vx']:.3f} | {r['wz']:.3f} | {100*r['left_fraction']:.1f}% | {r['delta_wz']:+.3f} |")
    report += ['', 'Tutte le direzioni e i singoli campioni sono nei CSV.', '',
               'Limiti: combinazioni non addestrate, possibili input fuori distribuzione, campioni correlati.',
               'Gli effetti dei gruppi possono interagire: non sono necessariamente additivi.',
               'Il test localizza la sensibilità del comando; non dimostra una causa unica di apprendimento né successo in simulazione.',
               'Nessun modello modificato, nessun limite alle rotazioni. Non usare lo scambio come correzione già validata.']
    (output/'REPORT.md').write_text('\n'.join(report)+'\n')
    print('\n'.join(report))
    print(f'\nSaved to: {output}')


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-100k', type=Path, default=here/'runs/general_map_test6_sac_20261028_201519_133017/policy_final.pt')
    parser.add_argument('--checkpoint-300k', type=Path, default=here/'runs/general_map_test6_sac_20261029_001023_953670/policy_final.pt')
    parser.add_argument('--samples', type=int, default=512, help='Samples per replay')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cross-normalizers', action='store_true', help='Cross actor weights and saved normalization; report clipping')
    parser.add_argument('--normalizer-groups', action='store_true', help='Fixed 300k actor; swap goal, other state, map normalization separately')
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    if args.cross_normalizers and args.normalizer_groups:
        parser.error('Choose only one normalization comparison mode')
    torch.set_num_threads(2)
    paths = {'100k': args.checkpoint_100k, '300k': args.checkpoint_300k}
    models, datasets, provenance = {}, {}, {}
    reference_pca = None
    reference_signature = None
    for label, path in paths.items():
        print(f'Loading {label}: {path}', flush=True)
        saved = torch.load(path, map_location='cpu', weights_only=False)
        transitions = saved['learner']['transitions']
        if transitions != {'100k': 100000, '300k': 300000}[label]:
            raise ValueError(f'{label}: unexpected transition count {transitions}')
        info = saved['training_state']['replay_snapshot']
        if info['transitions'] != transitions:
            raise ValueError('Replay/checkpoint transition mismatch')
        # Shared raw visual coordinates are meaningful only with the same frozen PCA.
        pca = {k: np.asarray(saved['pca'][k]) for k in ('mean', 'components')}
        if reference_pca is not None:
            for key in pca:
                np.testing.assert_array_equal(pca[key], reference_pca[key])
            if saved['signature'] != reference_signature:
                raise ValueError('Experiment signatures differ; verify observation/action compatibility')
        reference_pca, reference_signature = pca, saved['signature']
        model = SACPolicy().eval()
        model.load_state_dict(saved['learner']['policy'])
        models[label] = model
        replay = path.parent/info['file']
        with np.load(replay, allow_pickle=False) as archive:
            if int(archive['total']) != transitions:
                raise ValueError('Replay archive total mismatch')
        print(f'Sampling {label} replay (streaming, CPU only)', flush=True)
        raw, ids, eligible = sample_near(replay, args.samples, args.seed)
        datasets[label] = (torch.from_numpy(raw.copy()), ids)
        provenance[label] = dict(checkpoint=str(path.resolve()), replay=str(replay.resolve()),
                                 transitions=transitions, eligible=eligible, sampled=len(ids))
        del saved

    if args.normalizer_groups:
        compare_groups(models, datasets, provenance, args, here)
        return
    if args.cross_normalizers:
        compare_normalizers(models, datasets, provenance, args, here)
        return

    rows, summary = [], []
    angles = [None, -150, -110, -90, -60, -30, -10, 0, 10, 30, 60, 90, 110, 150, 180]
    with torch.inference_mode():
        for source, (raw, ids) in datasets.items():
            for angle in angles:
                # Same tensor for both policies; no angular-rate/action ablation here.
                changed = intervene(raw, angle)
                outputs = {}
                for label, model in models.items():
                    outputs[label] = navigation_commands(model.act(changed, None, deterministic=True)[0]).numpy()
                early, late = outputs['100k'], outputs['300k']
                delta = late[:, 2]-early[:, 2]
                name = 'original' if angle is None else str(angle)
                record = dict(replay=source, goal_angle_deg=name, samples=len(ids))
                for label, cmd in outputs.items():
                    record.update({f'vx_{label}': float(cmd[:, 0].mean()),
                                   f'wz_{label}': float(cmd[:, 2].mean()),
                                   f'left_fraction_{label}': float((cmd[:, 2] > .1).mean()),
                                   f'right_fraction_{label}': float((cmd[:, 2] < -.1).mean())})
                record.update(delta_wz_mean=float(delta.mean()),
                              abs_delta_wz_mean=float(np.abs(delta).mean()),
                              delta_wz_p10=float(np.quantile(delta, .1)),
                              delta_wz_p90=float(np.quantile(delta, .9)))
                summary.append(record)
                for i, index in enumerate(ids):
                    rows.append(dict(replay=source, replay_index=int(index), goal_angle_deg=name,
                                     original_goal_x=float(raw[i, 0]), original_goal_y=float(raw[i, 1]),
                                     distance_m=float(raw[i, :2].norm()),
                                     vx_100k=float(early[i, 0]), wz_100k=float(early[i, 2]),
                                     vx_300k=float(late[i, 0]), wz_300k=float(late[i, 2]),
                                     delta_wz=float(delta[i])))
    output = args.output or here/'analysis'/('goal_comparison_100k_300k_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output/'summary.csv', summary)
    write_csv(output/'samples.csv', rows)
    (output/'metadata.json').write_text(json.dumps(dict(seed=args.seed, sources=provenance,
                                                       normalization='own saved normalizer per policy'), indent=2)+'\n')
    report = ['# Confronto goal: 100k e 300k', '',
              'Entrambe le policy ricevono esattamente gli stessi input grezzi, da ciascuno dei due replay.',
              'Ogni policy usa la propria normalizzazione salvata. PCA e firma esperimento verificate uguali.',
              'Inferenza deterministica CPU; nessun aggiornamento, simulazione o limite alla rotazione.',
              'Goal modificato a distanza costante; mappa, velocità e azione precedente rimangono invariati.', '',
              '| Replay | Goal ° (+ sinistra) | vx 100k | vx 300k | wz 100k | wz 300k | Sinistra 100k | Sinistra 300k | Δ wz medio |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        report.append(f"| {r['replay']} | {r['goal_angle_deg']} | {r['vx_100k']:.3f} | {r['vx_300k']:.3f} | {r['wz_100k']:.3f} | {r['wz_300k']:.3f} | {100*r['left_fraction_100k']:.1f}% | {100*r['left_fraction_300k']:.1f}% | {r['delta_wz_mean']:+.3f} |")
    report += ['', '## Limiti', '',
               'vx in m/s, wz in rad/s. Sinistra = wz > 0.1 rad/s; Δ = 300k meno 100k sullo stesso campione.',
               'I replay contengono esperienza di training, non gli episodi visualizzati. Campioni temporalmente correlati.',
               'Le due sorgenti non hanno necessariamente la stessa distribuzione di missioni, distanza, terreno o stato.',
               'Una differenza fra policy sugli stessi input riguarda la policy completa, normalizzazione inclusa.',
               'Non isola da sola reward, ottimizzatore, curriculum o normalizzazione come causa del cambiamento.',
               'Goal artificiali e stati del replay tardivo possono essere fuori distribuzione, soprattutto per il modello 100k.',
               'Girare verso il goal non è sempre corretto: possono esserci ostacoli o terreni costosi.',
               'Questi comandi non misurano successo o efficienza futura; servirà una verifica in simulazione.']
    (output/'REPORT.md').write_text('\n'.join(report)+'\n')
    print('\n'.join(report))
    print(f'\nSaved to: {output}')


if __name__ == '__main__':
    main()
