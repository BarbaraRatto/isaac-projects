# Distance 18 m — experiment 1

Forty new missions on the existing held-out seven-terrain map, evaluated with test9. This directory is under `case_1_3_general_terrain/experiments/`; each future panel can have its own sibling directory. Earlier coordinates, models and results remain unchanged and separate.

From the workspace root:

```bash
bash spot_rl_energy_navigation/analysis_thesis_experiments/case_1_3_general_terrain/experiments/distance_18m_1/run_exp.sh --headless
```

From this directory, use `bash run_exp.sh --headless`. Remove `--headless` to open the viewer. The default is one robot executing all forty missions sequentially. Use `--check` instead to verify inputs and checkpoint compatibility without starting the simulator.

## Policy choice

The default checkpoint is pinned to:

`train_general_big_map_test9_SAC/checkpoint_trained_online/policy_best.pt`

It was saved at **600,048 transitions**, with best score `(3, 0.609375, -1840.863076234475)`. The training code ranks checkpoints by evaluated curriculum stage, then success rate, then lower mean absolute joint work among completed missions. This makes `policy_best.pt` a consistent starting choice before observing this panel's outcomes. It is not guaranteed to be best on these eighteen-metre missions.

`policy_final.pt` is available at 1,500,000 transitions. Both checkpoints report curriculum level 3, whose training band is 6–12 m, with easier-band replay. The eighteen-metre panel therefore evaluates distance extrapolation as well as a new terrain arrangement. The checkpoint is not changed automatically after looking at results. To evaluate another model on exactly these coordinates, pass `--checkpoint PATH`; the runner creates a separate result session and records the model path/hash.

The environment, network, PCA, normalizations and checkpoint identity are those of **test9**, not test7. The evaluation map is a copy of the same rearranged map already used for case 1.3; it is not replaced with the training map. Test9's own training map and mission identity are checked separately from evaluation-map provenance. Test9's map was itself built by relocating cells from the original test7 terrain, preserving cell geometry/materials.

## Files and random selection

- `exp_coordinates.json`: authoritative forty-mission list, complete precision, seed, geometric selection rules, yaw, footprint heights, deadlines, draw counts and crossed-cell evidence.
- `exp_coordinates.csv`, `exp_coordinates_18m.csv`: readable coordinate tables.
- `missions.py`: reproducible coordinate extraction and input checks.
- `selection_provenance.json`: generator/library versions and hashes, checkpoint selection details and curriculum state.
- `../../maps/`: shared evaluation map, metadata and PNG/PDF layout previews.
- `missions_plan_18m.pdf`: ten pages showing four assigned missions per page; the preview PNG shows the first four. Dashed segments are assigned start–goal lines, not robot trajectories.
- `create_exp.py`, `runtime.py`, `run_exp.sh`, `plots.py`: evaluation and measurement tools.
- `previous_experiment_hashes.json`: fingerprints of earlier case 1.3 files, before creating this panel.
- `results_exp/`: new timestamped session directory for each launch, without overwriting earlier sessions.

| Episodes | Category | Geometric criterion |
| --- | --- | --- |
| 1–10 | Regular | No ice or stairs on the direct segment; does not meet the multiple-surfaces rule. Other terrain changes are allowed. |
| 11–20 | Ice | At least 1 m of ice on the direct segment, no stairs; does not meet the multiple-surfaces rule. |
| 21–30 | Stairs | At least 1 m of stairs on the direct segment, no ice; does not meet the multiple-surfaces rule. |
| 31–40 | Multiple surfaces and choices | At least five crossed cells and four different terrain types; ice and stairs may both occur. |

The multiple-surfaces rule takes priority to avoid overlapping groups. This is an operational scene classification, not a claim about how many decisions the policy actually makes. Unlike the later six-metre corner-specific panel, the eighteen-metre ice category retains its original criterion.

Starts are drawn uniformly over the inset map bounds. Goal bearing is uniform and its distance from the start is exactly 18 m in XY. Rejection enforces each category and endpoint placement. Both endpoints exclude ice and stairs and lie at least 0.65 m from their cell edges. An 11 by 11 square footprint of 0.55 m half-width must have finite mesh heights with relief no larger than 0.16 m. Starts are at least 0.60 m from other starts; goals at least 0.60 m from other goals. Initial yaw is independently uniform and then fixed in the JSON. All settings and the seed are stored before running the policy. These are conditioned random missions, not unrestricted samples.

All non-ice, non-stairs cells form a connected cell-level region. This geometric fact and the local endpoint checks are not guarantees that Spot will physically traverse every route. The policy is assigned only endpoints and chooses its trajectory freely. No geodesic path or straight-line baseline is computed here.

## Saved results

Each launch stores the checkpoint path/hash/transitions and curriculum state, original test9 training identity, evaluation map identity, selected mission IDs, simulation timing and frozen input/source copies.

Raw files include `episodes.csv`, `trajectories.csv`, `power_samples.csv`, `fall_diagnostics.csv`, `summary.json` and `evaluation_config.json`. They preserve all outcomes: success, fall, timeout, out-of-bounds and alignment timeout. Absolute joint mechanical work, actual traveled distance, simulated time, initial/final positions, measured velocities and commands are recorded. Power is sampled at 0.005 s physics boundaries and integrated trapezoidally; trajectory states are sampled every 0.2 s, including the exact terminal state before auto-reset.

Goal radius is 0.55 m, deadline 87 s. Initial alignment is included in mission time and work; the one-time standing calibration is excluded. Work at the joints is not battery consumption. No learning, replay updates, PCA fitting or normalizer updates occur.

Route PDFs, category PNGs and mission metric plots are generated after all requested missions complete. `complete.json` means all forty missions finished and logging integrity checks passed, not forty successes. Interrupted runs retain their raw logs and an `interrupted.json` file.

No earlier result is merged and no thesis document or outcome analysis is modified by this runner.

## Directory organization

This experiment has its own input files, launch scripts, plans and `results_exp/`. The shared map is in `../../maps/`. All recorded mission coordinates, headings and deadlines are unchanged by this reorganization. Historical result files, copied inputs, logged absolute paths and previous fingerprint manifests are preserved exactly as recorded; old path strings describe the layout at execution time. Future sessions record the current paths.
