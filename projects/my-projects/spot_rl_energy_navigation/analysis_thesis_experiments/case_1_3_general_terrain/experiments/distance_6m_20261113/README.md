# New six-metre experiment — 2026-11-13

This is a separate, redesigned evaluation panel prepared after preliminary tests. It contains forty new missions, ten in each category. All original coordinates, scripts, 18 m missions and previous results remain in the sibling `distances_6m_18m_initial` directory and are not changed or merged.

Run from this directory:

```bash
bash run_exp.sh --headless
```

Remove `--headless` to open the viewer. Check inputs without starting Isaac Sim:

```bash
bash run_exp.sh --check
```

The directory already contains the inputs. Every launch creates a new timestamped session inside `results_exp/`. No previous session is overwritten. The default is one robot, forty sequential missions, six-metre Euclidean start–goal separation. The runner does not train the policy.

## Checkpoint

The pinned checkpoint is:

`train_general_big_map_test7_SAC/runs/general_map_test7_sac_20261030_003829_628341/policy_best.pt`

It was saved at 650,065 transitions, curriculum stage 3. Its file hash and best-score metadata differ from the previously used imported `policy_best_stage_3.pt`, but **all 66 policy state tensors, including normalization buffers, are identical**. Using this file alone does not change the learned policy. This panel changes the assigned missions and their selection rules. The tensor comparison and both checkpoint hashes are recorded in `selection_provenance.json`.

## Inputs and selection

- `exp_coordinates.json`: authoritative coordinates, initial yaw, heights, deadlines and geometric evidence for each category.
- `exp_coordinates.csv`, `exp_coordinates_6m.csv`: readable coordinate tables.
- `missions.py`: reproducible random generator and input validation.
- `selection_provenance.json`: seed-related provenance, generator/library versions, file hashes and checkpoint comparison. The actual random seed and rejection settings are saved in the coordinate JSON.
- `../../maps/`: shared evaluation map, metadata and layout previews. Terrain geometry and distribution have not changed.
- `missions_plan_6m.pdf`: ten pages, four assigned missions per page; its preview PNG shows the first four. Dashed segments are not robot trajectories.
- `previous_experiment_hashes.json`: fingerprints of previous inputs and result files, taken before creating this folder.

The new panel is sampled before observing its outcomes. It is a conditioned geometric sample, not an unrestricted random sample of map missions. Starts are uniform over the inset map bounds; goal bearing is uniform and the goal lies exactly 6 m from the start. Rejection applies the following categories and spatial quotas:

| Episodes | Category | Selection rule |
| --- | --- | --- |
| 1–10 | Regular | No ice/stairs; up to three traversable terrain types, including rocks and asphalt; direct segment clears hazardous cells by 0.55 m. |
| 11–20 | Ice corner | Exactly one ice cell crossed through two adjacent sides, 0.4–1.6 m of ice on the direct segment, no stairs; an outside-corner detour with 0.55 m clearance exists and is at most 1.45 times the direct length. |
| 21–30 | Stairs | At least 1 m of stairs on the direct segment, no ice. |
| 31–40 | Surface choice | At least 1.5 m of rocks on the direct segment; a one-bend alternative includes at least 0.75 m of asphalt and at least 1 m less rocks/obstacles, with length 1.08–1.80 times the direct distance. |

Surface-choice cases take priority over regular cases. For the ten regular missions, the start–goal midpoint assigns the zone: X=18 m splits west/east, Y=9 m splits south/north. Quotas are west-south 3, west-north 2, east-south 2, east-north 3: five on each side and five in each vertical half. This spreads regular cases across the map. Rough-surface exposure is geometric selection evidence, not a measured energy ranking.

Both endpoints exclude ice and stairs and remain at least 0.65 m from cell edges. An 11 by 11 footprint with 0.55 m half-width must have finite surface heights and relief at most 0.16 m. Starts are at least 0.60 m from other starts; goals at least 0.60 m from other goals. Initial yaw is independently uniform and fixed in the JSON. Footprint checks concern placement, not a guarantee that the entire mission is traversable.

Alternative waypoints are used only to document scene selection. They are not sent to the policy. The policy receives the usual observations and chooses its route freely. No geodesic distance or straight-line controller is implemented here.

## Measurements

The runner retains the parent experiment's measurement procedure:

- `episodes.csv`: assigned coordinates, category, actual traveled distance, absolute joint mechanical work, simulated time, mean path speed, outcome, alignment and fall diagnostics.
- `trajectories.csv`: initial, every 0.2 s and exact pre-reset terminal state; actual velocities, commands, yaw, cumulative work and distance.
- `power_samples.csv`: mechanical power every 0.005 s physics boundary for trapezoidal integration.
- `fall_diagnostics.csv`, `summary.json`, `evaluation_config.json` and frozen inputs/source copies.
- Recorded route PDFs, category PNGs and mission metrics after all forty missions finish.
- `complete.json` only after all missions finish and logging checks pass; `interrupted.json` on an interrupted run.

All outcomes are saved. Falls, timeout, out-of-bounds and alignment timeout remain distinct. The goal radius is 0.55 m and the deadline is 39 s. Time and joint work include initial alignment, excluding the one-time standing calibration. Energy is not battery energy. No thesis document, aggregate study result or collage with previous sessions is modified by this runner.

## Directory organization

This experiment has its own input files, launch scripts, plans and `results_exp/`. The shared map is in `../../maps/`. All recorded mission coordinates, headings and deadlines are unchanged by this reorganization. Historical result files, copied inputs, logged absolute paths and previous fingerprint manifests are preserved exactly as recorded; old path strings describe the layout at execution time. Future sessions record the current paths.
