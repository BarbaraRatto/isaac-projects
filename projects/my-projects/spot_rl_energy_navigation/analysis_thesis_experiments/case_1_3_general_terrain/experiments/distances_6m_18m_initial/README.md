# Case 1.3: fixed missions on a new map

Run from the workspace root. The two commands start separate sessions of forty missions each:

```bash
bash spot_rl_energy_navigation/analysis_thesis_experiments/case_1_3_general_terrain/experiments/distances_6m_18m_initial/run_exp.sh --distance 6 --headless
bash spot_rl_energy_navigation/analysis_thesis_experiments/case_1_3_general_terrain/experiments/distances_6m_18m_initial/run_exp.sh --distance 18 --headless
```

Run the second command after the first finishes. Remove `--headless` to open the viewer. Use `--distance all` to run all eighty missions in one session. One robot executes the missions sequentially. The default checkpoint is the imported `train_general_big_map_test7_SAC/checkpoints_importati/policy_best_stage_3.pt`, saved at 650,065 transitions. The checkpoint reports curriculum stage 3 (training band 6–12 m, plus easier bands). The 18 m group therefore tests distance extrapolation for this checkpoint as well as a new terrain layout. No automatic model selection is performed. Use `--checkpoint PATH` only to change it deliberately.

Check map, missions, assets, checkpoint identity and network compatibility without starting the simulator:

```bash
bash spot_rl_energy_navigation/analysis_thesis_experiments/case_1_3_general_terrain/experiments/distances_6m_18m_initial/run_exp.sh --distance 6 --check
bash spot_rl_energy_navigation/analysis_thesis_experiments/case_1_3_general_terrain/experiments/distances_6m_18m_initial/run_exp.sh --distance 18 --check
```

## Files

- `create_exp.py`: evaluation runner.
- `exp_coordinates.json`: authoritative list, with complete coordinates, heights, headings, deadlines, selection settings and direct-segment classifications.
- `exp_coordinates.csv`: readable coordinates for all eighty missions.
- `exp_coordinates_6m.csv`, `exp_coordinates_18m.csv`: the two groups.
- `missions.py`: random sampling, geometric classification and fixed mission loading.
- `selection_provenance.json`: generator/source hashes and numerical library versions. The JSON selection settings contain the random seed, sampling rules, thresholds and draw counts.
- `missions_plan_6m.pdf`, `missions_plan_18m.pdf`: ten pages per distance, four planned missions per page. Preview PNGs show the first page. Dashed lines show assigned start–goal segments, not robot trajectories.
- `runtime.py`: evaluation-only map adapter and measurement logging.
- `plots.py`: mission plans and result figures.
- `test_experiments.py`: offline tests of the mission panel, map switching and measurement integrity.
- `results_exp/`: a separate timestamped directory for every execution session. Existing runs are never overwritten.

The new map and metadata are in `../../maps/`. Original training maps, source files, checkpoints, PCA and normalization statistics are preserved. This is evaluation only: there is no learning, PCA fitting, replay buffer or normalizer update.

## Mission selection

Distances are Euclidean XY distances between the assigned start and goal: exactly 6 or 18 m to numerical precision. They are not route lengths. The goal is sampled by a uniform random bearing at the specified radius from a uniformly sampled start; rejection applies the following fixed conditions. Accepted pairs are therefore a conditioned sample, not an unfiltered sample of every possible mission on the map.

Both endpoints exclude ice and stairs. Each is at least 0.65 m from its cell edges. An 11 by 11 square footprint, 0.55 m half-width, is checked on the original collision mesh; height relief must not exceed 0.16 m. These checks concern initialization and goal placement, not guaranteed traversal. Within each distance group, starts are at least 0.60 m from other starts, and goals at least 0.60 m from other goals. The original standing calibration and initial heading alignment are retained. Initial yaw is independently sampled uniformly and saved; it is never redrawn at reset.

Every distance/category has ten missions:

| Distance | Episodes | Category | Fixed geometric rule |
| --- | --- | --- | --- |
| 6 m | 1–10 | Simple regular route | Direct segment on asphalt, ramp and/or gravel only; at most two terrain types. |
| 6 m | 11–20 | Ice, without stairs | At least 1 m of ice on the direct segment, no stairs. |
| 6 m | 21–30 | Stairs, without ice | At least 1 m of stairs on the direct segment, no ice. |
| 6 m | 31–40 | Surface choice | Direct segment has at least two terrain types, no ice/stairs, and at least 1.5 m of rocks/obstacles; an alternative surface option is recorded. |
| 18 m | 41–50 | Simple traversable route | No ice/stairs; does not meet the multiple-choice rule below. Several traversable surfaces may occur. |
| 18 m | 51–60 | Ice, without stairs | At least 1 m of ice, no stairs; does not meet the multiple-choice rule. |
| 18 m | 61–70 | Stairs, without ice | At least 1 m of stairs, no ice; does not meet the multiple-choice rule. |
| 18 m | 71–80 | Multiple surfaces and choices | At least five cells and four distinct terrain types on the direct segment; ice and stairs may both occur. |

At 18 m, multiple-choice cases take priority so the groups do not overlap. This is an operational geometric definition of complex scenes, not proof of how many decisions the policy will make. At 6 m, the surface-choice category takes priority over regular cases.

For a surface-choice mission, the recorded alternative is a two-segment polyline through a traversable cell center, 1.08–1.80 times the direct distance, with at least 1 m less rocks/obstacles. Its segments clear ice/stair rectangles expanded by 0.55 m. These waypoints are selection evidence only: they are never sent to the robot. Reduced exposure to rocks/obstacles is a surface-contrast proxy, not a measured energy ranking or a claim that the alternative is optimal. All traversable cells form a connected four-neighbor region at the cell level. This is not a physical reachability guarantee.

Full precision is preserved in JSON and CSV. Goal/start coordinates are fixed for every later rerun and straight-line reference. To create a different panel, use `missions.py --output NEW_PATH.json`; the agreed JSON is protected against accidental overwriting. Record a new seed explicitly in a separate generator version if a new random panel is intended. Editing the map invalidates the current panel hashes.

## Measurements

All outcomes are retained, including falls, timeouts, out-of-bounds and alignment timeouts. The last two are recorded separately, without silently treating them as falls or ordinary timeouts.

Each session saves:

- `inputs/`: map, metadata, full coordinate source, selected mission list, selection provenance, evaluation scripts and training Python sources.
- `evaluation_config.json`: model path/hash, transitions, original training identity, evaluation map identity, selected episode IDs, simulator settings, goal radius and timing.
- `episodes.csv`: start/goal, category, direct distance, actual traveled distance, energy, elapsed time, mean path speed, outcome and alignment/fall diagnostics.
- `trajectories.csv`: initial state, every navigation boundary (0.2 s) and terminal state before automatic reset; measured world/body velocities, commands, yaw, cumulative energy and path length.
- `power_samples.csv`: absolute joint mechanical power at every physics boundary (0.005 s), with unique boundaries for trapezoidal integration.
- `fall_diagnostics.csv`: tilt/contact diagnostics at navigation boundaries.
- `summary.json`: raw completed episode records, flushed after each mission.
- `complete.json`: all requested missions finished and raw logging checks passed. This does not mean all missions succeeded.
- `interrupted.json`: error or interruption, when applicable; completed and partial raw logs are retained.
- `routes_per_episode.pdf`, `routes_by_category.pdf`, category PNGs and `mission_metrics.pdf/.png`: recorded routes and measurements, including failures.

Energy means absolute mechanical work at the joints, not battery energy. Mission time and energy include the original initial alignment, but exclude the one-time 2 s standing calibration. Goal radius is 0.55 m. The original deadline rule gives 39 s at 6 m and 87 s at 18 m. Path length is the planar distance accumulated at 0.2 s navigation boundaries. Actual velocities are logged separately.

These files prepare policy evaluation only. The paired straight-line controller and any geodesic-distance calculation remain to be implemented later. Energy from an interrupted mission must not be treated as an energy saving over a completed mission. No thesis document or result analysis is produced here.

To regenerate result plots without simulation:

```bash
/home/isaac/isaaclab/2.3.0/_isaac_sim/python.sh spot_rl_energy_navigation/analysis_thesis_experiments/case_1_3_general_terrain/experiments/distances_6m_18m_initial/plots.py --run PATH_TO_SESSION
```

## Directory organization

This experiment has its own input files, launch scripts, plans and `results_exp/`. The shared map is in `../../maps/`. All recorded mission coordinates, headings and deadlines are unchanged by this reorganization. Historical result files, copied inputs, logged absolute paths and previous fingerprint manifests are preserved exactly as recorded; old path strings describe the layout at execution time. Future sessions record the current paths.
