# Twenty missions on the new map

Run from the workspace root:

    bash spot_rl_energy_navigation/train_general_big_map_test12_SAC_Re/test_new_map_20.sh --headless

Remove --headless to open the viewer. The default is one robot, twenty sequential missions, and policy_best.pt from general_map_test12_sac_re_20261109_203415_65592. This checkpoint was saved at 912,178 transitions during the completed 1,500,000-transition run. It is never selected from later training extensions. Use --checkpoint only to choose another model deliberately.

Check all inputs without starting the simulator:

    bash spot_rl_energy_navigation/train_general_big_map_test12_SAC_Re/test_new_map_20.sh --check

## Map and mission selection

The map is maps/lab_15x15_eval_01.usda with its matching metadata and resistance manifest. It has the agreed cell swaps: (4,3) and (3,3) are green; (4,4) and (3,4) are grey. Geometry, privileged contacts and green resistance all use this new layout. The training map, training mission bank, checkpoint identity and source files remain unchanged. The evaluation adapter changes only constructor dependencies in its own process and restores the original module references after construction.

The JSON is the authoritative mission list; the CSV is a readable export. Coordinates are sampled inside cells with at least 0.65 m clearance from cell edges, rounded to four decimal places and fixed before running the policy. Starts and goals are not redrawn at reset. Initial headings are fixed random angles; the original automatic alignment is retained and its time/energy are included. PCA, normalizations and policy weights remain frozen. There is no planner constraining the route.

| Episodes | Case | Direct distance |
| --- | --- | --- |
| 1 | Grey-only direct segment | 1.5–3 m |
| 2–4 | One isolated green cell on the direct segment | 3–6 m |
| 5–7 | One isolated green cell on the direct segment | 6–10 m |
| 8–10 | Several cells of one green cluster on the direct segment | 6–10 m |
| 11–13 | Separate green regions on the direct segment | 10–14 m |
| 14–16 | Several cells of one green cluster on the direct segment | 10–14 m |
| 17–18 | Goal on green | 3–6 m |
| 19–20 | Goal on green | 6–10 m |

All starts are grey. Sixteen goals are grey and four are green. Only episode 1 has a direct segment entirely on grey; every other direct segment includes green. These classifications describe the assigned start–goal line, not the route Spot must take. Twenty missions are a preliminary simulation suite; no subset for the laboratory is selected yet.

missions_plan.pdf shows the assigned segments in five pages. Dashed segments are a mission plan, not experiment results. missions_plan_preview.png shows the first four missions.

## Saved results

Each launch creates results/new_map_eval_01_DATE_TIME, without overwriting earlier runs:

- inputs/: exact map, resistance manifest, mission list and evaluation script snapshots.
- evaluation_config.json: checkpoint path/hash, actual checkpoint transitions, training budget, training identity and evaluation map identity.
- episodes.csv: start/goal, traveled distance, mechanical work, time, measured mean speed, outcome and alignment/fall diagnostics.
- trajectories.csv: initial state, every 0.2 s state, and terminal state before auto-reset; actual world/body velocities and commanded velocities are distinct fields.
- power_samples.csv: absolute joint mechanical power at every 0.005 s physics boundary, without duplicate boundaries.
- fall_diagnostics.csv: tilt/contact diagnostics at each navigation boundary.
- summary.json: completed episodes, updated after each mission.
- routes_overview.png / .pdf, routes_per_episode.pdf, mission_metrics.png / .pdf: plots of all completed outcomes.
- complete.json: all twenty missions finished and logging integrity checks passed. This does not imply twenty successes.

The deadline is the original distance-dependent rule, rounded up to a navigation step: clip(15 + 4 × direct distance, 30, 90) seconds. Goal radius is 0.55 m. Original persistent fall detection is retained. Out-of-bounds and alignment timeout are saved separately, never silently relabeled as falls or ordinary timeouts.

Energy is absolute joint mechanical work, not battery consumption. Distance is the planar path sampled at navigation steps. Measured speed is also logged directly from the simulator, so it is available separately from displacement-based mean speed. Initialization/standing calibration is excluded; initial alignment is included.

Partial runs preserve completed episode records and available raw logs. A normal interruption creates interrupted.json. No checkpoint copy, replay buffer, training update or straight-line energy baseline is created.

To regenerate result plots without simulation:

    /home/isaac/isaaclab/2.3.0/_isaac_sim/python.sh new_map_eval/plots.py --run results/new_map_eval_01_DATE_TIME

Run that command from the test12 directory. Existing binary-terrain and thesis documents are not changed by this evaluation.
