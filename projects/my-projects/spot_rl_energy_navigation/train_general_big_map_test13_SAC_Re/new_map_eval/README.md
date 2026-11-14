# Twenty missions on the new black/grey map

Run from the workspace root:

    bash spot_rl_energy_navigation/train_general_big_map_test13_SAC_Re/test_new_map_20.sh --headless

Remove --headless to open the viewer. One robot runs twenty fixed missions sequentially. The default checkpoint is policy_best.pt from general_map_test13_sac_re_20261110_104043_247976. It was saved at 1,142,939 transitions during the completed 1,500,000-transition run. Later training extensions are not selected automatically.

Check inputs without starting the simulator:

    bash spot_rl_energy_navigation/train_general_big_map_test13_SAC_Re/test_new_map_20.sh --check

## Map and missions

The map is maps/lab_15x15_eval_01.usda, with its matching metadata and disabled-resistance manifest. The layout matches the current test12 evaluation map, with black instead of green. There are 17 grey cells and 8 black cells, each 3 by 3 m. Black has the original test13 friction: static 0.10, dynamic 0.08. Grey retains static 1.0 and dynamic 0.8. No extra resistance is applied on either material.

The training map, mission bank and model remain unchanged. New geometry, privileged contact materials and resistance settings are selected only for this evaluation. The policy, normalizations and PCA remain frozen; the robot chooses its route freely.

All starts and goals are on grey, as in test13 training. The mission JSON is fixed before running the policy; its CSV export and missions_plan.pdf allow inspection of all coordinates. Endpoints have at least 0.65 m clearance from cell edges. No position noise is added at reset. Initial headings are fixed random angles and the original automatic alignment is retained.

| Episodes | Case | Direct distance |
| --- | --- | --- |
| 1 | Grey-only direct segment | 1.5–3 m |
| 2–4 | One isolated black cell on the direct segment | 3–6 m |
| 5–7 | One isolated black cell on the direct segment | 6–10 m |
| 8–10 | Several cells of one black cluster on the direct segment | 6–10 m |
| 11–13 | Separate black regions on the direct segment | 10–14 m |
| 14–16 | Several cells of one black cluster on the direct segment | 10–14 m |
| 17–18 | Additional grey-to-grey missions with black on the direct segment | 3–6 m |
| 19–20 | Additional grey-to-grey missions with black on the direct segment | 6–10 m |

Episodes 1–16 have exactly the same start, goal and initial heading as test12. In episodes 17–20, the original goals were on green, which becomes slippery black here. Those four goals were replaced by randomly selected grey goals in the same distance bands, retaining the original starts and headings. The JSON records this difference explicitly. The two suites therefore share sixteen identical missions, not all twenty.

Only episode 1 has a grey-only direct segment; all other direct segments intersect black. Dashed lines in the mission plan describe these assigned segments, not measured robot trajectories. This is a preliminary simulation suite; no laboratory subset is selected yet.

## Outputs

Each launch creates results/new_map_eval_01_DATE_TIME, without overwriting previous runs. The format matches the test12 evaluation:

- inputs/: frozen map, metadata, resistance manifest, mission list and evaluation code snapshots.
- evaluation_config.json: checkpoint identity and actual transition count, original training identity, new map identity and runtime settings.
- episodes.csv: start/goal, actual traveled distance, mechanical work, simulated time, mean speed, outcome and alignment/fall diagnostics.
- trajectories.csv: initial pose, navigation-step poses and terminal pose captured before auto-reset; actual and commanded velocities are separate fields.
- power_samples.csv: mechanical power at every physics boundary, without duplicate boundaries.
- fall_diagnostics.csv and summary.json: diagnostic samples and completed episode records.
- routes_overview.png / .pdf, routes_per_episode.pdf, mission_metrics.png / .pdf: all outcomes are retained.
- complete.json: all twenty missions finished and logging integrity checks passed. This does not imply twenty successes.

Energy means absolute joint mechanical work, not battery energy. Physics samples are 0.005 s apart; navigation samples are 0.2 s apart. Initial alignment time and energy are included, standing initialization is excluded. The goal radius remains 0.55 m. The original distance-dependent deadline and persistent fall detection are retained. Out-of-bounds and alignment timeout remain distinct outcomes.

Interrupted runs preserve completed results and available logs, with interrupted.json when possible. The runner does not train, change checkpoints, save replay buffers or launch the test12 evaluation.

To redraw plots without simulation, run from the test13 directory:

    /home/isaac/isaaclab/2.3.0/_isaac_sim/python.sh new_map_eval/plots.py --run results/new_map_eval_01_DATE_TIME
