# Paired straight-line baseline — 12 m and 18 m

Run from the `my-projects` workspace root, one session at a time:

```bash
bash spot_rl_energy_navigation/analysis_thesis_experiments/case_1_3_general_terrain/experiments/straight_line_baseline/run_exp.sh --distance 12 --headless
bash spot_rl_energy_navigation/analysis_thesis_experiments/case_1_3_general_terrain/experiments/straight_line_baseline/run_exp.sh --distance 18 --headless
```

Each command executes forty missions, ten per category. IDs E1–E40 are the thesis figure/table IDs at the requested distance. The preparation does not start either simulation.

For an optional short visual check, remove `--headless` and append `--episodes 1`. To run a category only, append `--category regular`, `ice`, `stairs`, `surface_choice` (12 m only), or `multiple_choices` (18 m only). Filtering preserves thesis IDs. A partial session is separate from the full forty-mission session; it does not automatically resume or merge runs.

## Frozen missions and speeds

`inputs/missions_12m.json` and `inputs/missions_18m.json` freeze the exact selected starts, goals, initial yaw, surface heights and deadlines. Matching CSV files expose these values and each assigned speed. Each case retains the original policy run, source episode, archived collection ID, original category, final presentation category and full policy result. The 18 m presentation grouping includes the previously requested category transfers.

The forward command is the original measured path length divided by original simulated episode time, using full-precision data. It includes the original alignment time, as in the thesis speed tables. Failed reference-policy missions use their measured values up to termination, even though those measurements are masked in thesis tables. Assigned speeds range from approximately 0.799 to 1.310 m/s at 12 m and 0.847 to 1.371 m/s at 18 m; none exceeds the original 1.6 m/s command limit.

`missions.py --build DISTANCE` was used once to create these panels. It refuses to replace an existing panel. Current selections and raw policy runs remain unchanged. No new start/goal is generated.

## Controller

The navigation actor is replaced by deterministic tracking of the fixed start–goal segment. After the original startup alignment, forward speed remains constant at the assigned value. A point 1 m ahead of the robot's projection onto the segment determines the desired heading, capped at the goal. Yaw command is 1.5 times the wrapped heading error, limited to ±1.5 rad/s. Commands are updated every 0.2 s and use the original action-to-command mapping. Lateral velocity command is zero.

This corrects physical drift toward the same fixed line. It performs no terrain avoidance and uses no image, material label, energy signal or policy action to choose a route. The line controller uses simulated robot pose and yaw, not perfect kinematic teleportation. Actual speed and path can differ from the command because of dynamics, terrain and falls; both commanded and actual velocities are saved. The original heading-alignment routine overrides the forward command to zero during startup, with its original tolerance, stability count and alignment timeout.

Robot, locomotion policy, mesh layout, material physics, contact sensors, 0.005 s physics step, 0.2 s navigation interval, 0.55 m goal radius, fall logic and safe-map limits are retained. Deadlines remain 63 s and 87 s respectively. Rendering/camera synchronization is kept, but DINO, PCA extraction and the learned navigation actor are not executed. The frozen learned locomotion controller remains active. No navigation checkpoint is required by this runner.

Standing calibration uses the original two-second procedure on an asphalt point from the source panel of the first selected mission. It is performed once per launched baseline batch and excluded from mission time/energy. The calibration point and per-case original calibration points are recorded. This reconstructs the same placement procedure; recorded historical simulator states were not available for exact replay.

## Results

Each launch creates a new directory, for example:

```
results_exp/
  distance_12m_all_YYYYMMDD_HHMMSS_microseconds/
  distance_18m_all_YYYYMMDD_HHMMSS_microseconds/
```

A session saves:

- `episodes.csv`: outcome, energy, path, simulated time, actual mean speed, assigned speed, phase breakdown, fall diagnostics and original-policy IDs.
- `trajectories.csv`: measured poses, velocities, applied commands, assigned speed, line error, cumulative path/energy and the pre-reset terminal sample.
- `power_samples.csv`: absolute mechanical joint power at every physics boundary, with trapezoidal integration.
- `fall_diagnostics.csv`: inclination, body contact and terrain-height diagnostics.
- `evaluation_config.json`, incremental `summary.json`, and frozen missions, map, local/training source files and input hashes.
- Route and measurement plots in PDF/PNG, unless `--no-plots` is supplied. All failures and partial-mission measurements remain in the raw logs and plots.

`complete.json` is written only after all requested missions finish and logging consistency checks pass. It does not imply all missions succeeded. Interrupted sessions keep logs and `interrupted.json`. Existing sessions are never overwritten. These results are not automatically inserted into thesis tables and no policy-versus-baseline statistical analysis is performed here.

## Offline validation

Append `--check` to either launch command to validate the frozen mission panel, terrain assets and locomotion model without launching Isaac Sim. Unit checks can be run with the Isaac Python interpreter and `test_baseline.py`. Preparation validates data and controller/logging logic offline; the actual Isaac Sim execution is left to the user.
