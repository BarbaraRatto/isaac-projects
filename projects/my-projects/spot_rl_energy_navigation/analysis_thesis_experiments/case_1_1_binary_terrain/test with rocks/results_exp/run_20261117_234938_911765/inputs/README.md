# Binary routes on the opposite map

From the my-projects workspace root:

```bash
bash "spot_rl_energy_navigation/analysis_thesis_experiments/case_1_1_binary_terrain/test with rocks/run_exp.sh" --headless
```

This launches two separate simulator sessions in sequence, five missions per session:

- a_routes_on_b: A1–A5 from the original rocks_right map, replayed on rocks_left.
- b_routes_on_a: B1–B5 from the original rocks_left map, replayed on rocks_right.

The path is not mirrored or rotated. Start, goal and initial yaw remain the recorded values. Swapping the maps puts rocks along the original asphalt branch; shared asphalt start/goal cells and the central ice remain in place. Inputs come from the reference run run_20261112_184253_967025. Frozen source trajectories, original results, identities and coordinates are saved in inputs. New tests do not overwrite the original experiment or thesis files.

## Route and speed control

At navigation step k, the forward command is the original recorded command at step k+1, held over the next 0.2 s interval. Step-zero is an initial state, not an applied command interval. This reproduces the original forward-command time series, rather than replacing it with a mission average.

A geometric pure-pursuit controller follows the recorded XY polyline, with 0.35 m lookahead and the original 1.5 rad/s yaw limit. Forward command is not reduced by the tracker. Yaw commands are corrected to track the curve; they are not replayed unchanged, because the dynamics differ on rocks. The original yaw commands are logged for comparison. Robot pose is measured from simulation; the test never imposes a trajectory by teleporting the robot or replacing its dynamics. The frozen locomotion controller remains active. Navigation actor and DINO/PCA feature extraction are disabled, while camera synchronization is retained.

The reference time profile can end before the replay reaches the goal. In that case the last original forward command is held; every extra step is flagged in speed_profile_extended and its duration is saved. The reference endpoint lies within the original goal radius, rather than at the exact goal center. Only when projection reaches the last 0.05 m of the reference route can the tracking target extend to the assigned goal; endpoint_extension records this. Duration, actual speed, terminal point and path length can therefore differ. Interpret energy together with these differences and tracking error; identical actual trajectories or velocities are not guaranteed.

Original success radius 0.55 m, 40 s deadline, fall criteria, 0.005 s physics step and 0.2 s navigation interval are retained. The two-second standing calibration is excluded from episode time and joint work. Energy is absolute mechanical joint work, integrated at every physics interval with the original trapezoidal rule. No battery-energy estimate is substituted.

## Outputs

Each launch creates results_exp/run_TIMESTAMP, containing a_routes_on_b and b_routes_on_a. Each group saves episodes.csv, trajectories.csv, joint_samples.csv, configuration, incremental summary and completion checks. Source inputs and runner/training code are copied at the run root. episodes.csv includes original energy, path and time alongside measured values, mean/max route error, actual mean speed, failure reason and extended-profile duration.

Plots include routes_overlay.pdf/png, routes_per_mission.pdf and commands_and_tracking.pdf. Dashed lines show original paths; solid lines show measured rock paths. Keep failed episodes and tracking errors when deciding whether a pair is suitable for an energy comparison. A failure is an incomplete mission, not an energy saving.

The source forward-command profile and energy/path integrals are checked before a group is marked complete. complete.json indicates that all requested missions finished and log checks passed; it does not imply that they all reached the goal.

Optional commands:

```bash
# Validate inputs and assets without launching the simulator.
bash "spot_rl_energy_navigation/analysis_thesis_experiments/case_1_1_binary_terrain/test with rocks/run_exp.sh" --check

# Preview one reference mission. Omit --headless to see the simulator.
bash "spot_rl_energy_navigation/analysis_thesis_experiments/case_1_1_binary_terrain/test with rocks/run_exp.sh" --group a_routes_on_b --missions A1
```

Run one group with --group a_routes_on_b or --group b_routes_on_a. Use --output with a new directory, --no-plots to skip figures, or --lookahead to change tracking. Offline tests are in test_replay.py. Preparation does not execute simulation.
