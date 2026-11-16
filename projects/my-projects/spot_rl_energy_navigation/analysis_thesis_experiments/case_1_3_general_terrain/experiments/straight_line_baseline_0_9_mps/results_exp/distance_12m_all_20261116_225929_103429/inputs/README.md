# Straight-line baseline at 0.9 m/s — 12 m

Run from the my-projects workspace root:

```bash
bash spot_rl_energy_navigation/analysis_thesis_experiments/case_1_3_general_terrain/experiments/straight_line_baseline_0_9_mps/run_exp.sh --distance 12 --headless
```

This runs forty missions, ten per category, with the same thesis IDs E1–E40, starts, goals, initial headings, map, locomotion model and termination rules as the earlier straight-line baselines. No new endpoints are generated. The constant forward command is 0.9 m/s after the original startup alignment. Forward command is zero during alignment. Actual speed can differ from the command. Time and absolute mechanical joint work include alignment; the one-time standing calibration is excluded.

Only the forward command changes relative to straight_line_baseline_1_3_mps. The line tracker retains its 1 m lookahead, heading gain 1.5 and yaw limit 1.5 rad/s. It does not avoid terrain. Physics timestep is 0.005 s, navigation interval 0.2 s, goal radius 0.55 m and mission deadline 63 s.

Inputs are frozen in inputs/missions_12m.json and the matching CSV. reference_speed_m_s is the assigned 0.9 m/s command; policy_mean_speed_m_s preserves the original policy mean. Original policy records remain unchanged.

Each launch creates a new dated directory under results_exp containing episodes.csv, trajectories.csv, power_samples.csv, fall_diagnostics.csv, configuration and source snapshots, summaries and PDF/PNG plots. Existing results are never overwritten. complete.json is written after all requested missions finish and log checks pass; it does not mean that all missions succeeded. Thesis files are not updated automatically.

Append --check to validate inputs and assets without starting the simulator. To preview one mission, remove --headless and append --episodes 1. Preparation does not launch simulations.
