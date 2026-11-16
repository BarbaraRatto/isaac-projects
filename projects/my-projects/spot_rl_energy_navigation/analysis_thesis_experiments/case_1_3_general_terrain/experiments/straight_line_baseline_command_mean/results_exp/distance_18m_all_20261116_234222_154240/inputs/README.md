# Straight-line baseline with mean policy commands — 12 m and 18 m

Run from my-projects:

```bash
bash spot_rl_energy_navigation/analysis_thesis_experiments/case_1_3_general_terrain/experiments/straight_line_baseline_command_mean/run_exp.sh --distance 12 --headless
bash spot_rl_energy_navigation/analysis_thesis_experiments/case_1_3_general_terrain/experiments/straight_line_baseline_command_mean/run_exp.sh --distance 18 --headless
```

At each distance, the forty missions, IDs E1–E40, starts, goals, initial headings, map, locomotion model and termination rules match the previous baselines. There are ten missions per category. No endpoints are generated.

For each mission, commands are extracted from its original policy episode in trajectories.csv. The end-of-step row holds the command used over the preceding step. Means are weighted by step duration, separately for alignment and navigation; the initial step-zero row is excluded. Failed policy episodes use commands recorded before termination.

- Alignment: zero forward speed; yaw-command magnitude is the absolute value of the signed mean original alignment command. Rotation direction follows the current heading error, and yaw is zero within the original 10-degree tolerance. This avoids continuing to rotate after reaching the desired heading. The original three stable samples and 12 s alignment deadline remain active. This replaces the previous proportional alignment speed; it does not replay the original yaw profile or impose its duration.
- Navigation: constant forward command equal to the original mean commanded forward speed during navigation only. The line tracker keeps its original heading corrections (1 m lookahead, gain 1.5, yaw limit 1.5 rad/s). A constant navigation yaw command would curve away from the line and is not used.

The mean yaw is signed; positive and negative values indicate opposite turns. Its absolute value is used as the rotation speed, with the direction corrected from the current error. alignment_abs_yaw_rad_s is also recorded for audit and is the mean absolute original command, which can differ from the absolute signed mean.

reference_speed_m_s stores the assigned navigation command. command_means stores both phase means, durations, source trajectory hashes and every original command interval used in the calculation. policy_effective_mean_speed_m_s preserves the previous path/time metric. Raw original policy results remain unchanged. Actual measured speeds can differ from the commands.

Each launch creates a dated results_exp subdirectory with episode results, trajectory commands and velocities, mechanical power, diagnostics, input/source snapshots, summaries and PDF/PNG plots. Episode records and trajectories include alignment_reference_yaw_rad_s. Absolute joint mechanical work and elapsed time include initial alignment; standing calibration is excluded. Goal radius is 0.55 m, mission deadline 63 s at 12 m and 87 s at 18 m, physics timestep 0.005 s and navigation interval 0.2 s.

Append --check to validate inputs and assets without launching Isaac Sim. Preparation does not launch simulations. Old tests, results and thesis files are unchanged.
