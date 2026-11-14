# Binary terrain experiments

Reference: multi_terrain_test13, PPO, policy_final.pt from test13_ppo_20261023_122929_726370.
The navigation policy was trained for 100,000 total transitions across 32 environments.

The CSV contains the ten agreed missions. Each start and goal coordinate was sampled with uniform noise within +/-0.05 m of (0, 0) and (4, 4). The script reads these fixed coordinates and adds no further position noise. Initial heading: 45 degrees.

Run from the workspace root:

    /home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/analysis_thesis_experiments/case_1_1_binary_terrain/experiments/create_exp.py --device cuda:0

One command runs five missions on rocks_right, then five on rocks_left. Each map uses a fresh simulator session with one robot. The viewer opens by default. Add --headless to hide it, or --map rocks_right / --map rocks_left to run one map. Add --check to check the input files without starting the simulator.

Each launch creates a new timestamped folder in results_exp. It includes copies of the runner and coordinates, plus input hashes. Each map folder contains:

- missions.csv: assigned start and goal coordinates.
- episodes.csv: outcome, joint work, time, traveled distance, straight-line distance, falls, timeouts and final position.
- trajectories.csv: initial pose and all navigation-step poses, including the terminal pose, cumulative energy and commands.
- joint_samples.csv: joint torques, joint velocities and absolute mechanical power at every physics step.
- run_config.json: checkpoint, training transitions, model identity and experiment settings.
- mission_plots.png / .pdf: trajectories, cumulative work and distance to goal.
- complete.json: all five missions finished and logging checks passed. A completed mission can end in success or failure.

Energy is absolute joint mechanical work, not battery consumption. It uses the original environment's solver torques, joint velocities and trapezoidal integration at 200 Hz. Mission time excludes initialization and measures simulated time. Navigation samples are saved at 5 Hz. Success uses the training threshold of 0.55 m; the deadline is 40 s. Out-of-bounds exits are recorded separately from falls and timeouts.

All outcomes are retained. Interrupted runs keep partial CSV files and an interrupted.json marker when possible. Closing the viewer stops that map session. The straight-line baseline and interpretation of the results will be added separately.

To redraw existing results, run plot_routes.py with --run followed by the run directory. It replaces mission_plots.png / .pdf in both map folders and writes separate map_a_routes and map_b_routes files in figures, in PDF, PNG and SVG formats. It never starts the simulator. The palette is stored in terrain_palette.json. Maps use horizontal X and vertical Y in world coordinates, with row,column labels in each cell. Times New Roman is used when installed, with Liberation Serif as the fallback. Use --font-file to load a local Times New Roman font file.
