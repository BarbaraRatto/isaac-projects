# Eighteen-metre attempt 1 — test7

Run from the workspace root:

```bash
bash spot_rl_energy_navigation/analysis_thesis_experiments/case_1_3_general_terrain/experiments/distance_18m_try1/run_exp.sh --headless
```

One robot executes forty newly sampled missions sequentially, numbered E1–E40. Every launch creates its own timestamped session in `results_exp/`; no previous session is overwritten. The default checkpoint is the imported `train_general_big_map_test7_SAC/checkpoints_importati/policy_best_stage_3.pt`, at 650,065 transitions. Use `--check` to validate inputs without starting the simulator. Initial calibration and alignment, observations, locomotion and energy measurement follow test7; this is evaluation only, with no learning, PCA refit or normalization updates.

The checkpoint reached curriculum stage 3 (6–12 m plus shorter missions). These eighteen-metre missions therefore test greater distances as well as a different terrain arrangement. The shared evaluation map is `../../maps/` and is unchanged.

`exp_coordinates.json` is the authoritative fixed mission list. It stores full-precision endpoints, initial yaw, footprint heights, deadlines, random seed, library versions, rejection settings, category evidence and draw counts. `exp_coordinates.csv` and `exp_coordinates_18m.csv` are readable exports. Coordinates are drawn before seeing outcomes.

| Local episodes | Category |
| --- | --- |
| E1–E10 | No ice/stairs on the direct segment; not a multiple-surface case |
| E11–E20 | At least 1 m ice on the direct segment, no stairs; not a multiple-surface case |
| E21–E30 | At least 1 m stairs on the direct segment, no ice; not a multiple-surface case |
| E31–E40 | At least five direct-segment cells and four terrain types; ice and stairs may both occur |

Multiple-surface classification has priority. Start and goal are exactly 18 m apart in XY and are both outside ice/stairs, at least 0.65 m from cell edges. Initialization and goal placement use an 11 by 11 footprint of 0.55 m half-width and maximum height relief of 0.16 m. Starts are at least 0.60 m from other starts; goals at least 0.60 m from other goals. Initial yaw is independently random and fixed. These are conditioned random samples, not guarantees of traversal. The policy chooses its route freely.

`missions_plan_18m.pdf` shows all assigned segments, four per page. Dashed segments are not recorded trajectories. Labels E1, E2, etc. are next to start dots in both plans and result figures. They identify this run's local episodes, independently of collection IDs in `../distance_18/`.

Each result session preserves all forty outcomes, measured trajectories, pre-reset final state, absolute joint work, power samples, elapsed simulated time, actual traveled distance, velocities, commands and fall diagnostics. Goal radius is 0.55 m and deadline is 87 s. Joint work is integrated at 0.005 s physics boundaries; trajectory states are logged every 0.2 s. Initial alignment is included; one-time standing calibration is excluded. Raw logs, configuration and input snapshots are retained for every outcome, including failures.

`complete.json` means all requested missions finished and measurement checks passed, not forty successes. Interrupted runs retain available logs. Result routes, category figures and mission metric plots are generated after completion.

The complete run remains separate from `../distance_18/`, which is a manually selected illustrative collection. No episode is automatically selected or merged. No baseline energy, geodesic distance, thesis text or outcome analysis is produced by this runner.
