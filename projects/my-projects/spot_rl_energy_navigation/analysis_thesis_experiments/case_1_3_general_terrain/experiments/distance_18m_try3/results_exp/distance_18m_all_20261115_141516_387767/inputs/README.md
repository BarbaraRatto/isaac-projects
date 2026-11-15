# Eighteen-metre attempt 3 — 400 missions

From the workspace root:

```bash
bash spot_rl_energy_navigation/analysis_thesis_experiments/case_1_3_general_terrain/experiments/distance_18m_try3/run_exp.sh --headless
```

Each launch creates a new timestamped directory in `results_exp/`. Previous runs and the selected collection are unchanged. No simulation was started during preparation.

Append `--category regular`, `--category ice`, `--category stairs` or `--category multiple_choices` to run only that category. Append `--check` for offline input validation. Remove `--headless` to show the viewer.

## Policy and map

The frozen policy is imported test7 `checkpoints_importati/policy_best_stage_3.pt`, at 650,065 transitions. This stage-3 policy was trained up to 12 m; the 18 m panel tests beyond that distance band. No learning, PCA refitting or normalizer updating occurs. The shared evaluation map is unchanged, in `../../maps/`; cells are 6 m along X and 3 m along Y.

## Fixed panel

| Episodes | Category | Missions |
| --- | --- | --- |
| E1–E100 | Regular | 100 |
| E101–E200 | Ice | 100 |
| E201–E300 | Stairs | 100 |
| E301–E400 | Multiple choices | 100 |

All assigned start–goal distances are exactly 18 m. Category rules match earlier 18 m attempts. Multiple choices takes priority when the direct segment crosses at least five cells and four terrain types. Otherwise ice requires at least 1 m of ice and no stairs; stairs requires at least 1 m of stairs and no ice; regular crosses neither ice nor stairs. Categories describe the assigned segment, not the eventual robot route or a guaranteed successful outcome.

`exp_coordinates.json` is authoritative: full-precision coordinates, initial yaw, endpoint surface heights, deadlines, seed, category evidence and sampling settings. The two CSV files export readable coordinates. `retained_reference.json` freezes all 32 currently selected episodes, including tentative cases. Later collection edits do not change this panel. `selection_provenance.json` records generator, mission, reference and model hashes. `coverage_summary.json` records cell and pair counts. There were no newly requested specific cell pairs; `requested_cell_pairs.json` is empty.

## Sampling and diversity

The generator enumerates feasible ordered non-ice/non-stairs cell pairs. It draws random bearings, and uniform starts inside the intersection of the inset start rectangle with the shifted goal rectangle. Goals are starts plus the 18 m displacement. Candidates are filtered before any policy evaluation. Regular and stairs pools receive additional candidates because their geometric options are limited. Selection fills regular, stairs, ice and multiple choices in that order, favoring unused cells and less repeated pairs with seeded random tie breaks. This is a geometrically conditioned and balanced panel, not an unrestricted uniform map sample.

All 35 eligible cells are used as starts and all 35 as goals. There are 101 distinct unordered cell pairs. Some pairs recur with different coordinates, particularly in the stairs category.

| Category | Start cells | Goal cells | Unordered pairs | Largest pair repetition |
| --- | --- | --- | --- | --- |
| regular | 11 | 11 | 9 | 29 |
| ice | 13 | 14 | 13 | 15 |
| stairs | 7 | 7 | 4 | 26 |
| multiple_choices | 31 | 29 | 75 | 2 |

Fixed novelty criteria:

- No unordered start-cell/goal-cell pair from the 32 saved episodes is reused, including reversed routes.
- Endpoint-pair RMS displacement is at least 3 m from every saved mission, checking both directions.
- Every endpoint is at least 0.8 m from every saved endpoint.
- Within try3, starts are at least 0.25 m apart and goals at least 0.25 m apart.
- Within try3, endpoint-pair RMS displacement is at least 0.4 m in both directions.

These are geometric differences in mission assignments; actual policy trajectories may still overlap. The same criteria are enforced for all four categories. No criteria were relaxed to complete this panel.

Endpoints exclude ice and stairs, lie at least 0.65 m from cell boundaries and pass mesh checks on an 11 by 11 footprint with 0.55 m half-width and at most 0.16 m relief. Initial yaw is independent uniform random and saved in the panel. These checks concern placement, not whole-route physical safety. No planner imposes the robot route.

## Outputs

`missions_plan_18m.pdf` contains 100 pages, four assigned segments per page. Dashed segments are mission plans, not measured trajectories; the preview PNG shows the first page. Start labels use the requested E1, E2, etc. sans-serif white boxes in episode colors. Result route plots show ten episodes per category page and retain the same style.

Each run saves `episodes.csv`, `trajectories.csv`, `power_samples.csv`, `fall_diagnostics.csv`, `summary.json`, `evaluation_config.json`, frozen inputs and source files. Logs include actual velocities, commands, measured path length, elapsed time, outcome and absolute mechanical joint work. Goal radius is 0.55 m and timeout is 87 s. Trajectories include initial and pre-reset terminal states. Power is sampled at 0.005 s physics boundaries and integrated trapezoidally; navigation boundaries are 0.2 s. Initial alignment is included; one-time standing calibration is excluded.

All outcomes are retained. `complete.json` means the requested panel finished with logging consistency checks, not that every mission succeeded. Interrupted runs retain partial logs. No result is automatically added to the selected illustrative collection. Straight-line baselines, geodesic distance and thesis edits remain separate work.
