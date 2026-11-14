# Eighteen-metre attempt 2 — 200 missions

Run all four categories, fifty missions each, from the workspace root:

```bash
bash spot_rl_energy_navigation/analysis_thesis_experiments/case_1_3_general_terrain/experiments/distance_18m_try2/run_exp.sh --headless
```

To execute only one category, append `--category regular`, `--category ice`, `--category stairs` or `--category multiple_choices`. Each category has fifty missions and retains its panel episode IDs. Use `--check` to validate inputs without starting the simulator. Remove `--headless` for the viewer. Each launch creates a new timestamped session in `results_exp/`, recording the requested category; earlier sessions are never overwritten.

The policy is the same imported test7 `checkpoints_importati/policy_best_stage_3.pt`, saved at 650,065 transitions. It was trained through curriculum stage 3, up to 12 m; these eighteen-metre missions extend beyond that training distance band. The shared evaluation map in `../../maps/` is unchanged. There is no learning, replay update, PCA refit or normalizer update.

## Fixed panel

| Episodes | Category | Missions |
| --- | --- | --- |
| E1–E50 | Regular: no ice/stairs on the direct segment, not a multiple-surface case | 50 |
| E51–E100 | Ice on the direct segment, no stairs, not a multiple-surface case | 50 |
| E101–E150 | Stairs on the direct segment, no ice, not a multiple-surface case | 50 |
| E151–E200 | At least five crossed cells and four terrain types, possibly both ice and stairs | 50 |

The category rules match try1. Multiple-surface cases take priority. Ice/stair categories require at least 1 m of the respective surface on the direct segment.

`exp_coordinates.json` is the authoritative panel with full-precision coordinates, orientation, footprint heights, deadlines, geometric category evidence, seed and selection rules. `exp_coordinates.csv` and `exp_coordinates_18m.csv` are readable exports. `selection_provenance.json` records model/generator/library hashes. `missions_plan_18m.pdf` has fifty pages, four assigned segments per page; dashed lines are not robot trajectories. All plan/result start dots are labeled E1, E2, etc., in the requested larger sans-serif white boxes, retaining episode colors.

## Diversity and saved-case exclusion

The frozen `retained_reference.json` contains the eight selected illustrative episodes present when try2 was built, including tentative E4. Later changes to the collection do not alter this reference or the fixed panel.

All 35 non-ice/non-stairs cells are used as starts and all 35 as goals. The panel contains 90 distinct unordered cell pairs, treating reversed pairs as the same spatial connection:

| Category | Start cells | Goal cells | Unordered cell pairs |
| --- | --- | --- | --- |
| Regular | 22 | 21 | 18 |
| Ice | 18 | 17 | 18 |
| Stairs | 8 | 8 | 5 |
| Multiple surfaces | 26 | 27 | 50 |

Some cell pairs recur with different points. In particular, the stairs-only category has few remaining compatible combinations on this map after excluding saved cases and respecting the multiple-surface priority. The largest repetition is 14 missions on one unordered cell pair. These are not fifty different cell pairs per category.

The fixed novelty constraints are:

- No retained start-cell/goal-cell pair is reused, including its reverse.
- The RMS displacement of corresponding endpoints from every retained mission is at least 3 m, also checking reversed direction.
- Every new endpoint is at least 0.8 m from every retained endpoint.
- Within try2, starts are at least 0.25 m from other starts and goals at least 0.25 m from other goals.
- Within try2, pair endpoint RMS distance is at least 0.4 m, checking both directions.

Endpoint RMS is a geometric similarity measure for assigned start–goal segments. It does not predict or guarantee that the policy's actual trajectories will differ. This is a prospective, geometrically conditioned and balanced mission panel, not an unrestricted uniform random map sample.

Candidate generation enumerates ordered cell pairs capable of an 18 m separation inside their inset rectangles. It draws a random bearing in the corner-defined angular span, then a uniform start in the intersection between the start rectangle and the goal rectangle shifted by the 18 m displacement. The goal is start plus that displacement. Candidates are filtered by category and novelty, then selected for cell coverage, low cell/pair repetition and random tie breaks, filling rarer categories first. Feasible user-suggested combinations are attempted first. All selection details, limits and coverage are recorded in the JSON; selection happens before observing any policy outcome.

Both endpoints exclude ice and stairs and remain 0.65 m from cell edges. An 11 by 11 footprint of 0.55 m half-width must have finite mesh heights with relief at most 0.16 m. Initial yaw is independently uniform and fixed. These checks validate placement, not the physical traversability of the whole route. No planner constrains the policy's actual route.

`coverage_summary.json` contains cell-use counts and pair diversity. `requested_cell_pairs.json` documents the proposed examples. E151 uses (6,2) to (4,0), and E152 uses (6,0) to (3,2). The pair (6,0) to (4,2) is geometrically possible but did not yield an accepted sampled mission under all fixed constraints. The pair (6,6) to (2,6) cannot reach 18 m: its maximum separation with the placement margin is approximately 14.48 m. Cells are 6 m along X and 3 m along Y.

## Results

Every run preserves all requested outcomes in `episodes.csv`, `trajectories.csv`, `power_samples.csv`, `fall_diagnostics.csv`, `summary.json` and `evaluation_config.json`, with frozen inputs and source files. Goal radius is 0.55 m and the deadline is 87 s. Trajectories include the initial state, each 0.2 s navigation boundary and the pre-reset terminal state. Joint mechanical work is integrated trapezoidally from power at 0.005 s physics boundaries. Initial alignment is included; one-time standing calibration is excluded. Actual velocities, commands, traveled distance and fall diagnostics are retained.

`complete.json` means the requested missions finished and logging integrity checks passed, not that all succeeded. Interrupted runs retain raw data. No failure is silently removed or substituted.

Result plots include a per-episode PDF, category route pages in batches of ten, batch PNGs and mission metric plots. Ten routes per page keep episode labels readable. The same ten trace colors repeat across batches; E labels uniquely identify the missions.

No episode is automatically added to the selected illustrative collection. Earlier runs, selected episodes and the thesis document remain unchanged. Straight-line energy baselines and geodesic distances remain separate future work.
