# Twelve-metre attempt 1 — 800 missions

Run all four categories, 200 missions each:

```bash
bash spot_rl_energy_navigation/analysis_thesis_experiments/case_1_3_general_terrain/experiments/distance_12m_try1/run_exp.sh --headless
```

One robot executes the missions sequentially, with a separate timestamped directory under `results_exp/` for each launch. Earlier sessions are not overwritten. Use `--check` without simulation, or remove `--headless` for the viewer. To run a single category, append `--category regular`, `--category ice`, `--category stairs` or `--category surface_choice`.

| Episodes | Category | Count |
| --- | --- | --- |
| E1–E200 | Regular | 200 |
| E201–E400 | Ice on the direct segment, without stairs | 200 |
| E401–E600 | Stairs on the direct segment, without ice | 200 |
| E601–E800 | Alternative routes with contrasting surface exposure | 200 |

The policy is imported test7 `checkpoints_importati/policy_best_stage_3.pt`, saved at 650,065 transitions. Twelve metres lies inside the recorded stage 3 training band of 6–12 m. Policy, PCA, normalization, initial alignment, locomotion and sensing stay frozen. The shared evaluation map is `../../maps/`.

## Coordinates and spread

`exp_coordinates.json` is the authoritative fixed panel. It stores full-precision endpoints, footprint heights, initial yaw, deadline, exact crossed-cell evidence, seed, numerical library version and sampling/balancing constraints. `exp_coordinates.csv` and `exp_coordinates_12m.csv` are readable exports. `selection_provenance.json` identifies the generator and model by hashes. The empty `retained_reference.json` records that no twelve-metre collection cases existed before this attempt.

Start–goal distance is exactly 12 m in XY. Endpoints exclude ice and stairs and remain at least 0.65 m from cell edges. Surface placement is checked on an 11 by 11 square footprint with 0.55 m half-width and at most 0.16 m height relief. Initial yaw is independently uniform and fixed before execution. These placement checks do not guarantee successful traversal.

Candidate generation enumerates feasible ordered cell pairs. A bearing is sampled within the angular span of their inset rectangles; the start is uniform in the intersection of the source rectangle and shifted goal rectangle, and the goal is start plus the 12 m displacement. A supplementary pool enriches the rarer surface-choice geometry. Candidates are selected for cell coverage and reduced repetition, with fixed random tie breaks. This is a conditioned and balanced prospective panel, not an unrestricted random map sample.

Within the panel, starts are at least 0.20 m from other starts, goals at least 0.20 m from other goals, and endpoint-pair RMS distance is at least 0.25 m, considering reversed direction. Some cell pairs recur with different coordinates. There are 173 distinct unordered cell pairs overall; all 35 permitted cells occur as starts and all 35 as goals.

| Category | Start cells | Goal cells | Unordered cell pairs | Largest repetition |
| --- | --- | --- | --- | --- |
| Regular | 35 | 35 | 68 | 5 |
| Ice | 32 | 30 | 67 | 5 |
| Stairs | 32 | 32 | 66 | 5 |
| Surface choice | 21 | 20 | 20 | 21 |

The surface-choice requirement constrains the available regions more strongly, so its repeated cell pairs use different coordinates. Full coverage counts are in `coverage_summary.json`.

## Category rules

Ice cases cross at least 1 m of ice and no stairs on the direct segment. Stair cases cross at least 1 m of stairs and no ice. Missions crossing both ice and stairs are not in this panel.

Among direct segments without ice/stairs, surface-choice cases require at least two surface types and at least 1.5 m of rocks/obstacles. A two-segment geometric alternative through an asphalt cell centre must reduce rocks/obstacles exposure by at least 1 m, include at least 0.75 m of asphalt, clear ice/stair rectangles expanded by 0.55 m, and have length 1.05–1.80 times the direct distance. Other hazard-free cases are regular. This is a surface-contrast proxy, not a measured energy ranking. Alternative waypoints are selection evidence only and are never given to the policy.

## Measurements and estimated resources

The goal radius is 0.55 m and the original distance-dependent deadline is 63 s. Initial alignment counts toward time and joint work; the one-time 2 s standing calibration does not. Trajectory states are saved initially, every 0.2 s and at the exact terminal state before auto-reset. Absolute joint mechanical power is logged at 0.005 s physics boundaries and integrated trapezoidally. Energy means joint work, not battery energy.

Each session saves all outcomes in `episodes.csv`, `trajectories.csv`, `power_samples.csv`, `fall_diagnostics.csv`, `summary.json`, `evaluation_config.json` and input/source snapshots. Model weights, replay buffers and videos are not copied into each result session. Completed and interrupted missions retain their available measurements. `complete.json` means the requested panel finished and logging checks passed, not that all goals were reached.

`missions_plan_12m.pdf` contains 200 pages, four planned missions per page. Its first page is also a PNG preview. Dashed lines are assigned start–goal segments, not recorded paths. Start dots use the requested colored E labels in white sans-serif boxes. Result routes are exported individually and in ten-episode category batches, together with measurement plots.

The previous 200-episode eighteen-metre run occupied approximately 56 MiB and took roughly 39 minutes on this computer. At preparation time the workspace had about 9 GiB free. For 800 episodes, roughly 0.2–0.8 GB is a practical storage estimate, depending on mission duration and plots. Runtime scales with actual simulation steps: several hours are plausible, but a run dominated by 63 s timeouts can take substantially longer. No automatic simulation has been launched.

`../distance_12/` is a separate, initially empty collection of manually selected illustrative cases, with provenance and confirmed/tentative reviews. No episode is automatically added there. Original eighteen-metre experiments and the thesis manuscript are unchanged. No straight-line energy baseline or geodesic-distance calculation is implemented here.
