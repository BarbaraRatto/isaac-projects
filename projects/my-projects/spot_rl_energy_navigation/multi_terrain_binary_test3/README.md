# Multi terrain binary test3

This stage continues the successful `multi_terrain_binary_test2` checkpoint.
It trains one shared PPO policy with 32 Spots on three 3×3 boards:

| Board | Spots | Start → goal (m) | Route on right | Route on left |
| --- | ---: | --- | --- | --- |
| Original asphalt/rocks rehearsal | 4 | (0,0) → (4,4) | rocks | asphalt |
| Rocks/gravel, rocks right | 14 | (0.25,0) → (4.25,4) | rocks | gravel |
| Rocks/gravel, rocks left | 14 | (-0.25,0) → (3.75,4) | gravel | rocks |

The new boards have asphalt at both endpoints and ice in the center. Their
start and goal are shifted 0.25 m toward the rocks corridor, so that route is
slightly shorter geometrically. The mirrored USD was held out in test2 but is
part of training in test3. All three USDs are **reused** from earlier runs;
there is no duplicated terrain file in this folder.

The source policy has 409,600 accumulated transitions. This stage adds
204,800 by default, giving 614,400 at completion. The 204,800 new transitions
are shared across the 32 Spots. Reward, 40 s episode limit, action limits,
ZED X camera, live DINO features, frozen locomotion and PPO settings are the
same as in test2. This run tests whether the policy can learn the same terrain
pair on *both* sides; it does not yet prove energy-optimal choice. Whole-route
energy has not been compared for these two layouts.

## Files

- `layouts.py`: exact 4/14/14 scene assignment and start/goal positions.
- `train.py`: technical probe and continuation from the test2 checkpoint;
  writes checkpoint, config, episode CSV, timing CSV and per-layout summary
  under `runs/`.
- `play.py`: one-layout evaluation, with a visible Isaac Sim window unless
  `--headless` is passed; writes episode, trajectory and grid CSVs to `results/`.

## Commands from the `my-projects` directory

The user launches the probe and training. The probe takes eight zero-command
vector steps; it does not train:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/multi_terrain_binary_test3/train.py --headless --probe-steps 8
```

Training:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/multi_terrain_binary_test3/train.py --headless --timesteps 204800
```

After training, substitute the printed run directory for `<RUN>` to evaluate
one map with Isaac Sim visible:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/multi_terrain_binary_test3/play.py \
  --checkpoint spot_rl_energy_navigation/multi_terrain_binary_test3/runs/<RUN>/policy_final.zip \
  --layout rocks_gravel__right_gravel --geometry shifted --episodes 10
```

Other accepted layout names: `rocks_gravel__right_rocks` and
`original_asphalt_rocks` (the latter requires `--geometry original`).
For a start/goal geometry not used in this training, use `--geometry centered`
on either rocks/gravel layout. More varied held-out terrain arrangements can
be prepared after this controlled left/right check.

## Diagnose the mirrored-board failures

`diagnose_mirror.py` is an evaluation only. It records separate flags for
body height, excessive tilt, and each world-boundary crossing, along with
the terminal position. In `both` mode, it first evaluates the learned policy
for three episodes and then drives the same Spot through waypoints on the
right/gravel corridor for three episodes. The guided commands do not train the
policy or change the checkpoint. Run without `--headless` to inspect the
motion in Isaac Sim:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/multi_terrain_binary_test3/diagnose_mirror.py --mode both --episodes 3
```

The script prints one `[DIAG]` row per episode and writes episode and
trajectory CSVs to `results/`. A guided success with gravel entry confirms
that this particular controlled route is physically traversable; guided
failure does not by itself prove that no route exists.
