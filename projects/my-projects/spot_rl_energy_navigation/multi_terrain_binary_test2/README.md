# Multi terrain binary test2

This is a continuation of the successful `binary_choice/asphalt_rocks` policy,
not a fresh PPO initialization. The old failed 32-condition multi-terrain
checkpoint is not used. Only the user runs the Isaac Sim probe, training and
evaluation commands below.

## Training maps

Each board has 3 × 3 cells of 2 × 2 m. Row 1 is at y=0 and column 1 at x=0.
The two new maps have flat asphalt at start (row 1, column 1) and goal (row 3,
column 3), with ice at the center. The top/right route has rocks or obstacles;
the left/bottom route has gravel or ramp. Start is (0.25, 0) m and goal is
(4.25, 4) m, shifting each point only 0.25 m toward the top/right route. The
original binary map keeps its original gravel start/goal and coordinates.

One policy controls 32 Spots: 12 on the original asphalt/rocks map, 10 on
rocks/gravel, 10 on obstacles/ramp. The old map is rehearsal to reduce
forgetting. All maps use the same locomotion policy, ZED X 640×360 camera,
frozen live DINO model and PCA axes, action limits, reward, 40 s episode
limit, PPO settings and seed as the successful binary run. The reward has not
been changed. The new stage adds 204,800 transitions to the source model's
204,800, for 409,600 accumulated transitions in the final checkpoint.

The top/right route is geometrically shorter in the new training maps. The
labels "rocks costly" and "obstacles costly" are hypotheses from terrain
energy-per-meter measurements, not verified rankings of *total route* energy.
Measure the two full routes separately before claiming an energy-optimal RL
choice.

Two held-out USDs swap left and right corridors. They are absent from the
training scene. `play.py` accepts `--geometry shifted` (0.25 m toward the
nominally costly corridor) and `--geometry centered` (original corners).

## Files

- `layouts.py`: map grids, train/test split, start/goal coordinates and 32
  environment assignments.
- `train.py`: technical probe and PPO continuation from the successful binary
  checkpoint; saves configuration, timing, episode CSV, checkpoints and summary
  in this folder's `runs/`.
- `play.py`: one-map evaluation with Isaac Sim GUI unless `--headless` is given;
  saves episode, trajectory and grid CSV in `results/`.
- `terrain_generator/create_multi_terrain_binary_test2.py`: static USD
  generator using the original real-terrain meshes and materials.
- `terrain_generator/multi_terrain_binary_test2/train/*.usd`: two new training
  maps. The old map uses `terrain_generator/binary_choice_asphalt_rocks.usd`.
- `terrain_generator/multi_terrain_binary_test2/test/*.usd`: two held-out maps.

## Commands from the `my-projects` directory

Technical check, no training:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/multi_terrain_binary_test2/train.py --headless --probe-steps 8
```

Training, started by the user only:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/multi_terrain_binary_test2/train.py --headless --timesteps 204800
```

Afterwards, replace `<RUN>` with the run directory printed by `train.py`.
Example GUI evaluation on one held-out map:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/multi_terrain_binary_test2/play.py \
  --checkpoint spot_rl_energy_navigation/multi_terrain_binary_test2/runs/<RUN>/policy_final.zip \
  --layout rocks_gravel__right_gravel --geometry shifted --episodes 10
```

Other layout names are `original_asphalt_rocks` (use `--geometry original`),
`rocks_gravel__right_rocks`, `obstacles_ramp__right_obstacles` and
`obstacles_ramp__right_ramp`. No evaluation performs training.
