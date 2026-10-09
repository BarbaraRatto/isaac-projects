# Multi terrain binary choice

One PPO policy controls 32 independent Spots. Each Spot has a 3 × 3 board
with 2 × 2 m cells, one ZED X RGB-D camera, and live features from the same
frozen DINOv2-small model. The old `binary_choice` training and its USD stay
available. This experiment uses the same locomotion policy, action limits,
reward, visual representation and 40 s episode limit.

There are eight terrain pairs: rocks/gravel, ramp/gravel, obstacles/gravel,
asphalt/ramp, asphalt/rocks, stairs/rocks, ramp/rocks and rocks/obstacles.
The centre cell is always ice; both corner cells are fine gravel. Each pair
has two USDs, one for each exchange of left/bottom and top/right corridors.
Each USD is assigned to two Spots: one with start (0,0) and goal (4,4), one
with the start and goal shifted 0.5 m toward the corridor classified as more
costly. Thus 16 maps yield 32 simultaneous training conditions. For stairs,
"costly" means fall risk; for the other pairs the classification is a
provisional J/m ranking from the earlier terrain-speed measurements. It does
not assert which entire route will use less energy.

Four additional USDs under `terrain_generator/multi_terrain_binary_choice/test/`
use the same terrain pairs in different cell orders. The training manifest
lists only the 16 maps under `train/`; these four cannot enter the training
scene. Only a later evaluation uses them.

## Files and measurements

`terrain_generator/create_multi_terrain_binary_choice.py` generates the USDs
by cropping the existing meshes, keeping their source materials and physics.
`layouts.py` is the single source of truth for the layouts and train/test split.
`scene.py` creates a distinct terrain reference in each environment with
`replicate_physics=False`. `env.py` supplies per-robot goals and records terrain
entries. `train.py` logs timing and `episodes.csv`, including a layout and
geometry identifier for every completed episode. `play.py` evaluates a saved
policy on one selected map and writes episode and trajectory CSVs.

The action is body `vx`, `vy`, yaw rate with maxima ±1.6 m/s, ±0.9 m/s,
±1.5 rad/s. Joint locomotion is controlled by the frozen IsaacRobotics policy.
Reward per decision: `3*goal_progress_m - 0.02*measured_energy_J - 0.01`,
plus 60 at the goal or minus 25 for a fall/board exit. PPO uses 131,072 total
transitions, 32 steps per rollout per Spot, batch 256, 5 epochs, gamma 0.999,
learning rate 0.0003 and network [128,128].

The previous homogeneous 32-robot run took about 2 h 54 min for 204,800
transitions. Heterogeneous physics scenes can run at a different speed, so
the 2–2.5 hour target is an estimate, not a guarantee. The log prints the
actual throughput and remaining-time estimate once training starts.

## Commands from `my-projects`

The USD catalog is already generated in the project. If it needs to be
regenerated, use Isaac's USD Python; this command does not start Isaac Sim:

```bash
/home/isaac/isaaclab/2.3.0/_isaac_sim/python.sh terrain_generator/create_multi_terrain_binary_choice.py --overwrite
```

The probe checks all 32 USD layouts, Spot and camera prims, DINO observations
and eight zero-command vector steps. It does not train or save a policy:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/multi_terrain_binary_choice/train.py --headless --probe-steps 8
```

Only after the probe ends with `[PROBE] completed=8 vector_actions`, launch:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/multi_terrain_binary_choice/train.py --headless --timesteps 131072
```

The script prints the run directory. Checkpoints, `config.json`, `timing.csv`,
`episodes.csv` and `summary.json` are saved under `runs/` in this folder. A
single saved policy is shared by all layouts. Ctrl+C saves an interrupted
checkpoint. The technical probe and training are run by the user.

After training, an example GUI evaluation on a held-out layout is:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/multi_terrain_binary_choice/play.py --checkpoint spot_rl_energy_navigation/multi_terrain_binary_choice/runs/<RUN>/policy_final.zip --layout asphalt_rocks__mixed --geometry centered --episodes 20
```

Evaluation does not train. It records one episode CSV and one trajectory CSV
under `results/`. `--layout` accepts the names in `layouts.py`; test layouts
have `split="test"`. A successful goal does not by itself establish that the
chosen route minimized total energy. Guided route-energy comparisons can be
performed after this training as planned.
