# Binary choice: asphalt versus rocks

One reusable trainer uses a named case selected with `--case`. The first case,
`asphalt_rocks`, loads `terrain_generator/binary_choice_asphalt_rocks.usd`.
The original `real_terrains.usd` is kept for older experiments.

| User row | Column 1: x=0 m | Column 2: x=2 m | Column 3: x=4 m |
| --- | --- | --- | --- |
| 1: y=0 m | Fine gravel: **start** | Rocks | Rocks |
| 2: y=2 m | Flat asphalt | Ice | Rocks |
| 3: y=4 m | Flat asphalt | Flat asphalt | Fine gravel: **goal** |

USD cell names use zero-based rows and columns (`Cell_0_0` through `Cell_2_2`).
Every cell is 2 × 2 m; the board spans x,y from -1 to 5 m. Spot starts at
(0, 0), yaw 45° toward (4, 4), with ±0.05 m reset jitter. The goal radius
is 0.55 m. Both the asphalt and rock route can reach the goal without using
ice. Training on a fixed layout alone cannot establish visual generalization
or which route has the lowest energy; the guided energy comparison will be
done after training as requested.

## Training settings

| Parameter | Value |
| --- | ---: |
| Environments / shared policy | 32 Spot / one PPO |
| DINO | One frozen dinov2-small, live RGB-D, 4-image microbatches |
| Camera | ZED X CameraRight optics, 640 × 360, 15° down |
| Observations | 11 state/goal values + 20 nearby DINO PCA-32 cells with validity flags |
| PCA | Existing `runs/ice_choice_atlas.npz` supplies axes only, not cached observations |
| Actions | `vx`, `vy`, yaw rate; frozen IsaacRobotics policy controls joints |
| Command maxima | `vx` ±1.6 m/s, `vy` ±0.9 m/s, yaw ±1.5 rad/s |
| Physics / RL decision | 0.005 s / 0.2 s |
| Episode limit | 40 s = 200 decisions |
| Reward each decision | `3 × progress_m − 0.02 × measured_energy_J − 0.01` |
| Terminal reward | +60 on goal; −25 on fall or board exit |
| PPO | 204,800 total transitions; 32 rollout steps per robot; batch 256; 5 epochs; gamma 0.999; learning rate 0.0003; network [128, 128] |

The mechanical energy is accumulated from projected joint forces and joint
velocities, `sum(abs(torque * velocity)) × dt`, after the initial 2 s settling.
The policy is trained to maximize discounted future reward, not to solve an
exact shortest-energy route. Terrain labels logged in CSV are taken from foot
XY positions; they are not policy observations or contact-force measurements.

## Commands from `my-projects`

1. Technical probe: eight vector actions, no training or checkpoint. This
checks 32 scene copies, Spot, cameras, DINO and finite observations. It may
still expose a problem specific to the new 2 m board; if it fails, do not
start the long run.

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/binary_choice/train.py --case asphalt_rocks --headless --num-envs 32 --probe-steps 8
```

2. Training, only after the probe completes without errors:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/binary_choice/train.py --case asphalt_rocks --headless --num-envs 32 --timesteps 204800
```

The run folder is `binary_choice/runs/asphalt_rocks_online_32env_<date>/`.
It contains `config.json`, `timing.csv`, `episodes.csv`, intermediate policy
checkpoints, `summary.json` and `policy_final.zip`. `--timesteps` counts
transitions from **all** robots. If interrupted with Ctrl+C, the script saves
`policy_interrupted_<steps>_steps.zip`; wait for the `checkpoint=...` line.
The rate printed during training gives the current time estimate. Earlier
Previous 32-camera runs took roughly 1.5 hours for 102,400 transitions.
At a similar rate, 204,800 transitions would take roughly 3 hours; the new
board can change that rate.

3. Visual evaluation of the trained policy. Replace `<RUN>` with the exact
run folder printed by training. Omitting `--headless` opens Isaac Sim:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/binary_choice/play.py --checkpoint spot_rl_energy_navigation/binary_choice/runs/<RUN>/policy_final.zip --episodes 20 --seed 0
```

Evaluation is not training. It writes episode, trajectory and grid CSVs under
`binary_choice/results/`. Mechanical energy is reported per entire episode;
compare the mean of successful arrivals and the number of failures separately.
The checkpoint's case, USD hash, PCA hash and action limits are checked before
playback.

For a future case, add a 3 × 3 USD with 2 m cells and one `cases/<name>.py`
module defining `CASE`. The shared `train.py`, `env.py` and `play.py` continue
to select it with `--case <name>`.
