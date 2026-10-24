# Multi terrain binary test4

This run trains **one new PPO policy from random initialization** on two
existing 3×3 USD maps. It does not load the `test2` or `test3` checkpoint.
Each terrain cell is 2×2 m. The first matrix row is at y=0.

| Map | Spots | y=0 | y=2 | y=4 |
| --- | ---: | --- | --- | --- |
| `rocks_gravel__right_rocks` | 16 | asphalt, rocks, rocks | gravel, ice, rocks | gravel, gravel, asphalt |
| `rocks_gravel__right_gravel` | 16 | asphalt, gravel, gravel | rocks, ice, gravel | rocks, rocks, asphalt |

All Spots start at (0, 0) m in the asphalt cell, face the goal at 45 degrees,
and aim for (4, 4) m in the other asphalt cell. Normal reset jitter remains
0.05 m. This removes the correlation between left/right terrain placement
and the shifted start/goal used in test3. The two existing USD files are
reused without editing them.

Training uses 32 Spots, live ZED X RGB-D and one shared frozen DINO model.
The PCA atlas supplies projection axes only. The existing reward, 40 s episode
limit, action limits, locomotion controller, PPO hyperparameters and seed 0
remain unchanged. The default is **409,600 total robot transitions**: each
Spot contributes one transition per vector step. Checkpoints are saved every
10,240 transitions and Ctrl+C writes an interrupted checkpoint. Training
outputs are under `runs/`; evaluation CSVs are under `results/`.

Run from `my-projects`. The user runs all Isaac Sim commands.

Technical probe, eight actions per robot, no training:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/multi_terrain_binary_test4/train.py --headless --probe-steps 8
```

Headless training:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/multi_terrain_binary_test4/train.py --headless --timesteps 409600
```

The final log line contains the checkpoint path. Evaluate each map separately
with a visible Isaac Sim window by substituting that path for `<CHECKPOINT>`:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/multi_terrain_binary_test4/play.py --checkpoint <CHECKPOINT> --layout rocks_gravel__right_rocks --episodes 20
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/multi_terrain_binary_test4/play.py --checkpoint <CHECKPOINT> --layout rocks_gravel__right_gravel --episodes 20
```

Both evaluations use the same centered start and goal as training. Success
on these maps alone does not establish generalization to shifted points or
other arrangements. The camera and feature comparison is a separate
diagnostic planned for later; it is not performed by this training script.
