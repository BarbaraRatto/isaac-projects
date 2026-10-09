"""Training duration and shared environment settings for all layouts."""

from dataclasses import replace

from binary_choice.config import NAV_CFG_BINARY, PPO_CFG


NAV_CFG_MULTI = NAV_CFG_BINARY
PPO_CFG_MULTI = replace(PPO_CFG, timesteps=131_072, num_envs=32)
