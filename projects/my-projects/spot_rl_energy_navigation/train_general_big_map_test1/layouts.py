"""One immutable USD replicated into separate, translated simulation environments."""
from dataclasses import dataclass
from pathlib import Path
import json
from functools import lru_cache

HERE = Path(__file__).resolve().parent

@dataclass(frozen=True)
class Layout:
    name: str = 'real_terrains_v2'
    usd: Path = HERE / 'maps/real_terrains_v2.usd'

    @property
    def grid(self):
        return tuple(tuple(row) for row in map_metadata()['grid'])

@dataclass(frozen=True)
class Condition:
    env_id: int
    layout: Layout = Layout()


@lru_cache(maxsize=1)
def map_metadata():
    return json.loads((HERE/'artifacts/map_metadata.json').read_text())

TRAIN_LAYOUTS = (Layout(),)

def training_conditions(num_envs=32):
    if num_envs < 1:
        raise ValueError('At least one environment is required')
    return tuple(Condition(i) for i in range(num_envs))
