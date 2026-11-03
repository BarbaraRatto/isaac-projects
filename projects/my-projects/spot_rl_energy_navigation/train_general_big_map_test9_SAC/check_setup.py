"""Validate frozen map, mission bank, DINO and PCA without starting SimulationApp."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_general_big_map_test9_SAC.config import DEFAULT_PCA, LOCAL_DINO
from train_general_big_map_test9_SAC.assets import validate_assets
from train_general_big_map_test9_SAC.artifacts import source_identity
from train_general_big_map_test9_SAC.calibration import validate_calibration
from train_general_big_map_test9_SAC.perception import FrozenPCA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pca', type=Path, default=DEFAULT_PCA)
    parser.add_argument('--dino-model', type=Path, default=LOCAL_DINO)
    args = parser.parse_args()
    validate_assets()
    validate_calibration(FrozenPCA.load(args.pca), source_identity(args.dino_model))
    print('[TEST9_SAC] Frozen map and mission bank verified. Original seven-material PCA reused; no refit.')


if __name__ == '__main__':
    main()
