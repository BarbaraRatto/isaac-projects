"""Validate frozen map, mission bank, DINO and PCA without starting SimulationApp."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_general_big_map_test10_SAC.config import DEFAULT_PCA, LOCAL_DINO
from train_general_big_map_test10_SAC.assets import validate_assets
from train_general_big_map_test10_SAC.artifacts import source_identity
from train_general_big_map_test10_SAC.calibration import validate_calibration
from train_general_big_map_test10_SAC.perception import FrozenPCA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pca', type=Path, default=DEFAULT_PCA)
    parser.add_argument('--dino-model', type=Path, default=LOCAL_DINO)
    args = parser.parse_args()
    validate_assets()
    validate_calibration(FrozenPCA.load(args.pca), source_identity(args.dino_model))
    from train_general_big_map_test10_SAC.frozen_scalers import load_scalers,validate_scalers
    from train_general_big_map_test10_SAC.perception import file_sha256
    validate_scalers(load_scalers(),source_identity(args.dino_model),file_sha256(args.pca))
    print('[TEST10_SAC] Map/missions/PCA and immutable FINAL-CELL scalers verified.')


if __name__ == '__main__':
    main()
