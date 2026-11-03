"""Read-only preflight; no fitting and no Isaac Sim startup."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from train_general_big_map_test8_SAC.frozen_scalers import load_scalers
if __name__ == '__main__':
    data=load_scalers()
    print('[TEST8] All input scalers are frozen. Goal /10 m; references:',data['physical_references'])
