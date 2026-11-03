"""Load validated immutable preprocessing; fitting is never performed by training."""
import json
from pathlib import Path
import numpy as np
from train_general_big_map_test8_SAC.perception import file_sha256

HERE = Path(__file__).resolve().parent
DEFAULT_SCALERS = HERE/'artifacts/frozen_scalers.json'


def load_scalers(path=DEFAULT_SCALERS):
    data = json.loads(Path(path).read_text())
    if data['schema'] != 1 or data['mode'] != 'frozen_before_training' or data['goal_scale_m'] != 10.:
        raise ValueError('Unsupported frozen scaler format')
    if data['pca_sha256'] != file_sha256(HERE/'artifacts/pca32_big_map.npz'):
        raise ValueError('Frozen scalers calibrated with another PCA')
    manifest = json.loads((HERE/'artifacts/manifest.json').read_text())
    if data['identity'] != manifest['identity']:
        raise ValueError('Frozen scalers and map/DINO identity differ')
    for key, size in [('visual_mean',32),('visual_scale',32),('state_scale',7),('geometry_scale',4),('privileged_scale',20)]:
        values=np.asarray(data[key])
        if values.shape != (size,) or not np.isfinite(values).all() or (key!='visual_mean' and not (values>0).all()):
            raise ValueError(f'Invalid frozen scaler: {key}')
    if data['clip_normalized'] != 10.: raise ValueError('Unexpected clipping')
    return data
