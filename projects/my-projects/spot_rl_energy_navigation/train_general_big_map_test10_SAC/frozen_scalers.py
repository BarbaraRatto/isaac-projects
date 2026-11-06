"""Immutable physical references and calibrated FINAL-CELL visual statistics."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
DEFAULT_SCALERS=HERE/'artifacts/frozen_scalers.json'
PHYSICAL=dict(state_scale=[2.]*6+[400.],geometry_scale=[.7,np.pi/2,np.pi/2,.05],
              privileged_scale=[1.]*8+[300.]*4+[1.5]*8)


def artifact_digest(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,allow_nan=False).encode()).hexdigest()


def load_scalers(path=DEFAULT_SCALERS,data=None):
    data=json.loads(Path(path).read_text()) if data is None else data
    if data.get('schema')!=1 or data.get('mode')!='frozen_final_cells_before_training':
        raise ValueError('A fresh final-cell scaler calibration is required')
    if data.get('goal_scale_m')!=10. or data.get('clip_normalized')!=10.:
        raise ValueError('Unexpected normalization contract')
    for name,size in (('visual_mean',32),('visual_scale',32),('state_scale',7),('geometry_scale',4),('privileged_scale',20)):
        value=np.asarray(data[name])
        if value.shape!=(size,) or not np.isfinite(value).all() or (name!='visual_mean' and (value<=0).any()):
            raise ValueError('Invalid frozen scaler: '+name)
    for name,expected in PHYSICAL.items():
        if not np.array_equal(data[name],expected):raise ValueError('Physical reference differs: '+name)
    return data


def validate_scalers(data,identity,pca_sha256):
    load_scalers(data=data)
    if data['identity']!=identity or data['pca_sha256']!=pca_sha256:
        raise ValueError('Scalers calibrated with a different map/PCA/DINO/mission bank')
    calibration=data['calibration']
    if min(calibration['train_counts'])<2000 or len(calibration['train_counts'])!=7:
        raise ValueError('Insufficient balanced material coverage')
    if min(calibration['validation_counts'])<250 or len(calibration['validation_counts'])!=7:
        raise ValueError('Insufficient independent-view validation')
    if calibration['fit_source']!='final_valid_grid_cells' or not calibration['disjoint_viewpoints']:
        raise ValueError('Must calibrate on actual final grid cells with disjoint viewpoints')
    if calibration['validation_clip_fraction']>.01 or max(calibration.get('clip_by_material',[0.]))>.01:
        raise ValueError('Frequent clipping on held-out calibration cells; fix calibration before training')
