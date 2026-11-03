"""Offline one-time scaler calibration from existing seven-terrain RGBD and motion data."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from train_general_big_map_test8_SAC.perception import FrozenPCA, file_sha256


def replay_sample(path, count=4096, seed=123):
    with zipfile.ZipFile(path) as archive:
        # Actor rows are streamed, avoiding allocation of the full replay.
        with archive.open('actor.npy') as stream:
            version = np.lib.format.read_magic(stream)
            if version != (1, 0): raise ValueError('Unsupported NPY format')
            shape, order, dtype = np.lib.format.read_array_header_1_0(stream)
            if order or shape[1:] != (3255,) or dtype != np.dtype('float32'):
                raise ValueError('Unexpected actor format')
            ids = np.sort(np.random.default_rng(seed).choice(shape[0], min(count, shape[0]), replace=False))
            parts = []
            for start in range(0, shape[0], 1024):
                n = min(1024, shape[0]-start)
                chunk = np.frombuffer(stream.read(n*shape[1]*4), dtype=dtype).reshape(n, shape[1])
                selected = ids[(ids>=start)&(ids<start+n)]-start
                if len(selected): parts.append(chunk[selected].copy())
        with archive.open('privileged.npy') as stream:
            privileged = np.load(stream, allow_pickle=False)[ids]
    return np.concatenate(parts), privileged


def reference(values, floor, step):
    values = np.asarray(values)
    if values.size == 0 or not np.isfinite(values).all(): raise ValueError('Invalid calibration values')
    p99 = float(np.quantile(np.abs(values), .99))
    return max(floor, math.ceil(p99/step)*step), p99


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'artifacts/frozen_scalers.json')
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Scaler already exists: choose a NEW output to recalibrate deliberately')
    source = HERE.parent/'train_general_big_map_test5/artifacts'
    pca_path = HERE/'artifacts/pca32_big_map.npz'
    assert file_sha256(source/'pca32_big_map.npz') == file_sha256(pca_path), 'PCA mismatch'
    samples = source/'pca32_big_map.samples.npz'
    with np.load(samples, allow_pickle=False) as stored:
        features, labels = stored['features'], stored['terrain_codes']
    codes, counts = np.unique(labels, return_counts=True)
    assert np.array_equal(codes,np.arange(7)) and np.all(counts==counts[0]) and counts[0]>=1000
    pca = FrozenPCA.load(pca_path)
    projected = pca.transform(features).astype(np.float64)
    mean, std = projected.mean(0), projected.std(0)
    std_floor = max(1e-3, .01*float(np.median(std)))
    visual_scale = np.maximum(std,std_floor)
    replay_paths = [HERE.parent/'train_general_big_map_test6_SAC/runs/general_map_test6_sac_20261029_001023_953670/replay_300000.npz',
                    HERE.parent/'train_general_big_map_test7_SAC/runs/general_map_test7_sac_20261029_163507_402204/replay_200000.npz']
    actors, privs, sources = [], [], []
    for path in replay_paths:
        print(f'[SCALERS] Reading motion samples: {path}',flush=True)
        a,p = replay_sample(path)
        actors.append(a); privs.append(p)
        sources.append(dict(path=str(path), sha256=file_sha256(path), sampled=len(a)))
    a,p = np.concatenate(actors),np.concatenate(privs)
    cells = a[:,15:].reshape(-1,40); geometry = cells[cells[:,37]>.5,32:36]
    contact = p[:,8:12]>.5
    scales, quantiles = {}, {}
    for name, values, floor, step in [
        ('linear_velocity_m_s',a[:,2:5],2.,.5), ('angular_velocity_rad_s',a[:,5:8],2.,.5),
        ('power_w',a[:,13],100.,100.), ('relative_height_m',geometry[:,0],.5,.1),
        ('roughness_m',geometry[:,3],.05,.01), ('force_n',p[:,12:16][contact],100.,50.),
        ('slip_m_s',p[:,16:24].reshape(-1,4,2)[contact],1.,.5)]:
        scales[name], quantiles[name] = reference(values,floor,step)
    state_scale = [scales['linear_velocity_m_s']]*3+[scales['angular_velocity_rad_s']]*3+[scales['power_w']]
    geometry_scale = [scales['relative_height_m'],math.pi/2,math.pi/2,scales['roughness_m']]
    # Friction remains raw (scale=1), forces and slip have fixed physical divisors.
    priv_scale = [1.]*8+[scales['force_n']]*4+[scales['slip_m_s']]*8
    artifact = dict(schema=1, mode='frozen_before_training', goal_scale_m=10., clip_normalized=10.,
                    pca_sha256=file_sha256(pca_path), identity=pca.metadata['identity'],
                    visual_mean=mean.tolist(), visual_scale=visual_scale.tolist(),
                    state_scale=state_scale, geometry_scale=geometry_scale, privileged_scale=priv_scale,
                    physical_references=scales, abs_p99=quantiles, visual_std_floor=std_floor,
                    provenance=dict(visual_samples=str(samples), visual_sha256=file_sha256(samples),
                                    material_counts=counts.tolist(), motion=sources, seed=123,
                                    policy='Physical divisors: ceil(abs p99 / step)*step, bounded below by stated physical floors; no centering.',
                                    limitations='Motion replay is not balanced across all materials. Visual data cover all seven equally; patch statistics precede grid averaging. No evaluation episodes used.'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream: json.dump(artifact,stream,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps(scales,indent=2)); print(f'[SCALERS] Saved {args.output}')


if __name__ == '__main__': main()
