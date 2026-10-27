"""Inspect existing DINO features; no simulator, PCA fitting or policy changes."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parents[1]
DATA = EXPERIMENT.parent/'extracted_features_comparison_v2/data/features/pilot_01'


def metrics(x, mean, components):
    x = x.astype(np.float64)
    centered = x-x.mean(axis=0)
    variance = np.square(centered).sum()
    offset = x-mean
    energy = np.square(offset).sum()
    return dict(samples=len(x),
                centered_variance_retained=float(np.square(centered@components.T).sum()/variance),
                calibration_center_retained=float(np.square(offset@components.T).sum()/energy))


def main():
    path = EXPERIMENT/'artifacts/pca32.npz'
    with np.load(path, allow_pickle=False) as pca:
        mean = pca['mean'].astype(np.float64)
        components = pca['components'].astype(np.float64)
        meta = json.loads(str(pca['metadata'].item()))
        original_variance = float(pca['explained'].sum())
    provenance = json.loads((DATA/'provenance.json').read_text())
    hashes = provenance['manifest']['files']
    for name, expected in meta['identity']['dino'].items():
        assert hashes['models/dinov2/'+name] == expected, name
    config = json.loads((DATA/'config.json').read_text())
    assert config['dino']['input_size'] == meta['identity']['input_size']
    np.testing.assert_allclose(components@components.T, np.eye(32), atol=1e-6)
    by_label = defaultdict(list)
    rows = [json.loads(line) for line in (DATA/'frames.jsonl').read_text().splitlines() if line.strip()]
    for row in rows:
        with np.load(DATA/row['features'], allow_pickle=False) as features:
            patches = features['patches']
            assert patches.ndim == 2 and patches.shape[1] == 384
            if len(patches):
                by_label[row['label']].append(patches.copy())
    arrays = {label: np.concatenate(values) for label, values in by_label.items()}
    result = dict(pca_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  original_metadata=meta, original_variance_retained=original_variance,
                  diagnostic_dataset=str(DATA.relative_to(EXPERIMENT.parent)),
                  frames=len({row['source_frame_id'] for row in rows}), records=len(rows),
                  per_material={label: metrics(x, mean, components) for label, x in arrays.items()},
                  all_materials=metrics(np.concatenate(list(arrays.values())), mean, components),
                  limitation='Existing pilot on donor cells from real_terrains.usd, not current real_terrains_v2. '
                             'No refitting; all acquisition splits pooled for descriptive audit only. '
                             'Patch samples are correlated; these are not classification or navigation scores.')
    (HERE/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    for label, item in result['per_material'].items():
        print(f"{label}: n={item['samples']}, centered variance retained={item['centered_variance_retained']:.1%}")
    print('Aggregate:', result['all_materials'])


if __name__ == '__main__':
    main()
