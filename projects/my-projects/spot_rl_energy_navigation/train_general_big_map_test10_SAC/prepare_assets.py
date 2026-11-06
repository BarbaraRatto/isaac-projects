"""Verify bundled immutable assets and build only missing offline mission artifacts."""
import json
from pathlib import Path
import sys
from dataclasses import asdict

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from train_general_big_map_test10_SAC.artifacts import source_identity
from train_general_big_map_test10_SAC.calibration import pca_digest,validate_calibration
from train_general_big_map_test10_SAC.perception import FrozenPCA,file_sha256
from train_general_big_map_test10_SAC.config import DEFAULT_PCA,MISSION_CFG,experiment_signature


def main():
    meta=json.loads((HERE/'artifacts/map_metadata.json').read_text())
    if file_sha256(HERE/'maps/real_terrains_test10.usd')!=meta['usd_sha256']:
        raise ValueError('Bundled map changed; do not regenerate terrain during a run')
    missions=HERE/'artifacts/missions.npz'
    if not missions.exists():
        from train_general_big_map_test10_SAC.prepare_missions import build
        build(missions)
    report=json.loads(missions.with_suffix('.json').read_text())
    contract=json.loads(json.dumps(asdict(MISSION_CFG)))
    if report.get('build_contract')!=contract:
        raise ValueError('Mission bank/config mismatch; deliberately build a new experiment instead')
    identity=source_identity()
    manifest_path=HERE/'artifacts/manifest.json'
    pca=FrozenPCA.load(DEFAULT_PCA)
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())['identity']!=identity:
            raise ValueError('Existing frozen manifest differs; assets are never silently overwritten')
    else:
        manifest=dict(schema=1,identity=identity,signature=experiment_signature(),
                      pca_source='artifacts/pca32_big_map.npz',pca_sha256=file_sha256(DEFAULT_PCA),
                      pca_transfer=dict(target_identity=identity,calibration_identity=pca.metadata['identity'],
                                        array_sha256=pca_digest(pca)),
                      provenance='Same immutable map/PCA as test9; new mixed mission bank and fresh final-cell scalers.')
        manifest_path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    validate_calibration(pca,identity)
    print('[TEST10] Frozen map, seven-material PCA and mixed missions verified; no simulation/training started.')


if __name__=='__main__':main()
