#!/usr/bin/env python3
"""
test_bev_projection.py

Script di test STANDALONE (nessun ROS richiesto) per validare la logica di
proiezione geometrica image -> BEV, usando dati sintetici al posto di
un'immagine e di embedding DINOv2 reali.

Utile per:
  - verificare che la geometria (omografia) sia corretta, PRIMA di avere
    la camera integrata in Isaac Sim (non ancora disponibile, vedi
    documento di contesto, Sez. 1.2);
  - fare esperimenti rapidi cambiando camera_pose (altezza, tilt) e vedere
    come cambia la copertura della gridmap risultante, per aiutare a
    decidere il posizionamento reale della camera.

Esecuzione:
    cd spot_terrain_gridmap
    python3 test/test_bev_projection.py

Non richiede rclpy: solo numpy (e opzionalmente matplotlib, se installato,
per una visualizzazione grafica della copertura).
"""

import sys
import os
import numpy as np

# Permette di importare il modulo senza installare il package ROS
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from spot_terrain_gridmap.bev_projection import (
    CameraIntrinsics,
    CameraExtrinsics,
    project_patch_grid_to_bev,
)


def make_synthetic_embeddings(grid_h=34, grid_w=46, embed_dim=8):
    """
    Crea una griglia di embedding sintetici (NON da DINOv2 reale), solo per
    testare che la geometria di proiezione funzioni. embed_dim ridotto a 8
    (invece di 384) solo per velocita' del test, la logica e' identica.
    """
    rng = np.random.default_rng(42)
    return rng.normal(size=(grid_h, grid_w, embed_dim)).astype(np.float32)


def main():
    # Parametri placeholder, identici al config YAML di default
    intrinsics = CameraIntrinsics(
        width=644,   # coerente con grid_w=46 * patch_size=14
        height=476,  # coerente con grid_h=34 * patch_size=14
        fx=525.0, fy=525.0, cx=322.0, cy=238.0,
    )
    extrinsics = CameraExtrinsics(x=0.35, y=0.0, z=0.30, tilt_deg=20.0)

    patch_size = 14
    grid_h, grid_w = 34, 46
    embeddings = make_synthetic_embeddings(grid_h, grid_w, embed_dim=8)

    bev_grid, valid_mask = project_patch_grid_to_bev(
        embeddings=embeddings,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        patch_size_px=patch_size,
        image_resized_hw=(grid_h * patch_size, grid_w * patch_size),
        gridmap_length_x=6.0,
        gridmap_length_y=6.0,
        gridmap_resolution=0.1,
    )

    n_valid = int(valid_mask.sum())
    n_total = valid_mask.size
    print(f'Gridmap BEV shape: {bev_grid.shape}')
    print(f'Celle valide: {n_valid}/{n_total} ({100.0 * n_valid / n_total:.1f}%)')

    # Sanity check di base: con tilt 20 deg e altezza 0.30m ci aspettiamo
    # che la copertura NON sia ne' 0% (proiezione rotta) ne' 100%
    # (impossibile: il campo visivo e' limitato).
    assert 0 < n_valid < n_total, (
        'Copertura sospetta: verificare i parametri di camera_pose o la '
        'logica di proiezione.'
    )
    print('Sanity check superato: la copertura e\' in un range plausibile.')

    # Visualizzazione opzionale (solo se matplotlib e' disponibile)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        plt.imshow(valid_mask, origin='lower', cmap='Greens')
        plt.title('Copertura gridmap BEV (celle raggiunte dalla proiezione)')
        plt.xlabel('cella Y (laterale)')
        plt.ylabel('cella X (avanti)')
        out_path = os.path.join(os.path.dirname(__file__), 'bev_coverage_preview.png')
        plt.savefig(out_path, dpi=120)
        print(f'Anteprima salvata in: {out_path}')
    except ImportError:
        print('(matplotlib non installato: skip anteprima grafica, solo controllo numerico)')


if __name__ == '__main__':
    main()
