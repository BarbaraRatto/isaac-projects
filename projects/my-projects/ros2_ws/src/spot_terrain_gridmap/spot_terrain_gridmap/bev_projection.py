#!/usr/bin/env python3
"""
bev_projection.py

Logica PURA (nessuna dipendenza da ROS/rclpy) per proiettare la griglia di
embedding DINOv2, che vive nello spazio dell'immagine (righe/colonne di
patch), in una griglia bird's-eye-view (BEV) nello spazio del mondo attorno
al robot.

Essendo un modulo puro (solo numpy), puo' essere testato in isolamento con
uno script standalone (vedi test/test_bev_projection.py), senza bisogno di
avere ROS o Isaac Sim in esecuzione.

--------------------------------------------------------------------------
Modello geometrico usato: omografia piano-terreno (planar ground assumption)
--------------------------------------------------------------------------
Assumiamo che il terreno immediatamente davanti al robot sia localmente
piano (approssimazione ragionevole per orizzonti di planning brevi, tipici
di un controller locale). Sotto questa ipotesi, la trasformazione tra un
punto nell'immagine e il corrispondente punto sul piano del terreno e'
descritta da un'omografia 3x3, calcolabile a partire da:
  - i parametri intrinseci della camera (fx, fy, cx, cy);
  - la posa estrinseca della camera rispetto al robot (altezza da terra e
    tilt verso il basso).

Questo evita di dover usare la depth camera (coerente con il documento di
contesto, che indica l'elevation layer come opzionale): con la sola
immagine RGB e la geometria nota della camera, possiamo comunque proiettare
correttamente sul piano orizzontale.

Limiti noti (da tenere a mente):
  - Il terreno realmente NON e' sempre piano (buche, sassi, pendenze):
    l'errore di proiezione cresce con l'irregolarita' del terreno e con la
    distanza dal robot.
  - La camera reale ZED2 potrebbe non essere montata esattamente come nel
    placeholder: i parametri vanno aggiornati non appena noti (vedi config
    YAML, camera_pose).
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class CameraIntrinsics:
    """Parametri intrinseci della camera (dal messaggio CameraInfo o da placeholder)."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def K(self) -> np.ndarray:
        """Matrice intrinseca 3x3 standard."""
        return np.array([
            [self.fx, 0.0,      self.cx],
            [0.0,      self.fy, self.cy],
            [0.0,      0.0,      1.0],
        ])


@dataclass
class CameraExtrinsics:
    """
    Posa della camera rispetto al frame del robot (es. base_link).

    Convenzione: x avanti, y sinistra, z su (REP-103, frame robot).
    tilt_deg: rotazione attorno all'asse y del robot, positiva = camera
    inclinata verso il basso (guarda il terreno davanti a se').
    """
    x: float
    y: float
    z: float          # altezza da terra
    tilt_deg: float

    def rotation_matrix_robot_to_camera(self) -> np.ndarray:
        """
        Ruota dal frame robot (x avanti, y sinistra, z su) al frame ottico
        camera (x destra, y giu', z avanti), includendo il tilt verso il
        basso configurato.
        """
        tilt = np.deg2rad(self.tilt_deg)

        # Rotazione di base robot -> ottico (cambio di convenzione assi)
        # robot: x avanti, y sinistra, z su
        # ottico: x destra, y giu', z avanti
        R_base = np.array([
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ])

        # Tilt aggiuntivo verso il basso, attorno all'asse x ottico (destra)
        c, s = np.cos(tilt), np.sin(tilt)
        R_tilt = np.array([
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ])

        return R_tilt @ R_base


def compute_ground_homography(
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
) -> np.ndarray:
    """
    Calcola l'omografia 3x3 che mappa punti immagine (u, v, 1) in punti sul
    piano del terreno (X, Y, 1), espressi nel frame del robot (piano z=0).

    Deriva dalla proiezione prospettica standard ristretta al piano
    z_world = 0, sotto l'ipotesi di terreno localmente piano.
    """
    K = intrinsics.K
    R = extrinsics.rotation_matrix_robot_to_camera()
    t = np.array([extrinsics.x, extrinsics.y, extrinsics.z])

    # Traslazione nel frame camera: posizione dell'origine mondo (robot)
    # vista dalla camera è -R @ t_cam_origin... costruiamo direttamente la
    # matrice di proiezione [R | t_cam] per il piano z_world = 0.
    # Per un punto mondo (X, Y, 0), la sua posizione in frame camera è:
    #   p_cam = R @ ([X, Y, 0] - cam_position_in_robot_frame)
    # Riscriviamo come omografia isolando le colonne X, Y (la colonna Z è
    # ininfluente perché il piano è z=0):
    cam_pos = np.array([extrinsics.x, extrinsics.y, extrinsics.z])

    # Colonne di R corrispondenti a X_world e Y_world, più il termine noto
    # dato dalla posizione della camera.
    Rt = np.column_stack([
        R[:, 0],                       # contributo di X_world
        R[:, 1],                       # contributo di Y_world
        -R @ cam_pos,                  # termine costante (traslazione)
    ])

    H = K @ Rt  # omografia immagine <- mondo (piano z=0)

    # Vogliamo la trasformazione inversa: da immagine a mondo.
    H_inv = np.linalg.inv(H)
    return H_inv


def image_point_to_ground(
    u: float, v: float, H_inv: np.ndarray
) -> tuple:
    """
    Proietta un punto immagine (u, v) sul piano del terreno, restituendo
    coordinate (X, Y) nel frame del robot.
    """
    p_img = np.array([u, v, 1.0])
    p_world_h = H_inv @ p_img
    if abs(p_world_h[2]) < 1e-9:
        # Punto all'orizzonte / dietro la camera: non proiettabile
        return None
    X = p_world_h[0] / p_world_h[2]
    Y = p_world_h[1] / p_world_h[2]
    return X, Y


def project_patch_grid_to_bev(
    embeddings: np.ndarray,
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
    patch_size_px: int,
    image_resized_hw: tuple,
    gridmap_length_x: float,
    gridmap_length_y: float,
    gridmap_resolution: float,
) -> tuple:
    """
    Proietta la griglia di embedding per-patch (spazio immagine) in una
    griglia BEV (spazio mondo, frame robot), tramite omografia piano-terreno.

    Parametri
    ---------
    embeddings : np.ndarray, shape (grid_h, grid_w, embed_dim)
        Embedding DINOv2 per patch, come prodotti da terrain_feature_node.
    intrinsics : CameraIntrinsics
        Parametri intrinseci (da CameraInfo reale o placeholder).
    extrinsics : CameraExtrinsics
        Posa della camera rispetto al robot (da config, placeholder finche'
        non nota con certezza).
    patch_size_px : int
        Dimensione in pixel di una patch DINOv2 (14 per ViT-S/B), riferita
        alla risoluzione EFFETTIVA data in input al modello.
    image_resized_hw : tuple (H, W)
        Risoluzione dell'immagine effettivamente data in input al modello
        (dopo l'eventuale resize del processor HuggingFace).
    gridmap_length_x, gridmap_length_y : float
        Estensione fisica [m] della gridmap BEV attorno al robot.
    gridmap_resolution : float
        Dimensione lato cella [m].

    Ritorna
    -------
    bev_grid : np.ndarray, shape (rows, cols, embed_dim)
        Griglia BEV con embedding aggregati per cella (media delle patch
        che cadono in quella cella; celle senza patch = NaN).
    valid_mask : np.ndarray, shape (rows, cols), dtype=bool
        True per le celle con almeno una patch proiettata.
    """
    grid_h, grid_w, embed_dim = embeddings.shape
    img_h, img_w = image_resized_hw

    H_inv = compute_ground_homography(intrinsics, extrinsics)

    # NOTA: intrinsics e' definita sulla risoluzione originale della camera
    # (es. da CameraInfo). Se l'immagine e' stata ridimensionata prima del
    # forward pass DINOv2, occorre riscalare fx, fy, cx, cy di conseguenza.
    scale_x = img_w / intrinsics.width
    scale_y = img_h / intrinsics.height
    intrinsics_scaled = CameraIntrinsics(
        width=img_w,
        height=img_h,
        fx=intrinsics.fx * scale_x,
        fy=intrinsics.fy * scale_y,
        cx=intrinsics.cx * scale_x,
        cy=intrinsics.cy * scale_y,
    )
    H_inv = compute_ground_homography(intrinsics_scaled, extrinsics)

    # Dimensioni della griglia BEV in celle
    bev_rows = int(round(gridmap_length_x / gridmap_resolution))
    bev_cols = int(round(gridmap_length_y / gridmap_resolution))

    # Accumulatori per media incrementale per cella
    sum_grid = np.zeros((bev_rows, bev_cols, embed_dim), dtype=np.float64)
    count_grid = np.zeros((bev_rows, bev_cols), dtype=np.int32)

    for gy in range(grid_h):
        for gx in range(grid_w):
            # Centro della patch (gx, gy) in coordinate pixel immagine
            u = (gx + 0.5) * patch_size_px
            v = (gy + 0.5) * patch_size_px

            ground_pt = image_point_to_ground(u, v, H_inv)
            if ground_pt is None:
                continue
            X, Y = ground_pt

            # X: distanza in avanti dal robot; Y: laterale (positivo = sinistra)
            # La gridmap e' centrata sul robot: X in [0, length_x] (solo
            # davanti), Y in [-length_y/2, length_y/2].
            if not (0.0 <= X <= gridmap_length_x):
                continue
            if not (-gridmap_length_y / 2.0 <= Y <= gridmap_length_y / 2.0):
                continue

            row = int(X / gridmap_resolution)
            col = int((Y + gridmap_length_y / 2.0) / gridmap_resolution)
            row = min(max(row, 0), bev_rows - 1)
            col = min(max(col, 0), bev_cols - 1)

            sum_grid[row, col] += embeddings[gy, gx]
            count_grid[row, col] += 1

    valid_mask = count_grid > 0
    bev_grid = np.full((bev_rows, bev_cols, embed_dim), np.nan, dtype=np.float64)
    bev_grid[valid_mask] = sum_grid[valid_mask] / count_grid[valid_mask, None]

    return bev_grid.astype(np.float32), valid_mask
