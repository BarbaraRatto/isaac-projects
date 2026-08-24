#!/usr/bin/env python3
"""
gridmap_builder.py

Costruisce un messaggio grid_map_msgs/msg/GridMap a partire dall'array
numpy BEV prodotto da bev_projection.py.

Nota sul formato grid_map_msgs/GridMap:
  - header: header standard (timestamp, frame_id) - attenzione: sta
    direttamente su GridMap, NON dentro info (GridMapInfo non ha header)
  - info.resolution: dimensione lato cella [m]
  - info.length_x, info.length_y: estensione fisica della mappa [m]
  - info.pose: posa del centro della mappa rispetto al frame_id
  - layers: lista di nomi dei layer (es. ["terrain_features_0", ...])
  - data: lista di std_msgs/Float32MultiArray, uno per layer

Per un embedding a N dimensioni (es. 384 per ViT-S), il modo standard di
rappresentarlo in GridMap multi-layer e' creare N layer scalari distinti
(uno per componente dell'embedding), dato che GridMap non supporta
nativamente celle vettoriali. Con nomi tipo "terrain_features_000",
"terrain_features_001", ... Questo e' verboso ma e' il modo idiomatico di
usare grid_map_msgs; l'alternativa (serializzare il vettore come bytes in
un layer custom) romperebbe la compatibilita' con i tool standard (RVIZ
grid_map plugin, grid_map_core, ecc.).
"""

import numpy as np
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
from geometry_msgs.msg import Pose


def build_gridmap_msg(
    bev_grid: np.ndarray,
    valid_mask: np.ndarray,
    resolution: float,
    frame_id: str,
    stamp,
    layer_prefix: str = "terrain_features",
) -> GridMap:
    """
    Costruisce il messaggio GridMap multi-layer dall'array BEV.

    Parametri
    ---------
    bev_grid : np.ndarray, shape (rows, cols, embed_dim)
        Griglia BEV con embedding per cella (NaN dove non valido).
    valid_mask : np.ndarray, shape (rows, cols)
        Maschera booleana delle celle valide (non usata direttamente qui,
        i NaN in bev_grid gia' codificano l'invalidita' - grid_map_core
        tratta i NaN come "cella vuota", comportamento standard).
    resolution : float
        Dimensione lato cella [m].
    frame_id : str
        Frame rispetto al quale e' espressa la mappa (es. "base_link").
    stamp :
        Timestamp ROS (rclpy.time.Time o builtin_interfaces/Time) da usare
        nell'header.
    layer_prefix : str
        Prefisso per i nomi dei layer scalari generati.

    Ritorna
    -------
    msg : grid_map_msgs.msg.GridMap
    """
    rows, cols, embed_dim = bev_grid.shape

    msg = GridMap()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.info.resolution = float(resolution)
    msg.info.length_x = float(rows * resolution)
    msg.info.length_y = float(cols * resolution)

    # Posa del centro della gridmap rispetto al frame_id. La gridmap e'
    # costruita con X in [0, length_x] (solo davanti al robot) e Y centrata
    # su 0 (vedi bev_projection.project_patch_grid_to_bev), quindi il
    # centro della mappa e' spostato in avanti di length_x/2 rispetto
    # all'origine del robot.
    pose = Pose()
    pose.position.x = msg.info.length_x / 2.0
    pose.position.y = 0.0
    pose.position.z = 0.0
    pose.orientation.w = 1.0
    msg.info.pose = pose

    msg.layers = [f"{layer_prefix}_{i:03d}" for i in range(embed_dim)]
    msg.basic_layers = []  # nessun layer "di base" (es. elevation) qui

    for i in range(embed_dim):
        layer_data = bev_grid[:, :, i]

        arr = Float32MultiArray()
        arr.layout = MultiArrayLayout()
        arr.layout.dim = [
            MultiArrayDimension(label="column_index", size=cols, stride=rows * cols),
            MultiArrayDimension(label="row_index", size=rows, stride=rows),
        ]
        arr.layout.data_offset = 0

        # grid_map_msgs si aspetta i dati in row-major a partire
        # dall'angolo INIZIALE della mappa; usiamo flatten in ordine
        # compatibile con grid_map_core (column-major, come da convenzione
        # del pacchetto grid_map -> flatten con order='F').
        arr.data = layer_data.astype(np.float32).flatten(order="F").tolist()

        msg.data.append(arr)

    return msg


def build_elevation_layer_placeholder(rows: int, cols: int) -> np.ndarray:
    """
    Placeholder per un futuro layer 'elevation' (opzionale, vedi Sez. 4.2.5
    del documento di contesto). Al momento non implementato: richiede una
    depth camera (ZED2 depth o depth simulata in Isaac Sim), non ancora
    disponibile. Ritorna una griglia di NaN (nessun dato).
    """
    return np.full((rows, cols), np.nan, dtype=np.float32)
