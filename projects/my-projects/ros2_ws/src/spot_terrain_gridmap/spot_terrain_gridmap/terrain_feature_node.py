#!/usr/bin/env python3
"""
terrain_feature_node.py

Nodo ROS 2 che implementa l'intera pipeline immagine -> gridmap del terreno
descritta in Sez. 4.2 del documento di contesto del progetto:

    /camera/rgb         ---\
    /camera/depth       ----+--> DINOv2 (frozen) --> proiezione BEV con Depth --> /terrain_gridmap
    /camera/camera_info --/
    /tf (ZED_X -> base_link)

Fasi aggiornate:
  1. Riceve immagini RGB, Depth e Camera Info sincronizzate.
  2. Estrae l'embedding continuo (RGB) tramite DINOv2 frozen.
  3. Cerca la trasformata TF in tempo reale tra 'base_link' e 'ZED_X' usando il timestamp dell'immagine.
  4. Mappa i centri delle patch sull'immagine di profondità e li deproietta in 3D.
  5. Applica la rototraslazione completa (Matrice di Rotazione 3x3 + Traslazione) letta da TF.
  6. Costruisce e pubblica il messaggio grid_map_msgs/GridMap risultante.
"""

import os
import sys

DINO_VENV_SITE_PACKAGES = os.environ.get(
    'DINO_VENV_SITE_PACKAGES',
    os.path.expanduser(
        '~/work/barbara/venvs/dino_env/lib/python3.10/site-packages'
    ),
)
if os.path.isdir(DINO_VENV_SITE_PACKAGES) and DINO_VENV_SITE_PACKAGES not in sys.path:
    sys.path.insert(0, DINO_VENV_SITE_PACKAGES)

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters
from sklearn.decomposition import PCA

import tf2_ros
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import TransformStamped

from transformers import AutoImageProcessor, AutoModel

from spot_terrain_gridmap.bev_projection import CameraIntrinsics
from spot_terrain_gridmap.gridmap_builder import build_gridmap_msg
from grid_map_msgs.msg import GridMap


def get_transform_matrix(transform_stamped: TransformStamped):
    """
    Converte un messaggio TransformStamped di ROS in una matrice di rotazione 3x3
    e un vettore di traslazione 1x3, senza richiedere librerie esterne addizionali.
    """
    trans = transform_stamped.transform.translation
    rot = transform_stamped.transform.rotation
    
    t = np.array([trans.x, trans.y, trans.z])
    x, y, z, w = rot.x, rot.y, rot.z, rot.w
    
    # Costruiamo la matrice di rotazione dal quaternione
    R = np.array([
        [1 - 2*y**2 - 2*z**2,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    
    return R, t


def project_patch_grid_to_bev_with_depth(
    embeddings: np.ndarray,
    depth_image: np.ndarray,
    intrinsics: CameraIntrinsics,
    tf_transform: TransformStamped,
    patch_size_px: int,
    image_resized_hw: tuple,
    gridmap_length_x: float,
    gridmap_length_y: float,
    gridmap_resolution: float
):
    """
    Mappa le patch DINO ai valori reali di profondità, deproietta in 3D nello
    spazio camera, e rototrasla dinamicamente nel robot frame usando TF2.
    """
    grid_h, grid_w, embed_dim = embeddings.shape
    depth_h, depth_w = depth_image.shape
    resized_h, resized_w = image_resized_hw

    # Fattori di scala per mappare i centri delle patch all'immagine depth
    scale_y = depth_h / resized_h
    scale_x = depth_w / resized_w

    v_idx, u_idx = np.indices((grid_h, grid_w))
    u_centers = (u_idx * patch_size_px + patch_size_px / 2.0) * scale_x
    v_centers = (v_idx * patch_size_px + patch_size_px / 2.0) * scale_y

    u_centers = np.clip(u_centers.astype(int), 0, depth_w - 1)
    v_centers = np.clip(v_centers.astype(int), 0, depth_h - 1)

    # Campiona la profondità
    Z = depth_image[v_centers, u_centers]
    
    # Filtro validità depth (es. compreso tra 10 cm e 15 metri)
    valid_depth = (Z > 0.1) & (Z < 15.0)

    # 1. Deproiezione Pinhole (Camera Optical Frame)
    X_cam = (u_centers - intrinsics.cx) * Z / intrinsics.fx
    Y_cam = (v_centers - intrinsics.cy) * Z / intrinsics.fy
    Z_cam = Z

    # Ricomponiamo in un array N x 3
    pts_cam = np.stack([X_cam.ravel(), Y_cam.ravel(), Z_cam.ravel()], axis=1)

    # 2. Rototraslazione al Robot Base Frame tramite TF
    R, t = get_transform_matrix(tf_transform)
    
    # Trasformazione geometrica p_base = R * p_cam + t
    pts_base = pts_cam @ R.T + t

    # Riorganizziamo la struttura della griglia originale
    X_base = pts_base[:, 0].reshape((grid_h, grid_w))
    Y_base = pts_base[:, 1].reshape((grid_h, grid_w))

    # 3. Discretizzazione per la BEV Grid
    grid_cells_x = int(gridmap_length_x / gridmap_resolution)
    grid_cells_y = int(gridmap_length_y / gridmap_resolution)

    bev_grid = np.zeros((grid_cells_x, grid_cells_y, embed_dim), dtype=np.float32)
    valid_mask = np.zeros((grid_cells_x, grid_cells_y), dtype=bool)

    # Attenzione: i calcoli BEV assumono X in avanti, Y a sinistra.
    idx_x = ((gridmap_length_x / 2.0 - X_base) / gridmap_resolution).astype(int)
    idx_y = ((gridmap_length_y / 2.0 - Y_base) / gridmap_resolution).astype(int)

    # Mantiene solo le patch valide e interne ai bordi della grid map
    in_bounds = (idx_x >= 0) & (idx_x < grid_cells_x) & (idx_y >= 0) & (idx_y < grid_cells_y)
    mask = valid_depth & in_bounds

    # Assegnazione alla mappa
    flat_x = idx_x[mask]
    flat_y = idx_y[mask]
    flat_emb = embeddings[mask]

    bev_grid[flat_x, flat_y] = flat_emb
    valid_mask[flat_x, flat_y] = True

    return bev_grid, valid_mask


class TerrainFeatureNode(Node):
    def __init__(self):
        super().__init__('terrain_feature_node')

        self._declare_parameters()
        self._read_parameters()
        self._setup_model()
        self._setup_ros_io()

        self._frame_count = 0
        self.get_logger().info('terrain_feature_node pronto e in attesa di messaggi sincronizzati e TF...')

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _declare_parameters(self):
        self.declare_parameter('image_topic', '/camera/rgb')
        self.declare_parameter('depth_topic', '/camera/depth')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('gridmap_topic', '/terrain_gridmap')
        self.declare_parameter('dino_pca_image_topic', '/dino_features_pca')
        self.declare_parameter('debug_valid_gridmap_topic', '/debug_valid_gridmap')
        
        # Sostituiamo i vecchi 'camera_pose' introducendo il camera_frame
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'ZED_X')

        self.declare_parameter('model_name', 'facebook/dinov2-small')
        self.declare_parameter('device', 'cuda')

        self.declare_parameter('gridmap.length_x', 6.0)
        self.declare_parameter('gridmap.length_y', 6.0)
        self.declare_parameter('gridmap.resolution', 0.1)

        self.declare_parameter('log_every_n', 30)

    def _read_parameters(self):
        gp = self.get_parameter

        self.image_topic = gp('image_topic').value
        self.depth_topic = gp('depth_topic').value
        self.camera_info_topic = gp('camera_info_topic').value
        self.gridmap_topic = gp('gridmap_topic').value
        self.dino_pca_image_topic = gp('dino_pca_image_topic').value
        self.debug_valid_gridmap_topic = gp('debug_valid_gridmap_topic').value
        
        self.robot_base_frame = gp('robot_base_frame').value
        self.camera_frame = gp('camera_frame').value

        self.model_name = gp('model_name').value
        self.device_str = gp('device').value

        self.gridmap_length_x = gp('gridmap.length_x').value
        self.gridmap_length_y = gp('gridmap.length_y').value
        self.gridmap_resolution = gp('gridmap.resolution').value

        self.log_every_n = gp('log_every_n').value

    def _setup_model(self):
        if self.device_str == 'cuda' and not torch.cuda.is_available():
            self.get_logger().warn(
                'CUDA richiesta ma non disponibile: fallback su CPU (sara\' lento).'
            )
            self.device_str = 'cpu'
        self.device = torch.device(self.device_str)

        self.get_logger().info(f'Caricamento modello {self.model_name} su {self.device}...')
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.patch_size = self.model.config.patch_size
        self.get_logger().info(f'Patch size del modello: {self.patch_size}px')

    def _setup_ros_io(self):
        self.bridge = CvBridge()
        
        # Inizializzazione TF2 Listener
        # ATTENZIONE: usiamo spin_thread=True per far girare l'aggiornamento
        # del buffer TF in un thread separato. In questo modo l'inferenza pesante
        # di DINOv2 non blocca la ricezione dei nuovi messaggi TF!
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.sub_rgb = message_filters.Subscriber(self, Image, self.image_topic, qos_profile=10)
        self.sub_depth = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=10)
        self.sub_info = message_filters.Subscriber(self, CameraInfo, self.camera_info_topic, qos_profile=10)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth, self.sub_info],
            queue_size=10,      # <-- AUMENTATO (per gestire i 42 Hz) e poi DIMINUITO 
            slop=0.1             # <-- AUMENTATO (per tollerare fino a mezzo secondo di sfasamento) e poi DIMINUITO 
        )
        self.sync.registerCallback(self.synced_callback)

        self.pub_gridmap = self.create_publisher(GridMap, self.gridmap_topic, 1)
        self.pub_dino_pca = self.create_publisher(Image, self.dino_pca_image_topic, 1)
        self.pub_debug_valid_gridmap = self.create_publisher(OccupancyGrid, self.debug_valid_gridmap_topic, 1)

        self.get_logger().info(
            f'In ascolto sui topic sincroni e TF ({self.camera_frame} -> {self.robot_base_frame})'
        )

    # ------------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------------
    def synced_callback(self, msg_rgb: Image, msg_depth: Image, msg_info: CameraInfo):
        # 1. Decodifica immagini e calcolo feature (indipendente dalla TF)
        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(msg_rgb, desired_encoding='rgb8')
            cv_depth = self.bridge.imgmsg_to_cv2(msg_depth, desired_encoding='passthrough')
            
            if cv_depth.dtype == np.uint16:
                depth_m = cv_depth.astype(np.float32) / 1000.0
            else:
                depth_m = cv_depth.astype(np.float32)

        except Exception as e:
            self.get_logger().error(f'Errore conversione immagine: {e}')
            return

        k = msg_info.k
        intrinsics = CameraIntrinsics(
            width=msg_info.width,
            height=msg_info.height,
            fx=k[0],
            fy=k[4],
            cx=k[2],
            cy=k[5],
        )

        embeddings, grid_h, grid_w, resized_hw = self._extract_features(cv_rgb)

        try:
            # --- DEBUG PCA ---
            # embeddings shape: (grid_h, grid_w, embed_dim)
            embed_dim = embeddings.shape[-1]
            flat_embeddings = embeddings.reshape(-1, embed_dim)
            
            pca = PCA(n_components=3)
            pca_features = pca.fit_transform(flat_embeddings)
            
            # Normalizzazione su [0, 255] per visualizzazione RGB
            pca_features_min = pca_features.min(axis=0)
            pca_features_max = pca_features.max(axis=0)
            pca_features_norm = (pca_features - pca_features_min) / (pca_features_max - pca_features_min + 1e-6)
            pca_image_rgb = (pca_features_norm * 255.0).astype(np.uint8)
            
            # Reshape alla risoluzione della griglia patch DINOv2
            pca_image_rgb = pca_image_rgb.reshape(grid_h, grid_w, 3)
            
            # Pubblica l'immagine
            msg_dino_pca = self.bridge.cv2_to_imgmsg(pca_image_rgb, encoding='rgb8')
            msg_dino_pca.header = msg_rgb.header
            self.pub_dino_pca.publish(msg_dino_pca)
            # -----------------
        except Exception as e:
            self.get_logger().warn(f"Impossibile calcolare o pubblicare PCA: {e}", throttle_duration_sec=5.0)

        # 2. Cerchiamo dinamicamente la trasformata usando l'albero TF
        try:
            # Raccogliamo la trasformazione al timestamp del frame (lasciando 0.5s di tolleranza all'albero)
            tf_transform = self.tf_buffer.lookup_transform(
                self.robot_base_frame,
                self.camera_frame,
                msg_rgb.header.stamp,
                rclpy.duration.Duration(seconds=0.5)
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            # Demoted to debug since the fallback works perfectly for Isaac Sim desync
            self.get_logger().debug('Sfasamento temporale TF rilevato (normale in Isaac Sim). Uso la TF più recente.', throttle_duration_sec=2.0)
            try:
                # Fallback: usiamo l'ultima trasformata disponibile nell'albero
                tf_transform = self.tf_buffer.lookup_transform(
                    self.robot_base_frame,
                    self.camera_frame,
                    rclpy.time.Time()
                )
            except Exception as e2:
                self.get_logger().error(f'Trasformata non trovata neanche come ultima disponibile: {e2}', throttle_duration_sec=2.0)
                return

        # 3. Costruzione della Gridmap (richiede TF)
        # Usiamo il tf_transform dinamico
        bev_grid, valid_mask = project_patch_grid_to_bev_with_depth(
            embeddings=embeddings,
            depth_image=depth_m,
            intrinsics=intrinsics,
            tf_transform=tf_transform,
            patch_size_px=self.patch_size,
            image_resized_hw=resized_hw,
            gridmap_length_x=self.gridmap_length_x,
            gridmap_length_y=self.gridmap_length_y,
            gridmap_resolution=self.gridmap_resolution,
        )

        gridmap_msg = build_gridmap_msg(
            bev_grid=bev_grid,
            valid_mask=valid_mask,
            resolution=self.gridmap_resolution,
            frame_id=self.robot_base_frame,
            stamp=msg_rgb.header.stamp,
        )
        self.pub_gridmap.publish(gridmap_msg)

        # --- DEBUG VALID GRIDMAP (OccupancyGrid) ---
        occ_msg = OccupancyGrid()
        occ_msg.header.stamp = msg_rgb.header.stamp
        occ_msg.header.frame_id = self.robot_base_frame
        occ_msg.info.resolution = self.gridmap_resolution
        occ_msg.info.height = valid_mask.shape[0]
        occ_msg.info.width = valid_mask.shape[1]
        
        # Origin is the bottom-left corner of the grid
        occ_msg.info.origin.position.x = -self.gridmap_length_x / 2.0
        occ_msg.info.origin.position.y = -self.gridmap_length_y / 2.0
        occ_msg.info.origin.orientation.w = 1.0
        
        # Flip per allineare gli indici (0,0) di valid_mask (che è avanti-sinistra) 
        # con (0,0) dell'OccupancyGrid (che è dietro-destra rispetto al robot)
        flipped_mask = np.flip(valid_mask, axis=(0, 1))
        
        occ_data = np.zeros_like(flipped_mask, dtype=np.int8)
        occ_data[flipped_mask] = 100
        
        occ_msg.data = occ_data.flatten(order='C').tolist()
        self.pub_debug_valid_gridmap.publish(occ_msg)
        # ---------------------------

        self._frame_count += 1
        if self._frame_count % self.log_every_n == 0:
            n_valid = int(valid_mask.sum())
            n_total = valid_mask.size
            self.get_logger().info(
                f'Frame {self._frame_count}: embeddings {embeddings.shape}, '
                f'gridmap BEV {bev_grid.shape[:2]} '
                f'({n_valid}/{n_total} celle valide, '
                f'{100.0 * n_valid / n_total:.1f}%)'
            )

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _extract_features(self, image_rgb: np.ndarray):
        inputs = self.processor(images=image_rgb, return_tensors='pt').to(self.device)
        outputs = self.model(**inputs)

        patch_tokens = outputs.last_hidden_state[0, 1:, :]

        pixel_values = inputs['pixel_values']
        img_h, img_w = pixel_values.shape[-2], pixel_values.shape[-1]
        grid_h = img_h // self.patch_size
        grid_w = img_w // self.patch_size

        embed_dim = patch_tokens.shape[-1]
        embeddings = patch_tokens.reshape(grid_h, grid_w, embed_dim)

        return embeddings.cpu().numpy(), grid_h, grid_w, (img_h, img_w)


def main(args=None):
    rclpy.init(args=args)
    node = TerrainFeatureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
