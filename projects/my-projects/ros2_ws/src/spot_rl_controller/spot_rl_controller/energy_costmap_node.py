#!/usr/bin/env python3
"""
energy_costmap_node.py

Nodo ROS2 che fa da bridge tra la gridmap DINOv2 (384 layer) prodotta da
feature_extraction_node e il formato compresso (16+1 layer) consumato
dall'agente RL.

Pipeline:
  /terrain_gridmap (GridMap, 384 layer f_0..f_383)
       +
  /energy/current_consumption (EnergyEstimate)
       |
       v
  PCA online incrementale (384 → N_PCA_COMPONENTS=16 dim per cella)
       +
  associazione cella → consumo energetico medio osservato (1 dim)
       |
       v
  /energy_costmap_tensor  (std_msgs/Float32MultiArray, shape [17, 50, 50])
  /energy_costmap_debug   (nav_msgs/OccupancyGrid, per RViz)

Il nodo NON usa lookup table: impara la corrispondenza feature→consumo
direttamente dalle misurazioni di /joint_states → EnergyEstimationNode.
"""

from __future__ import annotations

import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from grid_map_msgs.msg import GridMap
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from energy_msgs.msg import EnergyEstimate


class EnergyCostmapNode(Node):
    """
    Nodo che comprime la gridmap DINOv2 e la arricchisce con il layer
    di costo energetico osservato per cella.
    """

    def __init__(self) -> None:
        super().__init__("energy_costmap_node")

        # ── Dichiarazione parametri ────────────────────────────────────────
        self.declare_parameter("gridmap_topic", "/terrain_gridmap")
        self.declare_parameter("energy_topic", "/energy/current_consumption")
        self.declare_parameter("costmap_out_topic", "/energy_costmap_tensor")
        self.declare_parameter("costmap_debug_topic", "/energy_costmap_debug")
        self.declare_parameter("n_pca_components", 16)
        self.declare_parameter("pca_min_samples", 200)
        self.declare_parameter("energy_alpha", 0.1)   # EMA per l'aggiornamento per cella
        self.declare_parameter("p_max_normalization", 200.0)  # [W] per normalizzare il costo

        # ── Lettura parametri ──────────────────────────────────────────────
        gridmap_topic    = self.get_parameter("gridmap_topic").value
        energy_topic     = self.get_parameter("energy_topic").value
        costmap_out      = self.get_parameter("costmap_out_topic").value
        costmap_debug    = self.get_parameter("costmap_debug_topic").value
        self.n_pca       = self.get_parameter("n_pca_components").value
        self.pca_min_s   = self.get_parameter("pca_min_samples").value
        self.alpha       = self.get_parameter("energy_alpha").value
        self.p_max       = self.get_parameter("p_max_normalization").value

        # ── Stato interno ──────────────────────────────────────────────────
        self._lock = threading.Lock()

        # Ultimo consumo energetico ricevuto (media mobile)
        self._avg_power: float = 0.0
        self._inst_power: float = 0.0

        # PCA online: inizialmente None, si inizializza appena si hanno abbastanza campioni
        self._pca = None
        self._feature_buffer: list[np.ndarray] = []   # buffer per il fit iniziale
        self._pca_ready = False

        # Griglia di costo energetico per cella: None → non ancora inizializzata
        # Forma: (n_rows, n_cols) — EMA del consumo medio per cella
        self._energy_cost_grid: np.ndarray | None = None
        self._energy_hit_count: np.ndarray | None = None

        # Dimensioni gridmap (riempite al primo messaggio GridMap)
        self._n_rows: int | None = None
        self._n_cols: int | None = None
        self._resolution: float | None = None
        self._center_x: float = 0.0
        self._center_y: float = 0.0
        self._n_dino_layers: int | None = None

        # ── Subscriber ────────────────────────────────────────────────────
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._sub_gridmap = self.create_subscription(
            GridMap, gridmap_topic, self._on_gridmap, 1,
        )
        self._sub_energy = self.create_subscription(
            EnergyEstimate, energy_topic, self._on_energy, qos_best_effort,
        )

        # ── Publisher ─────────────────────────────────────────────────────
        self._pub_tensor = self.create_publisher(Float32MultiArray, costmap_out, 1)
        self._pub_debug  = self.create_publisher(OccupancyGrid, costmap_debug, 1)

        self.get_logger().info(
            f"energy_costmap_node avviato — "
            f"input: '{gridmap_topic}', '{energy_topic}', "
            f"PCA: 384→{self.n_pca} (min {self.pca_min_s} campioni)"
        )

    # ======================================================================
    # Callback: consumo energetico
    # ======================================================================
    def _on_energy(self, msg: EnergyEstimate) -> None:
        """Aggiorna la stima di potenza corrente."""
        with self._lock:
            self._avg_power  = msg.average_power
            self._inst_power = msg.instantaneous_power

    # ======================================================================
    # Callback: gridmap DINOv2
    # ======================================================================
    def _on_gridmap(self, msg: GridMap) -> None:
        """
        Elabora la gridmap DINOv2:
          1. Estrae il tensor (n_rows, n_cols, 384) dalla GridMap
          2. Aggiorna la PCA online
          3. Proietta le feature 384→16 dim
          4. Aggiorna il layer di costo energetico per cella
          5. Pubblica il tensor compresso (17, n_rows, n_cols)
        """
        # ── 0. Lettura metadati (solo al primo messaggio) ──────────────────
        with self._lock:
            if self._n_rows is None:
                self._n_dino_layers = len(msg.layers)
                if self._n_dino_layers == 0:
                    self.get_logger().warn("GridMap ricevuta senza layer, skip.")
                    return

                # Ricava n_rows, n_cols dal primo layer
                first = msg.data[0]
                n_c_dim = first.layout.dim[0].size  # colonne
                n_r_dim = first.layout.dim[1].size  # righe
                self._n_rows = n_r_dim
                self._n_cols = n_c_dim
                self._resolution = msg.info.resolution

                # Inizializza griglia di costo energetico
                self._energy_cost_grid = np.zeros(
                    (self._n_rows, self._n_cols), dtype=np.float32
                )
                self._energy_hit_count = np.zeros(
                    (self._n_rows, self._n_cols), dtype=np.int32
                )
                self.get_logger().info(
                    f"Gridmap inizializzata: {self._n_rows}×{self._n_cols} celle, "
                    f"risoluzione {self._resolution} m, "
                    f"{self._n_dino_layers} layer DINOv2"
                )

            n_rows        = self._n_rows
            n_cols        = self._n_cols
            n_layers      = self._n_dino_layers
            avg_power     = self._avg_power
            inst_power    = self._inst_power

        # ── 1. Estrazione tensor (n_rows, n_cols, n_layers) dalla GridMap ──
        # Ogni layer è memorizzato come Float32MultiArray in column-major (Fortran) order
        self._center_x = msg.info.pose.position.x
        self._center_y = msg.info.pose.position.y

        feature_tensor = np.full(
            (n_rows, n_cols, n_layers), np.nan, dtype=np.float32
        )
        for k, layer_arr in enumerate(msg.data):
            raw = np.array(layer_arr.data, dtype=np.float32)
            if raw.size == n_rows * n_cols:
                # Ricostruzione da column-major (Fortran order)
                feature_tensor[:, :, k] = raw.reshape((n_rows, n_cols), order='F')

        # ── 2. Maschera delle celle valide (senza NaN) ─────────────────────
        valid_mask = ~np.isnan(feature_tensor[:, :, 0])  # (n_rows, n_cols)
        valid_flat = feature_tensor[valid_mask]            # (N_valid, 384)

        if valid_flat.shape[0] == 0:
            self.get_logger().warn(
                "Nessuna cella valida nella gridmap, skip.",
                throttle_duration_sec=5.0,
            )
            return

        # ── 3. Aggiornamento PCA online ────────────────────────────────────
        projected = self._update_pca_and_project(valid_flat)  # (N_valid, n_pca) o None

        # ── 4. Aggiornamento layer costo energetico per cella ─────────────
        # Il consumo attuale viene associato alle celle valide correnti:
        # le celle dove il robot si trova ora "costano" avg_power.
        # Uso una EMA (Exponential Moving Average) per smorzare il rumore.
        current_power_norm = float(
            np.clip(avg_power / max(self.p_max, 1.0), 0.0, 1.0)
        )
        with self._lock:
            cost_grid = self._energy_cost_grid
            hit_count = self._energy_hit_count

        # Aggiorna le celle vicine al robot (nel centro della gridmap)
        # TODO: in futuro, usare la posizione robot reale per aggiornare
        # solo la cella corrente. Per ora aggiorniamo tutte le celle valide
        # con peso proporzionale alla vicinanza al centro.
        center_r = n_rows // 2
        center_c = n_cols // 2
        ii, jj = np.where(valid_mask)
        dist_from_center = np.sqrt(
            ((ii - center_r) / n_rows) ** 2
            + ((jj - center_c) / n_cols) ** 2
        )
        # Solo celle entro 1.5m dal robot (15 celle a 0.1m)
        close_mask = dist_from_center < 0.30  # 30% della griglia
        for idx in np.where(close_mask)[0]:
            r, c = ii[idx], jj[idx]
            if hit_count[r, c] == 0:
                cost_grid[r, c] = current_power_norm
            else:
                cost_grid[r, c] = (
                    (1 - self.alpha) * cost_grid[r, c]
                    + self.alpha * current_power_norm
                )
            hit_count[r, c] += 1

        # ── 5. Costruzione tensor compresso (n_pca + 1, n_rows, n_cols) ───
        # Se la PCA non è ancora pronta, usiamo solo zeri per le feature DINOv2
        n_out_layers = self.n_pca + 1  # 16 feature PCA + 1 costo energetico
        output_tensor = np.zeros(
            (n_out_layers, n_rows, n_cols), dtype=np.float32
        )

        if projected is not None:
            # Layer 0..15: feature DINOv2 compresse (PCA)
            pca_grid = np.zeros((n_rows, n_cols, self.n_pca), dtype=np.float32)
            pca_grid[valid_mask] = projected
            # Trasponi a (n_pca, n_rows, n_cols) — formato channels-first per CNN
            output_tensor[:self.n_pca] = pca_grid.transpose(2, 0, 1)

        # Layer 16: costo energetico normalizzato
        output_tensor[self.n_pca] = cost_grid

        # ── 6. Pubblicazione tensor ─────────────────────────────────────────
        self._publish_tensor(output_tensor, msg.header.stamp, n_rows, n_cols)
        self._publish_debug(cost_grid, msg.header.stamp, n_rows, n_cols)

    # ======================================================================
    # PCA online incrementale
    # ======================================================================
    def _update_pca_and_project(
        self, valid_features: np.ndarray
    ) -> np.ndarray | None:
        """
        Aggiorna la PCA online con i nuovi campioni e proietta le feature.

        Usa sklearn.decomposition.IncrementalPCA.

        Ritorna
        -------
        projected : (N_valid, n_pca) float32, oppure None se PCA non pronta.
        """
        from sklearn.decomposition import IncrementalPCA

        with self._lock:
            # Accumula campioni nel buffer finché non ne abbiamo abbastanza
            if not self._pca_ready:
                self._feature_buffer.append(valid_features)
                total_samples = sum(b.shape[0] for b in self._feature_buffer)

                if total_samples >= self.pca_min_s:
                    # Inizializzazione della PCA con tutti i campioni accumulati
                    all_samples = np.concatenate(self._feature_buffer, axis=0)
                    n_comp = min(self.n_pca, all_samples.shape[0], all_samples.shape[1])
                    self._pca = IncrementalPCA(n_components=n_comp)
                    self._pca.fit(all_samples)
                    self._pca_ready = True
                    self._feature_buffer = []   # libera memoria
                    self.get_logger().info(
                        f"PCA inizializzata con {all_samples.shape[0]} campioni. "
                        f"Varianza spiegata: "
                        f"{self._pca.explained_variance_ratio_.sum()*100:.1f}%"
                    )
                else:
                    # Non ancora pronti: aggiorna incrementalmente se esiste già
                    return None

            # Aggiornamento incrementale
            batch_size = max(self.n_pca + 1, valid_features.shape[0])
            if valid_features.shape[0] >= self.n_pca + 1:
                self._pca.partial_fit(valid_features)

            # Proiezione
            projected = self._pca.transform(valid_features).astype(np.float32)
            return projected

    # ======================================================================
    # Pubblicazione messaggi
    # ======================================================================
    def _publish_tensor(
        self,
        tensor: np.ndarray,
        stamp,
        n_rows: int,
        n_cols: int,
    ) -> None:
        """
        Pubblica il tensor compresso come Float32MultiArray con metadati di forma.
        Forma: (n_layers, n_rows, n_cols).
        """
        n_layers = tensor.shape[0]
        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label="layers", size=n_layers,
                                stride=n_layers * n_rows * n_cols),
            MultiArrayDimension(label="rows",   size=n_rows,
                                stride=n_rows * n_cols),
            MultiArrayDimension(label="cols",   size=n_cols,
                                stride=n_cols),
        ]
        msg.data = tensor.flatten().tolist()
        self._pub_tensor.publish(msg)

    def _publish_debug(
        self,
        cost_grid: np.ndarray,
        stamp,
        n_rows: int,
        n_cols: int,
    ) -> None:
        """
        Pubblica il layer di costo energetico come OccupancyGrid (0–100) per RViz.
        Celle più scure = consumo più alto.
        """
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = "odom"
        msg.info.resolution = float(self._resolution)
        msg.info.width  = n_cols
        msg.info.height = n_rows
        msg.info.origin.position.x = (
            self._center_x - (n_rows / 2.0) * self._resolution
        )
        msg.info.origin.position.y = (
            self._center_y - (n_cols / 2.0) * self._resolution
        )
        msg.info.origin.orientation.w = 1.0

        # Normalizza 0.0–1.0 → 0–100
        occ = (np.clip(cost_grid, 0.0, 1.0) * 100).astype(np.int8)
        msg.data = occ.T.flatten().tolist()
        self._pub_debug.publish(msg)


# ===========================================================================
# Entry point
# ===========================================================================
def main(args=None) -> None:
    rclpy.init(args=args)
    node = EnergyCostmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
