#!/usr/bin/env python3
"""
fake_camera_publisher.py

Nodo ROS 2 "finto pubblicatore": pubblica immagini sintetiche (a blocchi
colorati) su /camera/color/image_raw e i corrispondenti parametri
intrinseci placeholder su /camera/color/camera_info, a una frequenza
configurabile.

SCOPO: permettere di testare l'intera pipeline terrain_feature_node
(DINOv2 + proiezione BEV + pubblicazione GridMap) END-TO-END, PRIMA che la
camera reale o quella simulata in Isaac Sim siano disponibili (vedi
documento di contesto, Sez. 1.2: "TODO / decisione aperta" sulla camera).

NON fa parte della pipeline finale: è solo uno strumento di test/sviluppo,
tenuto volutamente separato dal nodo vero (spot_terrain_gridmap/
terrain_feature_node.py) per non mescolare codice di produzione e codice
di test.

--------------------------------------------------------------------------
USO
--------------------------------------------------------------------------
Terminale 1 (lancia il nodo vero):
    source /opt/ros/humble/setup.bash
    source ~/ros2_ws/install/setup.bash
    source ~/venvs/dino_env/bin/activate
    ros2 launch spot_terrain_gridmap terrain_gridmap.launch.py

Terminale 2 (lancia questo finto pubblicatore):
    source /opt/ros/humble/setup.bash
    source ~/ros2_ws/install/setup.bash
    python3 ~/ros2_ws/src/spot_terrain_gridmap/test/fake_camera_publisher.py

Nota: questo script NON richiede il venv con PyTorch (non usa DINOv2, solo
numpy e rclpy), quindi puoi lanciarlo anche solo con l'ambiente ROS 2 di
sistema, senza attivare dino_env.

Terminale 3 (opzionale, per osservare cosa succede):
    source /opt/ros/humble/setup.bash
    source ~/ros2_ws/install/setup.bash
    ros2 topic hz /terrain_gridmap
    # oppure, per vedere la struttura del messaggio:
    ros2 topic echo /terrain_gridmap --no-arr
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


class FakeCameraPublisher(Node):
    def __init__(self):
        super().__init__('fake_camera_publisher')

        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('frame_id', 'camera_color_optical_frame')
        self.declare_parameter('publish_rate_hz', 30.0)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('block_size_px', 80)

        self.image_topic = self.get_parameter('image_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        rate_hz = self.get_parameter('publish_rate_hz').value
        self.width = self.get_parameter('image_width').value
        self.height = self.get_parameter('image_height').value
        self.block_size = self.get_parameter('block_size_px').value

        self.bridge = CvBridge()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub_image = self.create_publisher(Image, self.image_topic, qos)
        self.pub_camera_info = self.create_publisher(CameraInfo, self.camera_info_topic, qos)

        self._synthetic_image = self._make_checkerboard_image()
        self._camera_info_msg = self._make_camera_info_msg()

        period = 1.0 / rate_hz
        self.timer = self.create_timer(period, self.publish_frame)

        self.get_logger().info(
            f'Fake camera publisher avviato: pubblico su {self.image_topic} '
            f'e {self.camera_info_topic} a {rate_hz} Hz '
            f'(immagine {self.width}x{self.height}, blocchi {self.block_size}px).'
        )

    def _make_checkerboard_image(self) -> np.ndarray:
        """
        Genera un'immagine sintetica a blocchi colorati (scacchiera con
        colori/texture diversi per blocco), per dare a DINOv2 qualcosa di
        strutturato su cui produrre embedding non banali - a differenza di
        un'immagine a colore uniforme o rumore puro casuale.
        """
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Palette di colori distinti, uno per tipo di "finto terreno"
        palette = [
            (139, 90, 43),    # marrone (finto terreno fangoso)
            (194, 178, 128),  # sabbia
            (105, 105, 105),  # grigio asfalto
            (34, 139, 34),    # verde erba
        ]

        rng = np.random.default_rng(0)

        for by in range(0, self.height, self.block_size):
            for bx in range(0, self.width, self.block_size):
                color = palette[rng.integers(0, len(palette))]
                # Aggiunge un po' di rumore per blocco, cosi' i blocchi
                # dello stesso "tipo" non sono tutti pixel-identici
                # (piu' realistico, evita un caso troppo degenere)
                block_h = min(self.block_size, self.height - by)
                block_w = min(self.block_size, self.width - bx)
                noise = rng.integers(-15, 15, size=(block_h, block_w, 3))
                block = np.clip(np.array(color) + noise, 0, 255).astype(np.uint8)
                img[by:by + block_h, bx:bx + block_w] = block

        return img

    def _make_camera_info_msg(self) -> CameraInfo:
        """
        Parametri intrinseci placeholder, coerenti con quelli in
        config/terrain_gridmap_params.yaml (camera_intrinsics_placeholder).
        """
        msg = CameraInfo()
        msg.width = self.width
        msg.height = self.height
        fx = fy = 525.0 * (self.width / 640.0)
        cx = self.width / 2.0
        cy = self.height / 2.0
        msg.k = [fx, 0.0, cx,
                 0.0, fy, cy,
                 0.0, 0.0, 1.0]
        msg.header.frame_id = self.frame_id
        return msg

    def publish_frame(self):
        now = self.get_clock().now().to_msg()

        img_msg = self.bridge.cv2_to_imgmsg(self._synthetic_image, encoding='rgb8')
        img_msg.header.stamp = now
        img_msg.header.frame_id = self.frame_id
        self.pub_image.publish(img_msg)

        self._camera_info_msg.header.stamp = now
        self.pub_camera_info.publish(self._camera_info_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FakeCameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
