"""
ROS 2 node: **feature_extraction**

Subscribes to RGB + Depth images (synchronised), runs DINOv2 to extract
patch‑level feature embeddings, projects them onto a local 5 m × 5 m grid‑map
centred on the robot, and publishes:

  • ``/feature_image``          – PCA debug image   (sensor_msgs/Image, RGB8)
  • ``/terrain_gridmap``        – feature grid‑map   (grid_map_msgs/GridMap, 384 layers)
  • ``/terrain_gridmap_debug``  – occupancy debug    (nav_msgs/OccupancyGrid)
"""

from __future__ import annotations

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

import message_filters
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
from grid_map_msgs.msg import GridMap, GridMapInfo
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge

import tf2_ros

# Package‑local (pure‑python) libraries.
from spot_terrain_gridmap.dino_feature_extractor import DinoFeatureExtractor
from spot_terrain_gridmap.gridmap_manager import GridmapManager
from spot_terrain_gridmap.projection_utils import (
    quaternion_to_rotation_matrix,
    unproject_patches_to_3d,
    apply_transform,
)


class FeatureExtractionNode(Node):
    """Single ROS 2 node that orchestrates the full pipeline."""

    def __init__(self) -> None:
        super().__init__("feature_extraction")

        # ── Declare parameters ───────────────────────────────────────────
        self._declare_params()

        # ── Read parameters ──────────────────────────────────────────────
        image_topic = self.get_parameter("image_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        camera_info_topic = self.get_parameter("camera_info_topic").value
        feature_image_topic = self.get_parameter("feature_image_topic").value
        gridmap_topic = self.get_parameter("gridmap_topic").value
        occupancy_topic = self.get_parameter("occupancy_topic").value

        self.camera_frame = self.get_parameter("camera_frame").value
        self.robot_base_frame = self.get_parameter("robot_base_frame").value
        self.odom_frame = self.get_parameter("odom_frame").value

        model_name = self.get_parameter("model_name").value
        device = self.get_parameter("device").value
        input_size = self.get_parameter("input_size").value

        gridmap_length_x = self.get_parameter("gridmap_length_x").value
        gridmap_length_y = self.get_parameter("gridmap_length_y").value
        gridmap_resolution = self.get_parameter("gridmap_resolution").value

        self.depth_min = self.get_parameter("depth_min").value
        self.depth_max = self.get_parameter("depth_max").value
        self.log_every_n = self.get_parameter("log_every_n").value

        # ── DINOv2 feature extractor ─────────────────────────────────────
        self.get_logger().info(
            f"Loading DINOv2 model '{model_name}' on {device} "
            f"(input {input_size}×{input_size}) …"
        )
        self.dino = DinoFeatureExtractor(
            model_name=model_name,
            device=device,
            input_size=input_size,
        )
        self.get_logger().info(
            f"DINOv2 ready — patches {self.dino.n_patches}×{self.dino.n_patches}, "
            f"feature dim {self.dino.feature_dim}"
        )

        # ── Gridmap manager ──────────────────────────────────────────────
        self.gridmap = GridmapManager(
            size_x=gridmap_length_x,
            size_y=gridmap_length_y,
            resolution=gridmap_resolution,
            feature_dim=self.dino.feature_dim,
        )
        self.get_logger().info(
            f"Gridmap {self.gridmap.n_rows}×{self.gridmap.n_cols} cells, "
            f"{gridmap_length_x}×{gridmap_length_y} m, "
            f"resolution {gridmap_resolution} m"
        )

        # ── CvBridge ─────────────────────────────────────────────────────
        self.bridge = CvBridge()

        # ── Camera intrinsics (filled by camera_info callback) ───────────
        self.fx: float | None = None
        self.fy: float | None = None
        self.cx: float | None = None
        self.cy: float | None = None
        self.camera_info_received = False

        # ── TF2 ──────────────────────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Subscribers ──────────────────────────────────────────────────
        # Camera info (latched – we only need the first message).
        self.create_subscription(
            CameraInfo, camera_info_topic,
            self._on_camera_info, 10,
        )

        # Synchronised RGB + Depth.
        self._rgb_sub = message_filters.Subscriber(self, Image, image_topic)
        self._depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub],
            queue_size=10,
            slop=0.05,
        )
        self._sync.registerCallback(self._on_image_depth)

        # ── Publishers ───────────────────────────────────────────────────
        self.pub_feature_image = self.create_publisher(
            Image, feature_image_topic, 10,
        )
        self.pub_gridmap = self.create_publisher(
            GridMap, gridmap_topic, 10,
        )
        self.pub_occupancy = self.create_publisher(
            OccupancyGrid, occupancy_topic, 10,
        )

        # ── Bookkeeping ──────────────────────────────────────────────────
        self._frame_count = 0

        # Pre‑build the layer name list once.
        self._layer_names = [f"f_{k}" for k in range(self.dino.feature_dim)]

        self.get_logger().info("feature_extraction node initialised ✓")

    # ==================================================================
    # Parameter declaration
    # ==================================================================
    def _declare_params(self) -> None:
        """Declare all ROS 2 parameters with sensible defaults."""
        self.declare_parameter("image_topic", "/camera/rgb")
        self.declare_parameter("depth_topic", "/camera/depth")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("feature_image_topic", "/feature_image")
        self.declare_parameter("gridmap_topic", "/terrain_gridmap")
        self.declare_parameter("occupancy_topic", "/terrain_gridmap_debug")

        self.declare_parameter("camera_frame", "ZED_X")
        self.declare_parameter("robot_base_frame", "base_link")
        self.declare_parameter("odom_frame", "odom")

        self.declare_parameter("model_name", "facebook/dinov2-small")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("input_size", 518)

        self.declare_parameter("gridmap_length_x", 2.5)
        self.declare_parameter("gridmap_length_y", 2.5)
        self.declare_parameter("gridmap_resolution", 0.1)

        self.declare_parameter("depth_min", 0.1)
        self.declare_parameter("depth_max", 10.0)

        self.declare_parameter("log_every_n", 30)

    # ==================================================================
    # Camera‑info callback
    # ==================================================================
    def _on_camera_info(self, msg: CameraInfo) -> None:
        """Store camera intrinsics from the first received message."""
        if self.camera_info_received:
            return
        K = msg.k  # 3×3 row‑major: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self.fx = K[0]
        self.fy = K[4]
        self.cx = K[2]
        self.cy = K[5]
        self.camera_info_received = True
        self.get_logger().info(
            f"Camera intrinsics received — "
            f"fx={self.fx:.1f}  fy={self.fy:.1f}  "
            f"cx={self.cx:.1f}  cy={self.cy:.1f}"
        )

    # ==================================================================
    # Main synchronised callback
    # ==================================================================
    def _on_image_depth(self, rgb_msg: Image, depth_msg: Image) -> None:
        """Process one synchronised RGB + Depth pair."""
        # ── 0. Guards ────────────────────────────────────────────────────
        if not self.camera_info_received:
            self.get_logger().warn(
                "Skipping frame — camera intrinsics not yet received.",
                throttle_duration_sec=5.0,
            )
            return

        stamp = rgb_msg.header.stamp
        ros_time = Time.from_msg(stamp)

        # ── 0.5 Check for time jump backwards ─────────────────────────────
        current_time_sec = ros_time.nanoseconds / 1e9
        if hasattr(self, '_last_time_sec'):
            if current_time_sec < self._last_time_sec - 1.0:
                self.get_logger().warn("Time jumped backwards! Clearing TF buffer.")
                self.tf_buffer.clear()
        self._last_time_sec = current_time_sec

        # ── 1. Convert images ────────────────────────────────────────────
        rgb_np = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        depth_np = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
        orig_h, orig_w = rgb_np.shape[:2]

        # ── 2. DINOv2 inference ──────────────────────────────────────────
        features = self.dino.extract(rgb_np)  # (37, 37, 384)

        # ── 3. PCA debug image ───────────────────────────────────────────
        pca_img = self.dino.compute_pca_image(features, orig_h, orig_w)
        
        # Bypassing cv_bridge.cv2_to_imgmsg due to OpenCV 5 compatibility issues in Humble.
        pca_msg = Image()
        pca_msg.header = rgb_msg.header
        pca_msg.height = pca_img.shape[0]
        pca_msg.width = pca_img.shape[1]
        pca_msg.encoding = "rgb8"
        pca_msg.is_bigendian = 0
        pca_msg.step = pca_img.shape[1] * pca_img.shape[2]
        pca_msg.data = pca_img.tobytes()
        
        self.pub_feature_image.publish(pca_msg)

        # ── 4. TF lookups ────────────────────────────────────────────────
        try:
            tf_cam: TransformStamped = self.tf_buffer.lookup_transform(
                self.odom_frame,
                rgb_msg.header.frame_id,  # Use optical frame from image message
                Time(),  # Use the latest available transform to avoid extrapolation errors
                timeout=Duration(seconds=0.0),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            self.get_logger().warn(
                f"TF {self.odom_frame}→{self.camera_frame} failed: {e}",
                throttle_duration_sec=5.0,
            )
            return

        try:
            tf_base: TransformStamped = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.robot_base_frame,
                Time(),  # Use the latest available transform to avoid extrapolation errors
                timeout=Duration(seconds=0.0),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            self.get_logger().warn(
                f"TF {self.odom_frame}→{self.robot_base_frame} failed: {e}",
                throttle_duration_sec=5.0,
            )
            return

        # ── 5. Unproject patch centres to 3‑D (camera frame) ─────────────
        points_cam, valid = unproject_patches_to_3d(
            depth_map=depth_np,
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            patch_size=self.dino.PATCH_SIZE,
            input_size=self.dino.input_size,
            orig_h=orig_h, orig_w=orig_w,
            n_patches_h=self.dino.n_patches,
            n_patches_w=self.dino.n_patches,
            depth_min=self.depth_min,
            depth_max=self.depth_max,
        )

        # ── 5.1. Ignore top half of the image ───────────────────────────
        # Mask out the top half of the features to only project nearby terrain
        half_idx = (self.dino.n_patches // 2) * self.dino.n_patches
        valid[:half_idx] = False

        # ── 5.5. Optical to Base frame conversion (if necessary) ────────
        # If the image's frame_id does not indicate an optical frame, 
        # it is likely a standard ROS base frame (X forward, Y left, Z up).
        # We must manually rotate the points from optical (Z forward, X right, Y down).
        if "optical" not in rgb_msg.header.frame_id.lower():
            points_cam = np.stack([
                points_cam[:, 2],     # X_base = Z_optical
                -points_cam[:, 0],    # Y_base = -X_optical
                -points_cam[:, 1]     # Z_base = -Y_optical
            ], axis=-1)

        # ── 6. Transform points: camera → odom ──────────────────────────
        t = tf_cam.transform.translation
        q = tf_cam.transform.rotation
        R = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
        translation = np.array([t.x, t.y, t.z])

        points_odom = apply_transform(points_cam, translation, R)

        # ── 7. Update gridmap centre (robot position) ────────────────────
        robot_x = tf_base.transform.translation.x
        robot_y = tf_base.transform.translation.y
        self.gridmap.update_center(robot_x, robot_y)

        # ── 8. Insert features ───────────────────────────────────────────
        features_flat = features.reshape(-1, self.dino.feature_dim)
        n_inserted = self.gridmap.insert_features(points_odom, features_flat, valid)

        # ── 9. Publish GridMap (384 layers) ──────────────────────────────
        gridmap_msg = self._build_gridmap_msg(stamp)
        self.pub_gridmap.publish(gridmap_msg)

        # ── 10. Publish OccupancyGrid (debug) ────────────────────────────
        occ_msg = self._build_occupancy_msg(stamp)
        self.pub_occupancy.publish(occ_msg)

        # ── 11. Logging ─────────────────────────────────────────────────
        self._frame_count += 1
        if self._frame_count % self.log_every_n == 0:
            n_valid = int(valid.sum())
            n_total = int(valid.size)
            n_occ = int(self.gridmap.valid_mask.sum())
            self.get_logger().info(
                f"Frame {self._frame_count}: "
                f"valid depth {n_valid}/{n_total} patches, "
                f"inserted {n_inserted}, "
                f"gridmap occupancy {n_occ}/{self.gridmap.n_rows * self.gridmap.n_cols}"
            )

    # ==================================================================
    # Message builders
    # ==================================================================
    def _build_gridmap_msg(self, stamp) -> GridMap:
        """Build a ``grid_map_msgs/GridMap`` with 384 feature layers."""
        msg = GridMap()
        msg.header.stamp = stamp
        msg.header.frame_id = self.odom_frame

        # GridMapInfo
        msg.info.resolution = self.gridmap.resolution
        msg.info.length_x = self.gridmap.size_x
        msg.info.length_y = self.gridmap.size_y
        msg.info.pose.position.x = self.gridmap.center_x
        msg.info.pose.position.y = self.gridmap.center_y
        msg.info.pose.position.z = 0.0
        msg.info.pose.orientation.w = 1.0

        msg.layers = list(self._layer_names)
        msg.basic_layers = []

        n_r = self.gridmap.n_rows
        n_c = self.gridmap.n_cols

        for k in range(self.dino.feature_dim):
            arr = Float32MultiArray()
            arr.layout.dim = [
                MultiArrayDimension(
                    label="column_index", size=n_c, stride=n_r * n_c,
                ),
                MultiArrayDimension(
                    label="row_index", size=n_r, stride=n_r,
                ),
            ]
            layer = self.gridmap.get_feature_layer(k)  # (n_r, n_c)
            # Column‑major (Eigen/Fortran order) as expected by grid_map.
            arr.data = layer.flatten(order="F").tolist()
            msg.data.append(arr)

        return msg

    def _build_occupancy_msg(self, stamp) -> OccupancyGrid:
        """Build a ``nav_msgs/OccupancyGrid`` for RViz visualisation."""
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.odom_frame

        msg.info.resolution = float(self.gridmap.resolution)
        msg.info.width = self.gridmap.n_rows
        msg.info.height = self.gridmap.n_cols

        # Origin = bottom‑left corner of the grid in odom.
        msg.info.origin.position.x = (
            self.gridmap.center_x
            - (self.gridmap.n_rows / 2.0) * self.gridmap.resolution
        )
        msg.info.origin.position.y = (
            self.gridmap.center_y
            - (self.gridmap.n_cols / 2.0) * self.gridmap.resolution
        )
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        occ = self.gridmap.get_occupancy_data()  # (n_rows, n_cols) int8
        # OccupancyGrid expects row-major data[y * width + x]. 
        # We transpose the data so that X (rows) varies fastest.
        msg.data = occ.T.flatten().tolist()

        return msg


# ======================================================================
# Entry point
# ======================================================================
def main(args=None):
    rclpy.init(args=args)
    node = FeatureExtractionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
