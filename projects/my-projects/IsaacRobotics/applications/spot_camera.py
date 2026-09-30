"""Mount the ZED_X asset and publish its simulated images to ROS 2.

The mount pose and ROS topic names match champ_container/sim/spot_space_data.usd.
Import this module only after creating the Isaac Sim SimulationApp.
"""

import math

import omni.graph.core as og
import usdrt.Sdf
from pxr import Gf, Sdf, UsdGeom, UsdPhysics


CAMERA_MOUNT_TRANSLATION = (0.42, 0.0, 0.07)
CAMERA_MOUNT_PITCH_DEG = 15.0
CAMERA_FRAME_ID = "ZED_X"
CAMERA_ASSET = "/Isaac/Sensors/Stereolabs/ZED_X/ZED_X.usdc"


def attach_zed_camera(stage, assets_root_path, robot_path="/World/Spot"):
    """Reference ZED_X beside Spot's body and fix its rigid body to Spot."""
    body_path = f"{robot_path}/body"
    mount_path = f"{robot_path}/ZED_X"
    if not stage.GetPrimAtPath(body_path).IsValid():
        raise RuntimeError(f"Spot body not found: {body_path}")
    if not assets_root_path:
        raise RuntimeError("Isaac Sim assets root is unavailable")

    mount = UsdGeom.Xform.Define(stage, mount_path)
    mount.GetPrim().GetReferences().AddReference(assets_root_path.rstrip("/") + CAMERA_ASSET)
    xform = UsdGeom.XformCommonAPI(mount)
    xform.SetTranslate(Gf.Vec3d(*CAMERA_MOUNT_TRANSLATION))
    mount.GetPrim().GetAttribute("xformOp:rotateZYX").Set(
        Gf.Vec3d(0.0, CAMERA_MOUNT_PITCH_DEG, 0.0)
    )

    joint = UsdPhysics.FixedJoint.Define(stage, f"{mount_path}/FixedJoint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(mount_path)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*CAMERA_MOUNT_TRANSLATION))
    half_pitch = math.radians(CAMERA_MOUNT_PITCH_DEG) / 2.0
    joint.CreateLocalRot0Attr(Gf.Quatf(math.cos(half_pitch), Gf.Vec3f(0.0, math.sin(half_pitch), 0.0)))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

    camera_path = f"{mount_path}/base_link/ZED_X/CameraRight"
    if not UsdGeom.Camera.Get(stage, camera_path):
        raise RuntimeError(f"ZED_X camera not found in asset: {camera_path}")
    return camera_path


def create_zed_ros_graph(camera_path):
    """Publish RGB, depth and intrinsics on the topics used by DINOv2."""
    keys = og.Controller.Keys
    graph_path = "/World/ROS_Camera"
    graph, _, _, _ = og.Controller.edit(
        {
            "graph_path": graph_path,
            "evaluator_name": "push",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
        },
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("RenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("RGBPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("DepthPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("CameraInfoPublish", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "RenderProduct.inputs:execIn"),
                ("RenderProduct.outputs:execOut", "RGBPublish.inputs:execIn"),
                ("RenderProduct.outputs:execOut", "DepthPublish.inputs:execIn"),
                ("RenderProduct.outputs:execOut", "CameraInfoPublish.inputs:execIn"),
                ("RenderProduct.outputs:renderProductPath", "RGBPublish.inputs:renderProductPath"),
                ("RenderProduct.outputs:renderProductPath", "DepthPublish.inputs:renderProductPath"),
                ("RenderProduct.outputs:renderProductPath", "CameraInfoPublish.inputs:renderProductPath"),
                ("Context.outputs:context", "RGBPublish.inputs:context"),
                ("Context.outputs:context", "DepthPublish.inputs:context"),
                ("Context.outputs:context", "CameraInfoPublish.inputs:context"),
            ],
            keys.SET_VALUES: [
                ("RenderProduct.inputs:cameraPrim", [usdrt.Sdf.Path(camera_path)]),
                ("RenderProduct.inputs:width", 640),
                ("RenderProduct.inputs:height", 360),
                ("RGBPublish.inputs:frameId", CAMERA_FRAME_ID),
                ("RGBPublish.inputs:topicName", "/camera/rgb"),
                ("RGBPublish.inputs:type", "rgb"),
                ("DepthPublish.inputs:frameId", CAMERA_FRAME_ID),
                ("DepthPublish.inputs:topicName", "/camera/depth"),
                ("DepthPublish.inputs:type", "depth"),
                ("CameraInfoPublish.inputs:frameId", CAMERA_FRAME_ID),
                ("CameraInfoPublish.inputs:topicName", "/camera/camera_info"),
            ],
        },
    )
    og.Controller.evaluate_sync(graph)
    return graph_path


def publish_zed_static_tf(ros_node):
    """Provide base_link -> ZED_X for the existing terrain gridmap node."""
    from geometry_msgs.msg import TransformStamped
    from tf2_ros import StaticTransformBroadcaster

    broadcaster = StaticTransformBroadcaster(ros_node)
    transform = TransformStamped()
    transform.header.frame_id = "base_link"
    transform.child_frame_id = CAMERA_FRAME_ID
    transform.transform.translation.x = CAMERA_MOUNT_TRANSLATION[0]
    transform.transform.translation.y = CAMERA_MOUNT_TRANSLATION[1]
    transform.transform.translation.z = CAMERA_MOUNT_TRANSLATION[2]
    half_pitch = math.radians(CAMERA_MOUNT_PITCH_DEG) / 2.0
    transform.transform.rotation.x = 0.0
    transform.transform.rotation.y = math.sin(half_pitch)
    transform.transform.rotation.z = 0.0
    transform.transform.rotation.w = math.cos(half_pitch)
    broadcaster.sendTransform(transform)
    return broadcaster
