"""One physical Spot, local mixed board, body-mounted RGB-D/semantic camera.

Import only after AppLauncher. No navigation policy or DINO is loaded here.
"""
import math
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.math import quat_apply, quat_mul
from isaacsim.core.utils.semantics import add_update_semantics
from pxr import Usd, UsdGeom
from common import camera_optics, local_path


def build_scene(config, manifest, device):
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=config["physics"]["dt"], device=device))
    board = sim_utils.UsdFileCfg(usd_path=str(local_path(manifest["board"])))
    board.func("/World/Board", board)
    # Ground truth comes from the renderer, not from DINO or assumed camera rays.
    for station in manifest["stations"]:
        path = "/World/Board" + station["prim_path"].removeprefix("/World")
        for prim in Usd.PrimRange(sim.stage.GetPrimAtPath(path)):
            if prim.IsA(UsdGeom.Mesh):
                add_update_semantics(prim, station["label"], type_label="class")
    light = sim_utils.DistantLightCfg(intensity=float(config["lighting"]["intensity"]), angle=1.0)
    light.func("/World/Light", light)
    robot = Articulation(ArticulationCfg(
        prim_path="/World/Spot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(local_path(manifest["robot"])),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, retain_accelerations=False, linear_damping=0., angular_damping=0.,
                max_linear_velocity=1000., max_angular_velocity=1000., max_depenetration_velocity=1.),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0)),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0., -2.8, config["physics"]["spawn_height_m"]),
            joint_pos={"[fh]l_hx": .1, "[fh]r_hx": -.1, "f[rl]_hy": .9, "h[rl]_hy": 1.1, ".*_kn": -1.5},
            joint_vel={".*": 0.}),
        actuators={
            "hips": ImplicitActuatorCfg(joint_names_expr=[".*_h[xy]"], effort_limit_sim=45., stiffness=60., damping=1.5),
            "knees": ImplicitActuatorCfg(joint_names_expr=[".*_kn"], effort_limit_sim=115., stiffness=60., damping=1.5)}))
    camera = Camera(CameraCfg(
        prim_path="/World/SpotCamera", width=config["camera"]["width"], height=config["camera"]["height"],
        update_period=0.0, update_latest_camera_pose=True,
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
        colorize_semantic_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(**camera_optics(config))))
    sim.reset()
    sim.set_camera_view(eye=(9., -12., 9.), target=(9., 1., 0.))
    return sim, robot, camera


def sync_camera(robot, camera, config):
    device = robot.data.root_pos_w.device
    offset = torch.tensor([config["camera"]["body_offset_m"]], device=device)
    pitch = math.radians(config["camera"]["pitch_down_deg"])/2
    rotation = torch.tensor([[math.cos(pitch), 0., math.sin(pitch), 0.]], device=device)
    camera.set_world_poses(
        positions=robot.data.root_pos_w + quat_apply(robot.data.root_quat_w, offset),
        orientations=quat_mul(robot.data.root_quat_w, rotation), convention="world")
