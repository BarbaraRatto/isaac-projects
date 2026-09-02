import os

# Fail-safe: if not explicitly chosen otherwise,
# ROS 2 communicates only on the local machine.
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")

print(
    "ROS 2 network mode:",
    "LOCALHOST" if os.environ["ROS_LOCALHOST_ONLY"] == "1" else "NETWORK",
)

import sys
import shlex
import subprocess
from pathlib import Path

dir_path = Path(__file__).resolve().parent
sys.path.append(str(dir_path / ".."))

ros_ws = dir_path / "msgs_ws"
setup_bash = ros_ws / "install" / "setup.bash"

if not setup_bash.exists():
    print("Building the msgs first...")
    subprocess.run(["colcon", "build"], cwd=ros_ws, check=True)

if os.environ.get("QUADRUPED_PYMPC_ROS2_SOURCED") != "1":
    print("Sourcing ROS2 workspace and restarting script...")
    cmd = (
        f"source {shlex.quote(str(setup_bash))} && "
        "export QUADRUPED_PYMPC_ROS2_SOURCED=1 && "
        f"exec {shlex.quote(sys.executable)} "
        + " ".join(shlex.quote(arg) for arg in [str(Path(__file__).resolve()), *sys.argv[1:]])
    )
    os.execv("/bin/bash", ["bash", "-c", cmd])


import rclpy 
from rclpy.node import Node 
from dls2_interface.msg import BaseState, BlindState, ControlSignal, TrajectoryGenerator, TimeDebug
from sensor_msgs.msg import Joy

import time
import numpy as np
np.set_printoptions(precision=3, suppress=True)

import threading
import multiprocessing
from multiprocessing import shared_memory, Value

import copy

# Gym and Simulation related imports
import mujoco
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.quadruped_utils import LegsAttr


# Config imports
from quadruped_pympc import config as cfg


# Set the priority of the process
pid = os.getpid()
print("PID: ", pid)
os.system("sudo renice -n -21 -p " + str(pid))
os.system("sudo echo -20 > /proc/" + str(pid) + "/autogroup")

# to reserve the core 4, 5 for the process, add in etc/default/grub
# GRUB_CMDLINE_LINUX_DEFAULT="quiet splash isolcpus=4-5" in etc/default/grub
# and then sudo update-grub
# and uncomment the lines below
#affinity_mask = {4, 5} 
#os.sched_setaffinity(pid, affinity_mask)

#for real time, launch it with chrt -r 99 python3 run_controller.py

USE_THREADED_MPC = False
USE_PROCESS_QUEUE_MPC = False
USE_PROCESS_SHARED_MEMORY_MPC = False
if(USE_PROCESS_SHARED_MEMORY_MPC):
        # -------------------- Shared-memory layout for MPC → WBC --------------------------------------
    # Payload layout (float64):
    # 0..11   : GRF   (4 legs × 3)
    # 12..23  : Footholds (4×3)
    # 24..35  : Joints pos (4×3)
    # 36..47  : Joints vel (4×3)
    # 48..59  : Joints acc (4×3)
    # 60..71  : Predicted state (12)
    # 72      : best_sample_freq (1)
    # 73      : last_mpc_loop_time (1)
    # 74      : stamp_mono (1)
    N_DBL = 75
    IDX_GRF   = slice(0, 12)
    IDX_FH    = slice(12, 24)
    IDX_JP    = slice(24, 36)
    IDX_JV    = slice(36, 48)
    IDX_JA    = slice(48, 60)
    IDX_PRED  = slice(60, 72)
    IDX_BSF   = 72
    IDX_LAST  = 73
    IDX_STAMP = 74

    def legsattr_to12(legs: LegsAttr) -> np.ndarray:
        return np.concatenate([np.asarray(legs.FL).reshape(-1),
                            np.asarray(legs.FR).reshape(-1),
                            np.asarray(legs.RL).reshape(-1),
                            np.asarray(legs.RR).reshape(-1)], axis=0)


    def vec12_to_legsattr(vec12: np.ndarray) -> LegsAttr:
        v = np.asarray(vec12).reshape(4, 3)
        return LegsAttr(FL=v[0].copy(), FR=v[1].copy(), RL=v[2].copy(), RR=v[3].copy())
    

MPC_FREQ = 100 
RENDER_MUJOCO_VIEWER = False
RENDER_FREQ = 30

USE_SCHEDULER = False # This enable a call to the run function every tot seconds, instead of as fast as possible
SCHEDULER_FREQ = 250 # this is only valid if USE_SCHEDULER is True

USE_FIXED_LOOP_TIME = False # This is used to fix the clock time of periodic gait gen to 1/SCHEDULER_FREQ
USE_SATURATED_LOOP_TIME = True # This is used to cap the clock time of periodic gait gen to max 250Hz

USE_SMOOTH_VELOCITY = False
USE_SMOOTH_HEIGHT = True


def expand_per_leg_values(value, name):
    """Expand a scalar or one-leg [hx, hy, knee] values to all 12 joints."""
    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size == 1:
        return np.full(12, values.item())
    if values.size == 3:
        return np.tile(values, 4)
    if values.size == 12:
        return values.copy()
    raise ValueError(f"{name} must contain 1, 3, or 12 values")

# Shell for the controllers ----------------------------------------------
class Quadruped_PyMPC_Node(Node):
    def __init__(self):
        super().__init__('Quadruped_PyMPC_Node')

        # Subscribers and Publishers
        self.subscription_base_state = self.create_subscription(BaseState,"/base_state", self.get_base_state_callback, 1)
        self.subscription_blind_state = self.create_subscription(BlindState,"/blind_state", self.get_blind_state_callback, 1)
        self.subscription_joy = self.create_subscription(Joy,"joy", self.get_joy_callback, 1)
        self.publisher_control_signal = self.create_publisher(ControlSignal,"/control_signal", 1)
        self.publisher_trajectory_generator = self.create_publisher(TrajectoryGenerator,"/trajectory_generator", 1)
        self.publisher_time_debug = self.create_publisher(TimeDebug,"/time_debug", 1)
        if(USE_SCHEDULER):
            self.timer = self.create_timer(1.0/SCHEDULER_FREQ, self.compute_control_callback)
        

        # Safety check to not do anything until a first base and blind state are received
        self.first_message_base_arrived = False
        self.first_message_joints_arrived = False 

        # Timing stuff
        self.loop_time = cfg.simulation_params.get(
            'ros_interface_dt', cfg.simulation_params['dt']
        )
        self.last_start_time = None
        self.state_timestamp = None
        self.last_state_timestamp = None
        self.last_mpc_loop_time = 0.0
        self.last_standup_log_time = 0.0
        self.last_standing_log_time = 0.0

        # Base State
        self.position = np.zeros(3)
        self.orientation = np.zeros(4)
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        # Blind State
        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)
        self.feet_contact = np.zeros(4)
        # Desired PD gain
        self.impedence_joint_position_gain = np.ones(12)*cfg.simulation_params['impedence_joint_position_gain']
        self.impedence_joint_velocity_gain = np.ones(12)*cfg.simulation_params['impedence_joint_velocity_gain']
        self.standup_joint_position_gain = expand_per_leg_values(
            cfg.simulation_params.get('standup_joint_position_gain', [12.0, 30.0, 30.0]),
            'standup_joint_position_gain',
        )
        self.standup_joint_velocity_gain = expand_per_leg_values(
            cfg.simulation_params.get('standup_joint_velocity_gain', [5.0, 8.0, 8.0]),
            'standup_joint_velocity_gain',
        )


        # Mujoco env
        self.env = QuadrupedEnv(
            robot=cfg.robot,
            scene=cfg.simulation_params['scene'],
            sim_dt=cfg.simulation_params['dt'],
            base_vel_command_type="human"
        )
        self.env.mjModel.opt.gravity[2] = -cfg.gravity_constant
        
        self.feet_traj_geom_ids, self.feet_GRF_geom_ids = None, LegsAttr(FL=-1, FR=-1, RL=-1, RR=-1)
        self.legs_order = ["FL", "FR", "RL", "RR"]
        self.env.reset(random=False)
        self.last_mpc_time = time.time()
        self.last_render_time = time.time()

        if RENDER_MUJOCO_VIEWER:
            self.env.render()
            self.env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            self.env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False


        self.stand_up_and_down_actions = LegsAttr(*[np.zeros((1, int(self.env.mjModel.nu/4))) for _ in range(4)])
        keyframe_id = mujoco.mj_name2id(self.env.mjModel, mujoco.mjtObj.mjOBJ_KEY, "down")
        self.down_pose_initialized = keyframe_id >= 0
        if self.down_pose_initialized:
            goDown_qpos = self.env.mjModel.key_qpos[keyframe_id]
            self.stand_up_and_down_actions.FL = goDown_qpos[7:10]
            self.stand_up_and_down_actions.FR = goDown_qpos[10:13]
            self.stand_up_and_down_actions.RL = goDown_qpos[13:16]
            self.stand_up_and_down_actions.RR = goDown_qpos[16:19]
        else:
            self.get_logger().warning(
                f"Robot '{cfg.robot}' has no MuJoCo 'down' keyframe; "
                "the first measured joint pose will be used as the down pose."
            )


        # Quadruped PyMPC controller initialization -------------------------------------------------------------
        from quadruped_pympc.interfaces.srbd_controller_interface import SRBDControllerInterface
        from quadruped_pympc.interfaces.srbd_batched_controller_interface import SRBDBatchedControllerInterface
        from quadruped_pympc.interfaces.wb_interface import WBInterface

        self.wb_interface = WBInterface(initial_feet_pos = self.env.feet_pos(frame='world'), legs_order = self.legs_order)
        self.srbd_controller_interface = SRBDControllerInterface()

        # This variable are shared between the MPC and the whole body controller
        # in the case of the use of thread. In any case, i initialize them here
        self.nmpc_GRFs = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))
        self.nmpc_footholds = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))
        self.nmpc_joints_pos = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))
        self.nmpc_joints_vel = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))
        self.nmpc_joints_acc = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))
        self.nmpc_predicted_state = np.zeros(12)
        
        self.best_sample_freq = self.wb_interface.pgg.step_freq
        self.state_current = None
        self.ref_state = None
        self.contact_sequence = None
        self.inertia = None
        self.optimize_swing = None

        # Torque vector
        self.tau = LegsAttr(*[np.zeros((self.env.mjModel.nv, 1)) for _ in range(4)])
        # Torque limits
        tau_soft_limits_scalar = 0.9
        configured_tau_limits = cfg.simulation_params.get('joint_torque_limits')
        if configured_tau_limits is not None:
            configured_tau_limits = np.asarray(configured_tau_limits, dtype=float)
            configured_tau_limits = np.column_stack((-configured_tau_limits, configured_tau_limits))
        self.tau_limits = LegsAttr(
            FL=(configured_tau_limits if configured_tau_limits is not None else self.env.mjModel.actuator_ctrlrange[self.env.legs_tau_idx.FL])*tau_soft_limits_scalar,
            FR=(configured_tau_limits if configured_tau_limits is not None else self.env.mjModel.actuator_ctrlrange[self.env.legs_tau_idx.FR])*tau_soft_limits_scalar,
            RL=(configured_tau_limits if configured_tau_limits is not None else self.env.mjModel.actuator_ctrlrange[self.env.legs_tau_idx.RL])*tau_soft_limits_scalar,
            RR=(configured_tau_limits if configured_tau_limits is not None else self.env.mjModel.actuator_ctrlrange[self.env.legs_tau_idx.RR])*tau_soft_limits_scalar)

        # Let's start in FULL STANCE in any case
        self.wb_interface.pgg.gait_type = 7 

        # Threaded MPC
        if(USE_THREADED_MPC):
            thread_mpc = threading.Thread(target=self.compute_mpc_thread_callback)
            thread_mpc.daemon = True
            thread_mpc.start()
        if(USE_PROCESS_QUEUE_MPC):
            self.input_data_process = multiprocessing.Queue(maxsize=1)
            self.output_data_process = multiprocessing.Queue(maxsize=1)
            process_mpc = multiprocessing.Process(target=self.compute_mpc_process_queue_callback, args=(self.input_data_process, self.output_data_process))
            process_mpc.daemon = True
            process_mpc.start()
        if(USE_PROCESS_SHARED_MEMORY_MPC):
            # WBC → MPC: keep a latest-only queue for complex inputs
            self.input_data_process = multiprocessing.Queue(maxsize=1)

            # MPC → WBC: shared-memory SPSC with seqlock
            self.shm_out = shared_memory.SharedMemory(create=True, size=N_DBL * 8)
            self.shm_out_name = self.shm_out.name
            np.ndarray((N_DBL,), dtype=np.float64, buffer=self.shm_out.buf)[:] = 0.0
            self.seq_out = Value('Q', 0, lock=False)  # 64-bit sequence, even=stable

            process_mpc = multiprocessing.Process(
                target=self.compute_mpc_process_shared_memory_callback,
                args=(self.input_data_process, self.shm_out_name, self.seq_out),
            )
            process_mpc.daemon = True
            process_mpc.start()
            

        # Interactive Command Line ----------------------------
        from console import Console
        self.console = Console(controller_node=self)
        thread_console = threading.Thread(target=self.console.interactive_command_line)
        thread_console.daemon = True
        thread_console.start()


        # Init for real robot and simulation gain, since real robot needs different values
        #    self.wb_interface.stc.position_gain_fb = 100
        #    self.wb_interface.stc.velocity_gain_fb = 10
        #    self.wb_interface.stc.use_feedback_linearization = False
        #    self.wb_interface.stc.use_friction_compensation = False


    def compute_mpc_thread_callback(self):
        # This thread runs forever!
        last_mpc_thread_time = time.time()
        while True:
            if time.time() - last_mpc_thread_time > 1.0 / MPC_FREQ:
                if(self.state_current is not None):
                    self.nmpc_GRFs,  \
                    self.nmpc_footholds, \
                    self.nmpc_joints_pos, \
                    self.nmpc_joints_vel, \
                    self.nmpc_joints_acc, \
                    self.best_sample_freq,\
                    self.nmpc_predicted_state = self.srbd_controller_interface.compute_control(self.state_current,
                                                                            self.ref_state,
                                                                            self.contact_sequence,
                                                                            self.inertia,
                                                                            self.wb_interface.pgg.phase_signal,
                                                                            self.wb_interface.pgg.step_freq,
                                                                            self.optimize_swing)
                    
                    if(cfg.mpc_params['type'] != 'sampling' and cfg.mpc_params['use_RTI']):
                        # If the controller is gradient and is using RTI, we need to linearize the mpc after its computation
                        # this helps to minize the delay between new state->control in a real case scenario.
                        self.srbd_controller_interface.compute_RTI()
                    last_mpc_thread_time = time.time()


    def compute_mpc_process_queue_callback(self, input_data_process, output_data_process):
        pid = os.getpid()
        os.system("sudo renice -n -21 -p " + str(pid))
        os.system("sudo echo -20 > /proc/" + str(pid) + "/autogroup")
        #affinity_mask = {6, 7} 
        #os.sched_setaffinity(pid, affinity_mask)
        
        # This process runs forever!
        last_mpc_process_time = time.time()
        while True:
            #if time.time() - last_mpc_process_time > 1.0 / MPC_FREQ:
            if(not input_data_process.empty()):
                data = input_data_process.get()
                state_current = data[0]
                ref_state = data[1]
                contact_sequence = data[2]
                inertia = data[3]
                optimize_swing = data[4]
                phase_signal = data[5]
                step_freq = data[6]
                
                nmpc_GRFs,  \
                nmpc_footholds, \
                nmpc_joints_pos, \
                nmpc_joints_vel, \
                nmpc_joints_acc, \
                best_sample_freq,\
                nmpc_predicted_state = self.srbd_controller_interface.compute_control(state_current,
                                                                        ref_state,
                                                                        contact_sequence,
                                                                        inertia,
                                                                        phase_signal,
                                                                        step_freq,
                                                                        optimize_swing)
                
                
                last_mpc_loop_time = time.time() - last_mpc_process_time
                output_data_process.put([nmpc_GRFs, nmpc_footholds, nmpc_joints_pos, nmpc_joints_vel, nmpc_joints_acc, best_sample_freq, nmpc_predicted_state, last_mpc_loop_time])
                
                
                if(cfg.mpc_params['type'] != 'sampling' and cfg.mpc_params['use_RTI']):
                    # If the controller is gradient and is using RTI, we need to linearize the mpc after its computation
                    # this helps to minize the delay between new state->control in a real case scenario.
                    self.srbd_controller_interface.compute_RTI()
                last_mpc_process_time = time.time()


    def compute_mpc_process_shared_memory_callback(self, input_data_process, shm_out_name: str, seq_out: Value):
        pid = os.getpid()
        os.system("sudo renice -n -21 -p " + str(pid))
        os.system("sudo echo -20 > /proc/" + str(pid) + "/autogroup")
        #affinity_mask = {6, 7} 
        #os.sched_setaffinity(pid, affinity_mask)

        shm = shared_memory.SharedMemory(name=shm_out_name)
        arr = np.ndarray((N_DBL,), dtype=np.float64, buffer=shm.buf)

        last_mpc_process_time = time.time()
        while True:
            #if time.time() - last_mpc_process_time > 1.0 / MPC_FREQ:
            if(not input_data_process.empty()):
                data = input_data_process.get()
                state_current = data[0]
                ref_state = data[1]
                contact_sequence = data[2]
                inertia = data[3]
                optimize_swing = data[4]
                phase_signal = data[5]
                step_freq = data[6]
                
                nmpc_GRFs,  \
                nmpc_footholds, \
                nmpc_joints_pos, \
                nmpc_joints_vel, \
                nmpc_joints_acc, \
                best_sample_freq,\
                nmpc_predicted_state = self.srbd_controller_interface.compute_control(state_current,
                                                                        ref_state,
                                                                        contact_sequence,
                                                                        inertia,
                                                                        phase_signal,
                                                                        step_freq,
                                                                        optimize_swing)
                last_mpc_loop_time = time.time() - last_mpc_process_time

                # Publish to SHM with seqlock: odd=writing, even=stable
                s = seq_out.value
                if s % 2 == 0:
                    seq_out.value = s + 1
                # pack payload
                arr[IDX_GRF]  = legsattr_to12(nmpc_GRFs)
                arr[IDX_FH]   = legsattr_to12(nmpc_footholds)
                arr[IDX_JP]   = (nmpc_joints_pos if nmpc_predicted_state is not None else np.zeros(12).reshape(-1)[:12])
                arr[IDX_JV]   = (nmpc_joints_pos if nmpc_predicted_state is not None else np.zeros(12).reshape(-1)[:12])
                arr[IDX_JA]   = (nmpc_joints_pos if nmpc_predicted_state is not None else np.zeros(12).reshape(-1)[:12])
                arr[IDX_PRED] = np.asarray(nmpc_predicted_state).reshape(-1)[:12]
                arr[IDX_BSF]  = float(best_sample_freq)
                arr[IDX_LAST] = float(last_mpc_loop_time)
                arr[IDX_STAMP]= float(time.monotonic())
                # mark stable
                seq_out.value = (s | 1) + 1

                if cfg.mpc_params['type'] != 'sampling' and cfg.mpc_params['use_RTI']:
                    self.srbd_controller_interface.compute_RTI()


    def get_base_state_callback(self, msg):
        
        if(USE_SMOOTH_HEIGHT):
            # Smooth the height of the base
            self.position[2] = 0.5*self.position[2] + 0.5*np.array(msg.pose.position)[2]
        else:
            self.position[2] = np.array(msg.pose.position)[2]
        self.position[0:2] = np.array(msg.pose.position)[0:2]


        if(USE_SMOOTH_VELOCITY):
            self.linear_velocity = 0.5*self.linear_velocity + 0.5*np.array(msg.velocity.linear)
        else:
            self.linear_velocity = np.array(msg.velocity.linear)

        # For the quaternion, the order is [w, x, y, z] on mujoco, and [x, y, z, w] on DLS2
        self.orientation = np.roll(np.array(msg.pose.orientation), 1)
        # For the angular velocity, mujoco is in the base frame, and DLS2 is in the world frame
        self.angular_velocity = np.array(msg.velocity.angular) 


        self.first_message_base_arrived = True



    def get_blind_state_callback(self, msg):
        
        self.joint_positions = np.array(msg.joints_position)
        self.joint_velocities = np.array(msg.joints_velocity)
        self.feet_contact = np.array(msg.feet_contact)
        self.state_timestamp = float(msg.timestamp) if msg.timestamp > 0.0 else None

        if not self.down_pose_initialized and len(self.joint_positions) == 12:
            self.stand_up_and_down_actions.FL = self.joint_positions[0:3].copy()
            self.stand_up_and_down_actions.FR = self.joint_positions[3:6].copy()
            self.stand_up_and_down_actions.RL = self.joint_positions[6:9].copy()
            self.stand_up_and_down_actions.RR = self.joint_positions[9:12].copy()
            self.down_pose_initialized = True
            self.get_logger().info("Captured the measured Spot down pose.")

        self.first_message_joints_arrived = True
        

        if(not USE_SCHEDULER):
            self.compute_control_callback()



    def get_joy_callback(self, msg):
        """
        Callback function to handle joystick input. Joystick used is a 
        8Bitdi Ultimate 2C Wireless Controller.
        """
        self.env._ref_base_lin_vel_H[0] = msg.axes[1]/3.5  # Forward/Backward
        self.env._ref_base_lin_vel_H[1] = msg.axes[0]/3.5  # Left/Right
        self.env._ref_base_ang_yaw_dot = msg.axes[3]/2.  # Yaw


        #kill the node if the button is pressed
        if msg.buttons[8] == 1:
            self.get_logger().info("Joystick button pressed, shutting down the node.") 
            # This will kill the robot hal
            os.system("kill -9 $(ps -u | grep -m 1 hal | grep -o \"^[^ ]* *[0-9]*\" | grep -o \"[0-9]*\")")
            # This will kill the process running this script
            os.system("pkill -f play_ros2.py") 
            exit(0)




    def compute_control_callback(self):
        
        # Update the loop time
        if self.state_timestamp is not None:
            nominal_dt = cfg.simulation_params.get(
                'ros_interface_dt', cfg.simulation_params['dt']
            )
            if self.last_state_timestamp is None:
                simulation_dt = nominal_dt
            else:
                simulation_dt = self.state_timestamp - self.last_state_timestamp
                if simulation_dt <= 0.0 or simulation_dt > 0.05:
                    simulation_dt = nominal_dt
            self.last_state_timestamp = self.state_timestamp
            self.loop_time = simulation_dt
        elif(USE_FIXED_LOOP_TIME):
            simulation_dt = 1./SCHEDULER_FREQ
        else:
            start_time = time.perf_counter()
            if(self.last_start_time is not None):
                self.loop_time = (start_time - self.last_start_time)
            self.last_start_time = start_time
            simulation_dt = self.loop_time
            
            if(USE_SATURATED_LOOP_TIME):
                if(simulation_dt > 0.005):
                    simulation_dt = 0.005

        # Safety check to not do anything until a first base and blind state are received
        if(self.first_message_base_arrived==False or self.first_message_joints_arrived==False):
            return

        
        # Update the mujoco model
        self.env.mjData.qpos[0:3] = copy.deepcopy(self.position) # s.e. height
        #self.env.mjData.qpos[0:3] = np.zeros(3) # proprioceptive height
        self.env.mjData.qpos[3:7] = copy.deepcopy(self.orientation)
        self.env.mjData.qvel[0:3] = copy.deepcopy(self.linear_velocity)
        self.env.mjData.qvel[3:6] = copy.deepcopy(self.angular_velocity)
        self.env.mjData.qpos[7:] = copy.deepcopy(self.joint_positions)
        self.env.mjData.qvel[6:] = copy.deepcopy(self.joint_velocities)
        self.env.mjModel.opt.timestep = simulation_dt
        self.env.mjModel.opt.disableflags = 16 # Disable the collision detection
        mujoco.mj_forward(self.env.mjModel, self.env.mjData)   

        # Visualize the current mjData state at a limited rate.
        if RENDER_MUJOCO_VIEWER and (time.time() - self.last_render_time > 1.0 / RENDER_FREQ):
            self.env.render()
            self.last_render_time = time.time()


        # And get the state of the robot
        legs_order = ["FL", "FR", "RL", "RR"]
        feet_pos = self.env.feet_pos(frame='world')
        feet_vel = self.env.feet_vel(frame='world')
        hip_pos = self.env.hip_positions(frame='world')
        base_lin_vel = self.env.base_lin_vel(frame='world')
        base_ang_vel = self.env.base_ang_vel(frame='base')
        base_ori_euler_xyz = self.env.base_ori_euler_xyz
        base_pos = self.env.base_pos
        com_pos = self.env.com


        # Get the reference base velocity in the world frame
        ref_base_lin_vel, ref_base_ang_vel = self.env.target_base_vel()
        
        # Get the inertia matrix
        if(cfg.simulation_params['use_inertia_recomputation']):
            inertia = self.env.get_base_inertia().flatten()  # Reflected inertia of base at qpos, in world frame
        else:
            inertia = cfg.inertia.flatten()

        # Get the qpos and qvel
        qpos, qvel = self.env.mjData.qpos, self.env.mjData.qvel
        joints_pos = LegsAttr(FL=qpos[7:10], FR=qpos[10:13],
                                RL=qpos[13:16], RR=qpos[16:19])
        
        # Get Centrifugal, Coriolis, Gravity, Friction for the swing controller
        legs_mass_matrix = self.env.legs_mass_matrix
        legs_qfrc_bias = self.env.legs_qfrc_bias
        legs_qfrc_passive = self.env.legs_qfrc_passive

        # Compute feet jacobian
        feet_jac = self.env.feet_jacobians(frame='world', return_rot_jac=False)
        feet_jac_dot = self.env.feet_jacobians_dot(frame='world', return_rot_jac=False)


        # Idx of the leg
        legs_qvel_idx = self.env.legs_qvel_idx
        legs_qpos_idx = self.env.legs_qpos_idx

        # Get the heightmaps
        heightmaps = None

        
        # Update the state and reference -------------------------
        state_current, \
        ref_state, \
        contact_sequence, \
        step_height, \
        optimize_swing = self.wb_interface.update_state_and_reference(com_pos,
                                                base_pos,
                                                base_lin_vel,
                                                base_ori_euler_xyz,
                                                base_ang_vel,
                                                feet_pos,
                                                hip_pos,
                                                joints_pos,
                                                heightmaps,
                                                legs_order,
                                                simulation_dt,
                                                ref_base_lin_vel,
                                                ref_base_ang_vel)


        
        # Console commands hacks
        ref_state["ref_position"][2] += self.console.height_delta
        ref_state["ref_orientation"][1] += self.console.pitch_delta


        
        # Publish to the MPC controller
        if(USE_THREADED_MPC):
            self.state_current = state_current
            self.ref_state = ref_state
            self.contact_sequence = contact_sequence
            self.inertia = inertia
            self.optimize_swing = optimize_swing

        elif(USE_PROCESS_QUEUE_MPC):
            if(not self.input_data_process.full()):
                self.input_data_process.put_nowait([state_current, ref_state, contact_sequence, inertia, optimize_swing, self.wb_interface.pgg.phase_signal, self.wb_interface.pgg.step_freq])

            if(not self.output_data_process.empty()):
                data = self.output_data_process.get_nowait()
                self.nmpc_GRFs = data[0]
                self.nmpc_footholds = data[1]
                self.nmpc_joints_pos = data[2]
                self.nmpc_joints_vel = data[3]
                self.nmpc_joints_acc = data[4]
                self.best_sample_freq = data[5]
                self.nmpc_predicted_state = data[6]
                self.last_mpc_loop_time = data[7]
        
        elif(USE_PROCESS_SHARED_MEMORY_MPC):
            if(not self.input_data_process.full()):
                self.input_data_process.put_nowait([state_current, ref_state, contact_sequence, inertia, optimize_swing, self.wb_interface.pgg.phase_signal, self.wb_interface.pgg.step_freq])
            
            # Read MPC output from shared memory with seqlock and stale-data guard
            if self.shm_out is not None and self.seq_out is not None:
                s1 = self.seq_out.value
                if s1 % 2 == 0:  # writer not in progress
                    buf = np.ndarray((N_DBL,), dtype=np.float64, buffer=self.shm_out.buf)
                    tmp = buf.copy()  # local copy
                    s2 = self.seq_out.value
                    if s1 == s2 and (s2 % 2 == 0):
                        self.nmpc_GRFs        = vec12_to_legsattr(tmp[IDX_GRF])
                        self.nmpc_footholds   = vec12_to_legsattr(tmp[IDX_FH])
                        self.nmpc_joints_pos  = vec12_to_legsattr(tmp[IDX_JP])
                        self.nmpc_joints_vel  = vec12_to_legsattr(tmp[IDX_JV])
                        self.nmpc_joints_acc  = vec12_to_legsattr(tmp[IDX_JA])
                        self.nmpc_predicted_state = tmp[IDX_PRED].copy()
                        self.best_sample_freq  = float(tmp[IDX_BSF])
                        self.last_mpc_loop_time = float(tmp[IDX_LAST])
                        self.last_mpc_update_mono = float(tmp[IDX_STAMP])
                        
        else:
            if time.time() - self.last_mpc_time > 1.0 / MPC_FREQ:
                self.nmpc_GRFs,  \
                self.nmpc_footholds, \
                self.nmpc_joints_pos, \
                self.nmpc_joints_vel, \
                self.nmpc_joints_acc, \
                self.best_sample_freq, \
                self.nmpc_predicted_state = self.srbd_controller_interface.compute_control(state_current,
                                                                        ref_state,
                                                                        contact_sequence,
                                                                        inertia,
                                                                        self.wb_interface.pgg.phase_signal,
                                                                        self.wb_interface.pgg.step_freq,
                                                                        optimize_swing)
                
                if(cfg.mpc_params['type'] != 'sampling' and cfg.mpc_params['use_RTI']):
                    # If the controller is gradient and is using RTI, we need to linearize the mpc after its computation
                    # this helps to minize the delay between new state->control in a real case scenario.
                    self.srbd_controller_interface.compute_RTI()
        
                    
                self.last_mpc_time = time.time()
                
        
        
        # Compute Swing and Stance Torque ---------------------------------------------------------------------------
        self.tau, \
        pd_target_joints_pos, \
        pd_target_joints_vel = self.wb_interface.compute_stance_and_swing_torque(simulation_dt,
                                                                                qpos,
                                                                                qvel,
                                                                                feet_jac,
                                                                                feet_jac_dot,
                                                                                feet_pos,
                                                                                feet_vel,
                                                                                legs_qfrc_passive,
                                                                                legs_qfrc_bias,
                                                                                legs_mass_matrix,
                                                                                self.nmpc_GRFs,
                                                                                self.nmpc_footholds,
                                                                                legs_qpos_idx,
                                                                                legs_qvel_idx,
                                                                                self.tau,
                                                                                optimize_swing,
                                                                                self.best_sample_freq,
                                                                                self.nmpc_joints_pos,
                                                                                self.nmpc_joints_vel,
                                                                                self.nmpc_joints_acc, 
                                                                                self.nmpc_predicted_state)
        

        # Limit tau between tau_limits
        for leg in ["FL", "FR", "RL", "RR"]:
            tau_min, tau_max = self.tau_limits[leg][:, 0], self.tau_limits[leg][:, 1]
            self.tau[leg] = np.clip(self.tau[leg], tau_min, tau_max)

        # Abort the joint-PD part before a small imbalance becomes a rollover.
        # The console thread notices transition_abort and stops interpolating;
        # hold the measured pose instead of continuing to push toward home.
        if (
            self.console.transition_active
            and self.console.isDown
            and self.console.support_ramp_start is None
            and self.console.transition_abort is None
        ):
            pd_unsafe_reason = None
            if abs(base_ori_euler_xyz[0]) > 0.35 or abs(base_ori_euler_xyz[1]) > 0.35:
                pd_unsafe_reason = "PD stand-up base tilt exceeded 0.35 rad"
            elif abs(base_lin_vel[2]) > 0.75:
                pd_unsafe_reason = "PD stand-up vertical velocity exceeded 0.75 m/s"
            elif np.max(np.abs(self.joint_velocities)) > 8.0:
                joint_labels = [
                    'fl_hx', 'fl_hy', 'fl_kn',
                    'fr_hx', 'fr_hy', 'fr_kn',
                    'hl_hx', 'hl_hy', 'hl_kn',
                    'hr_hx', 'hr_hy', 'hr_kn',
                ]
                fastest_joint = int(np.argmax(np.abs(self.joint_velocities)))
                pd_unsafe_reason = (
                    "PD stand-up joint velocity exceeded 8 rad/s: "
                    f"{joint_labels[fastest_joint]}="
                    f"{self.joint_velocities[fastest_joint]:.2f} rad/s"
                )

            if pd_unsafe_reason is not None:
                self.console.transition_abort = pd_unsafe_reason
                self.console.support_fault = pd_unsafe_reason
                self.stand_up_and_down_actions.FL = self.joint_positions[0:3].copy()
                self.stand_up_and_down_actions.FR = self.joint_positions[3:6].copy()
                self.stand_up_and_down_actions.RL = self.joint_positions[6:9].copy()
                self.stand_up_and_down_actions.RR = self.joint_positions[9:12].copy()
                self.get_logger().error(pd_unsafe_reason)

        # Once goUp has completed, progressively hand control back to the MPC.
        # During the joint-space stand-up itself support_scale stays at zero,
        # because the centroidal model is invalid with the trunk on the floor.
        if self.console.support_ramp_start is not None:
            unsafe_reason = None
            ramp_elapsed = time.monotonic() - self.console.support_ramp_start
            ramp_alpha = np.clip(
                ramp_elapsed / self.console.support_ramp_duration, 0.0, 1.0
            )
            candidate_support_scale = (
                ramp_alpha * ramp_alpha * (3.0 - 2.0 * ramp_alpha)
            )
            home_target = np.concatenate([
                self.stand_up_and_down_actions.FL,
                self.stand_up_and_down_actions.FR,
                self.stand_up_and_down_actions.RL,
                self.stand_up_and_down_actions.RR,
            ]).reshape(-1)
            wbc_target = np.concatenate([
                pd_target_joints_pos.FL,
                pd_target_joints_pos.FR,
                pd_target_joints_pos.RL,
                pd_target_joints_pos.RR,
            ]).reshape(-1)
            # During the ramp the commanded position is already moving from
            # home to the WBC inverse-kinematics target.  Comparing the robot
            # with home here falsely treated that intentional motion as a
            # tracking fault and abruptly removed support near scale=0.45.
            commanded_target = (
                (1.0 - candidate_support_scale) * home_target
                + candidate_support_scale * wbc_target
            )
            tracking_error = commanded_target - self.joint_positions
            worst_joint = int(np.argmax(np.abs(tracking_error)))
            joint_labels = [
                'fl_hx', 'fl_hy', 'fl_kn',
                'fr_hx', 'fr_hy', 'fr_kn',
                'hl_hx', 'hl_hy', 'hl_kn',
                'hr_hx', 'hr_hy', 'hr_kn',
            ]
            if abs(tracking_error[worst_joint]) > 0.35:
                unsafe_reason = (
                    f"ramp tracking error: {joint_labels[worst_joint]}="
                    f"{tracking_error[worst_joint]:+.2f} rad"
                )
            elif abs(base_ori_euler_xyz[0]) > 0.35 or abs(base_ori_euler_xyz[1]) > 0.35:
                unsafe_reason = "base tilt exceeded 0.35 rad"
            elif abs(base_lin_vel[2]) > 0.75:
                unsafe_reason = "vertical velocity exceeded 0.75 m/s"
            elif np.max(np.abs(self.joint_velocities)) > 8.0:
                unsafe_reason = "joint velocity exceeded 8 rad/s"

            if unsafe_reason is not None:
                self.console.support_scale = 0.0
                self.console.support_ramp_start = None
                self.console.support_fault = unsafe_reason
                self.get_logger().error(
                    f"MPC support disabled during goUp: {unsafe_reason}"
                )
            else:
                self.console.support_scale = candidate_support_scale
                if ramp_alpha >= 1.0:
                    self.console.support_scale = 1.0
                    self.console.support_ramp_start = None

        # Guard the first stationary MPC hold after goUp.  Previously there
        # was no safety monitoring after the ramp had reached 1.0, so a height
        # oscillation could grow until Spot lost every ground contact.  Freeze
        # the measured pose and restore quasi-static support on the first clear
        # instability; walking has its own expected joint velocities and is
        # deliberately excluded from this bring-up guard.
        if (
            not self.console.transition_active
            and not self.console.isDown
            and not self.console.walking
            and self.console.support_scale >= 0.99
        ):
            standing_fault = None
            max_tilt = float(
                cfg.simulation_params.get('standing_safety_max_tilt', 0.25)
            )
            max_joint_velocity = float(
                cfg.simulation_params.get(
                    'standing_safety_max_joint_velocity', 5.0
                )
            )
            height_margin = float(
                cfg.simulation_params.get(
                    'standing_safety_height_margin', 0.10
                )
            )
            if (
                abs(base_ori_euler_xyz[0]) > max_tilt
                or abs(base_ori_euler_xyz[1]) > max_tilt
            ):
                standing_fault = (
                    f"stationary base tilt is roll={base_ori_euler_xyz[0]:+.2f}, "
                    f"pitch={base_ori_euler_xyz[1]:+.2f} rad"
                )
            elif abs(base_pos[2] - cfg.simulation_params['ref_z']) > height_margin:
                standing_fault = (
                    f"stationary base height is {base_pos[2]:.3f} m "
                    f"(reference {cfg.simulation_params['ref_z']:.3f} m)"
                )
            elif np.max(np.abs(self.joint_velocities)) > max_joint_velocity:
                fastest_joint = int(np.argmax(np.abs(self.joint_velocities)))
                joint_labels = [
                    'fl_hx', 'fl_hy', 'fl_kn',
                    'fr_hx', 'fr_hy', 'fr_kn',
                    'hl_hx', 'hl_hy', 'hl_kn',
                    'hr_hx', 'hr_hy', 'hr_kn',
                ]
                standing_fault = (
                    f"stationary joint velocity: {joint_labels[fastest_joint]}="
                    f"{self.joint_velocities[fastest_joint]:+.2f} rad/s"
                )

            if standing_fault is not None:
                self.console.support_scale = 0.0
                self.console.support_fault = standing_fault
                self.console.standup_gravity_scale = 1.0
                self.stand_up_and_down_actions.FL = self.joint_positions[0:3].copy()
                self.stand_up_and_down_actions.FR = self.joint_positions[3:6].copy()
                self.stand_up_and_down_actions.RL = self.joint_positions[6:9].copy()
                self.stand_up_and_down_actions.RR = self.joint_positions[9:12].copy()
                self.get_logger().error(
                    f"MPC stationary hold disabled: {standing_fault}"
                )

            now = time.monotonic()
            if now - self.last_standing_log_time >= 0.5:
                fastest_joint = int(np.argmax(np.abs(self.joint_velocities)))
                joint_labels = [
                    'fl_hx', 'fl_hy', 'fl_kn',
                    'fr_hx', 'fr_hy', 'fr_kn',
                    'hl_hx', 'hl_hy', 'hl_kn',
                    'hr_hx', 'hr_hy', 'hr_kn',
                ]
                standing_torque = np.concatenate([
                    self.tau.FL, self.tau.FR, self.tau.RL, self.tau.RR
                ]).reshape(-1)
                strongest_joint = int(np.argmax(np.abs(standing_torque)))
                self.get_logger().info(
                    "standing: base_z=%.3f, roll=%.2f, pitch=%.2f, "
                    "max_qd=%s:%+.2f, max_tau=%s:%+.1f, contacts=%s"
                    % (
                        base_pos[2],
                        base_ori_euler_xyz[0],
                        base_ori_euler_xyz[1],
                        joint_labels[fastest_joint],
                        self.joint_velocities[fastest_joint],
                        joint_labels[strongest_joint],
                        standing_torque[strongest_joint],
                        self.feet_contact.astype(int).tolist(),
                    )
                )
                self.last_standing_log_time = now

        if(self.console.isDown):
            pd_target_joints_pos = self.stand_up_and_down_actions
            pd_target_joints_vel.FL = pd_target_joints_vel.FL*0.0
            pd_target_joints_vel.FR = pd_target_joints_vel.FR*0.0
            pd_target_joints_vel.RL = pd_target_joints_vel.RL*0.0
            pd_target_joints_vel.RR = pd_target_joints_vel.RR*0.0

        elif self.console.support_scale < 1.0:
            # Avoid a target-position discontinuity when leaving the explicit
            # home pose and returning to the WBC inverse-kinematics target.
            blend = self.console.support_scale
            for leg in ["FL", "FR", "RL", "RR"]:
                pd_target_joints_pos[leg] = (
                    (1.0 - blend) * self.stand_up_and_down_actions[leg]
                    + blend * pd_target_joints_pos[leg]
                )
                pd_target_joints_vel[leg] = blend * pd_target_joints_vel[leg]

        for leg in ["FL", "FR", "RL", "RR"]:
            self.tau[leg] = self.console.support_scale * self.tau[leg]

        # During stand-up, replace the invalid MPC force with quasi-static
        # weight compensation.  Do not assume m*g/4 on every foot: when the
        # CoM is not exactly halfway between the axles that creates a pitch
        # moment and makes Spot slide nose-first.  Choose the four vertical
        # forces so that they support the weight and their resultant passes
        # through the model CoM, then add a small pitch-PD correction.
        # While MPC support ramps in, blend this term out by (1-support_scale).
        static_support_scale = (
            self.console.standup_gravity_scale
            * (1.0 - self.console.support_scale)
        )
        static_front_fz = 0.0
        static_rear_fz = 0.0
        if static_support_scale > 0.0:
            leg_names = ["FL", "FR", "RL", "RR"]
            weight = cfg.mass * cfg.gravity_constant

            # Project foot-to-CoM vectors onto the robot's horizontal forward
            # direction.  This remains valid if the robot has a non-zero yaw.
            yaw = base_ori_euler_xyz[2]
            forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            foot_levers = np.array([
                np.dot(np.asarray(feet_pos[leg]).reshape(3) - com_pos, forward)
                for leg in leg_names
            ])

            # For an upward foot force, tau_pitch = -lever_x * Fz.  Therefore
            # sum(lever_x*Fz) = kp*pitch + kd*pitch_rate produces the restoring
            # pitch torque.  Clamp only the feedback moment; the CoM balancing
            # part remains exact.
            pitch_moment = (
                float(cfg.simulation_params.get('standup_pitch_kp', 40.0))
                * base_ori_euler_xyz[1]
                + float(cfg.simulation_params.get('standup_pitch_kd', 4.0))
                * base_ang_vel[1]
            )
            pitch_moment_limit = float(
                cfg.simulation_params.get('standup_pitch_moment_limit', 15.0)
            )
            pitch_moment = np.clip(
                pitch_moment, -pitch_moment_limit, pitch_moment_limit
            )

            force_constraints = np.vstack([
                np.ones(4),
                foot_levers,
            ])
            desired_resultant = np.array([weight, pitch_moment])
            nominal_fz = np.full(4, weight / 4.0)
            correction = force_constraints.T @ np.linalg.pinv(
                force_constraints @ force_constraints.T
            ) @ (desired_resultant - force_constraints @ nominal_fz)
            foot_fz = np.clip(nominal_fz + correction, 0.0, weight)
            static_front_fz = float(
                static_support_scale * (foot_fz[0] + foot_fz[1])
            )
            static_rear_fz = float(
                static_support_scale * (foot_fz[2] + foot_fz[3])
            )

            for leg, fz in zip(leg_names, foot_fz):
                foot_force = np.array([0.0, 0.0, fz])
                static_tau = -np.matmul(
                    feet_jac[leg][:, legs_qvel_idx[leg]].T,
                    foot_force,
                )
                # Let the hx joint-space PD hold the measured splayed pose.
                # Direct vertical-force mapping on hx accelerated the lateral
                # joints while the feet/trunk were constrained by the floor.
                static_tau[0] = 0.0
                self.tau[leg] = (
                    self.tau[leg] + static_support_scale * static_tau
                )

        if self.console.transition_active:
            now = time.monotonic()
            if now - self.last_standup_log_time >= 0.5:
                target = np.concatenate([
                    pd_target_joints_pos.FL,
                    pd_target_joints_pos.FR,
                    pd_target_joints_pos.RL,
                    pd_target_joints_pos.RR,
                ]).reshape(-1)
                joint_labels = [
                    'fl_hx', 'fl_hy', 'fl_kn',
                    'fr_hx', 'fr_hy', 'fr_kn',
                    'hl_hx', 'hl_hy', 'hl_kn',
                    'hr_hx', 'hr_hy', 'hr_kn',
                ]
                joint_error = target - self.joint_positions
                worst_joint = int(np.argmax(np.abs(joint_error)))
                torque = np.concatenate([
                    self.tau.FL, self.tau.FR, self.tau.RL, self.tau.RR
                ]).reshape(-1)
                self.get_logger().info(
                    "goUp: support=%.2f, base_z=%.3f, roll=%.2f, pitch=%.2f, "
                    "max_q_error=%s:%+.2f, max_feedforward=%.1f, "
                    "front/rear_fz=%.1f/%.1f"
                    % (
                        self.console.support_scale,
                        base_pos[2],
                        base_ori_euler_xyz[0],
                        base_ori_euler_xyz[1],
                        joint_labels[worst_joint],
                        joint_error[worst_joint],
                        np.max(np.abs(torque)),
                        static_front_fz,
                        static_rear_fz,
                    )
                )
                self.last_standup_log_time = now




        trajectory_generator_msg = TrajectoryGenerator()
        if self.state_timestamp is not None:
            trajectory_generator_msg.timestamp = self.state_timestamp
        trajectory_generator_msg.joints_position = np.concatenate([pd_target_joints_pos.FL, pd_target_joints_pos.FR, pd_target_joints_pos.RL, pd_target_joints_pos.RR], axis=0).flatten().tolist()
        trajectory_generator_msg.joints_velocity = np.concatenate([pd_target_joints_vel.FL, pd_target_joints_vel.FR, pd_target_joints_vel.RL, pd_target_joints_vel.RR], axis=0).flatten().tolist()
        if self.console.isDown or self.console.support_scale < 1.0:
            gain_blend = 0.0 if self.console.isDown else self.console.support_scale
            kp = (
                (1.0 - gain_blend) * self.standup_joint_position_gain
                + gain_blend * self.impedence_joint_position_gain
            )
            kd = (
                (1.0 - gain_blend) * self.standup_joint_velocity_gain
                + gain_blend * self.impedence_joint_velocity_gain
            )
        else:
            kp = self.impedence_joint_position_gain
            kd = self.impedence_joint_velocity_gain
        trajectory_generator_msg.kp = kp.tolist()
        trajectory_generator_msg.kd = kd.tolist()
        self.publisher_trajectory_generator.publish(trajectory_generator_msg)

        # Publish the matching feed-forward command only after its PD target.
        # The translator rejects commands whose trajectory timestamp differs.
        control_signal_msg = ControlSignal()
        if self.state_timestamp is not None:
            control_signal_msg.timestamp = self.state_timestamp
        control_signal_msg.torques = np.concatenate([self.tau.FL, self.tau.FR, self.tau.RL, self.tau.RR], axis=0).flatten().tolist()
        self.publisher_control_signal.publish(control_signal_msg)

        time_debug_msg = TimeDebug()
        time_debug_msg.time_wbc = self.loop_time
        time_debug_msg.time_mpc = self.last_mpc_loop_time
        self.publisher_time_debug.publish(time_debug_msg)



def main():
    print('Hello from Quadruped-PyMPC ros interface.')
    rclpy.init()

    controller_node = Quadruped_PyMPC_Node()

    rclpy.spin(controller_node)
    controller_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
