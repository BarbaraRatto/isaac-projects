import readline
import readchar
import time
import numpy as np
import copy

# Config imports
from quadruped_pympc import config as cfg

from gym_quadruped.utils.quadruped_utils import LegsAttr
import mujoco

class Console():
    def __init__(self, controller_node):
        self.controller_node = controller_node

        # Walking and Stopping
        self.walking = False

        # Go Up and Go Down motion
        self.isDown = not cfg.simulation_params.get('robot_starts_up', False)
        self.height_delta = (
            -cfg.simulation_params['ref_z'] if self.isDown else 0.0
        )
        self.transition_active = False
        self.support_scale = 0.0 if self.isDown else 1.0
        self.support_ramp_start = None
        self.support_fault = None
        self.transition_abort = None
        self.standup_gravity_scale = 0.0
        self.support_ramp_duration = float(
            cfg.simulation_params.get('mpc_support_ramp_duration', 3.0)
        )

        # Pitch Up and Pitch Down
        self.pitch_delta = 0

        # Step Height holder to keep track of the step height
        self.step_height_holder = cfg.simulation_params['step_height']

        # Repeated key presses change a velocity setpoint; they are not
        # individual steps. Keep bring-up commands inside a conservative range.
        self.max_forward_speed = 0.6
        self.max_lateral_speed = 0.3
        self.max_yaw_rate = 0.6

        # Autocomplete setup
        self.commands = [
            "help",
            "stw",
            "ooo",
            "ictp",
            "narrowStance",
            "wideStance",
            "goUp",
            "goDown",
            "setGaitTimer",
            "setupGaitTimer",
            "setupLegsGains",
            "setupGeneral"
        ]
        readline.set_completer(self.complete)
        readline.parse_and_bind("tab: complete")


    def complete(self, text, state):
        options = [cmd for cmd in self.commands if cmd.startswith(text)]
        if state < len(options):
            print(options[state])
            return options[state]
        else:
            return None


    def interactive_command_line(self, ):
        self.print_all_commands()
        while True:
            input_string = input(">>> ")
            try:
                if(input_string == "stw"):
                    if(self.isDown == True):
                        print("The robot is down, please go up before starting to walk")
                    elif(self.support_fault is not None or self.support_scale < 0.99):
                        print("MPC support is not ready; reset the simulation before walking")
                    else:
                        if(self.walking):
                            print("The robot is already walking")
                        print("Starting Walking")
                        self.walking = True
                        self.controller_node.wb_interface.pgg.gait_type = self.controller_node.wb_interface.pgg.previous_gait_type
                        self.controller_node.wb_interface.pgg.reset()
                

                elif(input_string == "ooo"):
                    print("Stopping Walking")
                    self.walking = False
                    self.controller_node.wb_interface.pgg.gait_type = 7 # FULL_STANCE
                

                elif(input_string == "narrowStance"):
                    print("Narrow Stance")
                    self.controller_node.wb_interface.frg.hip_offset -= 0.03 
                

                elif(input_string == "wideStance"):
                    print("Wide Stance")
                    self.controller_node.wb_interface.frg.hip_offset += 0.03 


                elif(input_string == "setGaitTimer"):
                    print("Press one of the following numbers to set the gait type")
                    print("0: TROT")
                    print("1: PACE")
                    print("2: BOUNDING")
                    print("3: CIRCULARCRAWL")
                    print("4: BFDIAGONALCRAWL")
                    print("5: BACKDIAGONALCRAWL")
                    print("6: FRONTDIAGONALCRAWL")
                    print("7: FULL_STANCE")

                    if(self.walking):
                        print("Please stop the robot before changing the gait type")
                        continue
                    
                    gait_type = int(input("Gait Type: >>> "))
                    if(gait_type == 7):
                        gait_name = "full_stance"
                    elif(gait_type == 0):
                        gait_name = "trot"
                    elif(gait_type == 1):
                        gait_name = "pace"
                    elif(gait_type == 2):
                        gait_name = "bound"
                    elif(gait_type == 3):
                        gait_name = "crawl"
                    elif(gait_type == 4):
                        gait_name = "crawl"
                    elif(gait_type == 5):
                        gait_name = "crawl"
                    elif(gait_type == 6):
                        gait_name = "crawl"


                    if(gait_type >= 0 and gait_type <= 7):
                        gait_params = cfg.simulation_params['gait_params'][gait_name]
                        gait_type, duty_factor, step_frequency = gait_params['type'], gait_params['duty_factor'], gait_params['step_freq']
                        
                        
                        self.controller_node.wb_interface.pgg.step_freq = step_frequency
                        self.controller_node.wb_interface.pgg.duty_factor = duty_factor
                        self.controller_node.wb_interface.pgg.gait_type = gait_type
                        self.controller_node.wb_interface.pgg.previous_gait_type = gait_type
                        self.controller_node.wb_interface.pgg.reset()
                        
                        self.controller_node.wb_interface.frg.stance_time = (1 / self.controller_node.wb_interface.pgg.step_freq) * self.controller_node.wb_interface.pgg.duty_factor
                        swing_period = (1 - self.controller_node.wb_interface.pgg.duty_factor) * (1 / self.controller_node.wb_interface.pgg.step_freq)
                        self.controller_node.wb_interface.stc.regenerate_swing_trajectory_generator(step_height=self.step_height_holder, swing_period=swing_period)
                        
                    else:
                        print("Invalid Gait Type")

                
                elif(input_string == "setupGaitTimer"):
                    
                    print("Current Step Frequency: ", self.controller_node.wb_interface.pgg.step_freq)
                    temp = input("Step Frequency: >>> ")
                    if(temp != ""):
                        temp = max(0.4, min(float(temp), 2.0))
                        self.controller_node.wb_interface.pgg.step_freq = temp
                        self.controller_node.wb_interface.frg.stance_time = (1 / self.controller_node.wb_interface.pgg.step_freq) * self.controller_node.wb_interface.pgg.duty_factor
                        swing_period = (1 - self.controller_node.wb_interface.pgg.duty_factor) * (1 / self.controller_node.wb_interface.pgg.step_freq)
                        self.controller_node.wb_interface.stc.regenerate_swing_trajectory_generator(step_height=self.step_height_holder, swing_period=swing_period)

                    
                    print("Current Duty Factor: ", self.controller_node.wb_interface.pgg.duty_factor)
                    temp = input("Duty Factor: >>> ")
                    if(temp != ""):
                        temp = max(0.4, min(float(temp), 0.9))
                        self.controller_node.wb_interface.pgg.duty_factor = temp
                        self.controller_node.wb_interface.frg.stance_time = (1 / self.controller_node.wb_interface.pgg.step_freq) * self.controller_node.wb_interface.pgg.duty_factor
                        swing_period = (1 - self.controller_node.wb_interface.pgg.duty_factor) * (1 / self.controller_node.wb_interface.pgg.step_freq)
                        self.controller_node.wb_interface.stc.regenerate_swing_trajectory_generator(step_height=self.step_height_holder, swing_period=swing_period)

                    print("Start and Stop Gait: ", self.controller_node.wb_interface.pgg.start_and_stop_activated)
                    temp = input("Start and Stop Gait: >>> ")
                    if(temp != ""):
                        if(temp == "True"):
                            self.controller_node.wb_interface.pgg.start_and_stop_activated = True
                        elif(temp == "False"):
                            self.controller_node.wb_interface.pgg.start_and_stop_activated = False

                elif(input_string == "setupLegsGains"):

                    print("Current Gains Swing Kp: ", self.controller_node.wb_interface.stc.position_gain_fb)
                    temp = input("Gains Swing Kp: >>> ")
                    if(temp != ""):
                        self.controller_node.wb_interface.stc.position_gain_fb = float(temp)
                    
                    
                    print("Current Gains Swing Kd: ", self.controller_node.wb_interface.stc.velocity_gain_fb)
                    temp = input("Gains Swing Kd: >>> ")
                    if(temp != ""):
                        self.controller_node.wb_interface.stc.velocity_gain_fb = float(temp)

                    print("Current Gain Stance Kp: ", self.controller_node.impedence_joint_position_gain)
                    temp = input("Gain Stance Kp: >>> ")
                    if(temp != ""):
                        self.controller_node.impedence_joint_position_gain = np.ones(12)*float(temp)
                    
                    print("Current Gain Stance Kd: ", self.controller_node.impedence_joint_velocity_gain)
                    temp = input("Gain Stance Kd: >>> ")
                    if(temp != ""):
                        self.controller_node.impedence_joint_velocity_gain = np.ones(12)*float(temp)

                elif(input_string == "setupGeneral"):
                    
                    print("Current Base Height: ", cfg.simulation_params['ref_z'] + self.height_delta)
                    height_temp = input("CoM Height: >>> ")
                    if(height_temp != ""):
                        height_delta_temp = float(height_temp) - cfg.simulation_params['ref_z']
                        min_value = -0.1
                        max_value = 0.1
                        self.height_delta = max(min_value, min(height_delta_temp, max_value))


                    print("Current Step Height: ", self.controller_node.wb_interface.stc.swing_generator.step_height)
                    step_height_temp = input("Step Height: >>> ")
                    if(step_height_temp != ""):
                        self.step_height_holder = max(0.05, min(float(step_height_temp), 0.25))
                        swing_period_temp =  self.controller_node.wb_interface.stc.swing_period
                        self.controller_node.wb_interface.stc.regenerate_swing_trajectory_generator(self.step_height_holder, swing_period_temp)
                    
                    
                    print("Use FeedbackLin: ", self.controller_node.wb_interface.stc.use_feedback_linearization)
                    temp = input("Use FeedbackLin: >>> ")
                    if(temp != ""):
                        if(temp == "True"):
                            self.controller_node.wb_interface.stc.use_feedback_linearization = True
                        elif(temp == "False"):
                            self.controller_node.wb_interface.stc.use_feedback_linearization = False

                    
                    print("Use Friction Compensation: ", self.controller_node.wb_interface.stc.use_friction_compensation)
                    temp = input("Use Friction Compensation: >>> ")
                    if(temp != ""):
                        if(temp == "True"):
                            self.controller_node.wb_interface.stc.use_friction_compensation = True
                        elif(temp == "False"):
                            self.controller_node.wb_interface.stc.use_friction_compensation = False
                    
                    

                    print("Use Integrators in MPC: ", self.controller_node.srbd_controller_interface.controller.use_integrators)
                    temp = input("Use Integrators in MPC: >>> ")
                    if(temp != ""):
                        if(temp == "True"):
                            self.controller_node.srbd_controller_interface.controller.use_integrators = True
                        elif(temp == "False"):
                            self.controller_node.srbd_controller_interface.controller.use_integrators = False

                    
                    print("Com Offset: ", self.controller_node.wb_interface.frg.com_pos_offset_b)
                    temp = input("Set CoM offset x: >>> ")
                    if(temp != ""):
                        temp = max(-0.1, min(float(temp), 0.1))
                        self.controller_node.wb_interface.frg.com_pos_offset_b[0] = float(temp)
                    temp = input("Set CoM offset y: >>> ")
                    if(temp != ""):
                        temp = max(-0.1, min(float(temp), 0.1))
                        self.controller_node.wb_interface.frg.com_pos_offset_b[1] = float(temp)
                    temp = input("Set CoM offset z: >>> ")
                    if(temp != ""):
                        temp = max(-0.1, min(float(temp), 0.1))
                        self.controller_node.wb_interface.frg.com_pos_offset_b[2] = float(temp)

                    print("Use Reflexes: ", self.controller_node.wb_interface.esd.activated)
                    temp = input("Use Reflexes: >>> ")
                    if(temp != ""):
                        if(temp == "True"):
                            self.controller_node.wb_interface.esd.activated = True
                        elif(temp == "False"):
                            self.controller_node.wb_interface.esd.activated = False


                    
                
                elif(input_string == "goUp"):
                    print("Going Up")
                    if(self.walking):
                        print("Please stop the robot before going down")
                        continue
                    if(not self.isDown):
                        print("The robot is already up")
                        continue
                    
                    self.transition_active = True
                    # A centroidal MPC is not valid while the body is resting on
                    # the ground.  Keep its feed-forward torque disabled during
                    # stand-up and blend it in only after reaching the home pose.
                    self.support_scale = 0.0
                    self.support_ramp_start = None
                    self.support_fault = None
                    self.transition_abort = None
                    self.standup_gravity_scale = 0.0
                    initial_height = -cfg.simulation_params['ref_z']

                    temp = copy.deepcopy(self.controller_node.joint_positions)
                    initial_joint_positions = LegsAttr(*[np.zeros((1, int(self.controller_node.env.mjModel.nu/4))) for _ in range(4)])
                    initial_joint_positions.FL = temp[0:3]
                    initial_joint_positions.FR = temp[3:6]
                    initial_joint_positions.RL = temp[6:9]
                    initial_joint_positions.RR = temp[9:12]

                    reference_joint_positions = LegsAttr(*[np.zeros((1, int(self.controller_node.env.mjModel.nu/4))) for _ in range(4)])
                    keyframe_id = mujoco.mj_name2id(self.controller_node.env.mjModel, mujoco.mjtObj.mjOBJ_KEY, "home")
                    standUp_qpos = self.controller_node.env.mjModel.key_qpos[keyframe_id]
                    
                    reference_joint_positions.FL = standUp_qpos[7:10].copy()
                    reference_joint_positions.FR = standUp_qpos[10:13].copy()
                    reference_joint_positions.RL = standUp_qpos[13:16].copy()
                    reference_joint_positions.RR = standUp_qpos[16:19].copy()

                    # Do not pull the abduction joints from their naturally
                    # splayed, ground-contact pose to zero while the trunk is
                    # still resting on the floor.  That was the source of the
                    # 8 rad/s fr_hx/hr_hx events.  The later MPC blend closes
                    # the stance only after the trunk has been lifted.
                    for leg in ["FL", "FR", "RL", "RR"]:
                        reference_joint_positions[leg][0] = initial_joint_positions[leg][0]

                    crouch = np.asarray(
                        cfg.simulation_params.get(
                            'standup_crouch_pose', [0.0, 1.45, -2.65]
                        ),
                        dtype=float,
                    )
                    crouch_joint_positions = LegsAttr(*[
                        crouch.copy() for _ in range(4)
                    ])
                    for leg in ["FL", "FR", "RL", "RR"]:
                        crouch_joint_positions[leg][0] = initial_joint_positions[leg][0]

                    # Phase 1: fold all four legs into the same compact pose.
                    # This removes the asymmetric ragdoll configuration without
                    # trying to lift the trunk at the same time.
                    print("goUp phase 1/2: folding legs symmetrically")
                    start_time = time.time()
                    fold_duration = float(
                        cfg.simulation_params.get('standup_fold_duration', 3.0)
                    )
                    while (
                        time.time() - start_time < fold_duration
                        and self.transition_abort is None
                    ):
                        time_diff = time.time() - start_time
                        alpha = np.clip(time_diff / fold_duration, 0.0, 1.0)
                        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                        interpolated_positions = [
                            (1 - alpha) * initial + alpha * reference
                            for initial, reference in zip(
                                initial_joint_positions, crouch_joint_positions
                            )
                        ]

                        self.controller_node.stand_up_and_down_actions.FL = interpolated_positions[0]
                        self.controller_node.stand_up_and_down_actions.FR = interpolated_positions[1]
                        self.controller_node.stand_up_and_down_actions.RL = interpolated_positions[2]
                        self.controller_node.stand_up_and_down_actions.RR = interpolated_positions[3]

                        self.height_delta = initial_height
                        time.sleep(0.01)

                    if self.transition_abort is not None:
                        self.transition_active = False
                        self.support_scale = 0.0
                        self.standup_gravity_scale = 0.0
                        print(f"goUp stopped safely: {self.transition_abort}")
                        continue

                    self.controller_node.stand_up_and_down_actions = copy.deepcopy(
                        crouch_joint_positions
                    )

                    # Phase 2: extend from a symmetric crouch to home.  Only now
                    # raise the base-height reference (MPC torque remains zero).
                    print("goUp phase 2/2: lifting from crouch")
                    start_time = time.time()
                    time_motion = float(
                        cfg.simulation_params.get('standup_duration', 6.0)
                    )
                    while (
                        time.time() - start_time < time_motion
                        and self.transition_abort is None
                    ):
                        time_diff = time.time() - start_time
                        alpha = np.clip(time_diff / time_motion, 0.0, 1.0)
                        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                        interpolated_positions = [
                            (1 - alpha) * initial + alpha * reference
                            for initial, reference in zip(
                                crouch_joint_positions, reference_joint_positions
                            )
                        ]
                        self.controller_node.stand_up_and_down_actions.FL = interpolated_positions[0]
                        self.controller_node.stand_up_and_down_actions.FR = interpolated_positions[1]
                        self.controller_node.stand_up_and_down_actions.RL = interpolated_positions[2]
                        self.controller_node.stand_up_and_down_actions.RR = interpolated_positions[3]
                        self.height_delta = initial_height + cfg.simulation_params['ref_z'] * alpha
                        self.standup_gravity_scale = alpha
                        time.sleep(0.01)

                    if self.transition_abort is not None:
                        self.transition_active = False
                        self.support_scale = 0.0
                        self.standup_gravity_scale = 0.0
                        print(f"goUp stopped safely: {self.transition_abort}")
                        continue

                    # End exactly at the keyframe rather than at the last
                    # sampled value just below alpha=1.
                    self.controller_node.stand_up_and_down_actions = copy.deepcopy(
                        reference_joint_positions
                    )
                    self.height_delta = 0
                    self.standup_gravity_scale = 1.0

                    # The smooth trajectory has only just reached home.  Hold
                    # it with MPC still disabled so the loaded knees can catch
                    # up before evaluating the tracking-error gate.
                    settle_duration = float(
                        cfg.simulation_params.get('standup_home_settle_duration', 3.0)
                    )
                    print("Holding home pose to complete the lift")
                    settle_start = time.time()
                    while (
                        time.time() - settle_start < settle_duration
                        and self.transition_abort is None
                    ):
                        time.sleep(0.01)

                    if self.transition_abort is not None:
                        self.transition_active = False
                        self.support_scale = 0.0
                        self.standup_gravity_scale = 0.0
                        print(f"goUp stopped safely: {self.transition_abort}")
                        continue

                    self.isDown = False
                    self.support_ramp_start = time.monotonic()
                    print(
                        "Joint-space stand target accepted; gradually enabling "
                        "MPC support..."
                    )
                    while self.support_ramp_start is not None:
                        time.sleep(0.05)
                    self.transition_active = False
                    if self.support_fault is None:
                        print("goUp completed")
                    else:
                        print(f"goUp stopped safely: {self.support_fault}")


                elif(input_string == "goDown"):
                    print("Going Down")
                    if(self.walking):
                        print("Please stop the robot before going down")
                        continue
                    if(self.isDown):
                        print("The robot is already down")
                        continue

                    self.transition_active = True
                    self.support_ramp_start = None
                    self.standup_gravity_scale = 0.0
                    start_time = time.time()
                    time_motion = 5.
                    initial_height = 0
                    while(time.time() - start_time < time_motion):
                        time_diff = time.time() - start_time
                        self.height_delta = initial_height - cfg.simulation_params['ref_z']*time_diff/time_motion
                        time.sleep(0.01)

                    self.height_delta = -cfg.simulation_params['ref_z']
                    self.isDown = True
                    self.transition_active = False
                    self.support_scale = 0.0
                    self.standup_gravity_scale = 0.0
                
                
                elif(input_string == "help"):
                    self.print_all_commands()

                
                elif(input_string == "ictp"):
                    print("Interactive Keyboard Control")
                    print("w: Move Forward")
                    print("s: Move Backward")
                    print("a: Move Left")
                    print("d: Move Right")
                    print("q: Rotate Left")
                    print("e: Rotate Right")
                    print("0: Stop")
                    print("1: Pitch Up")
                    print("2: Reset Pitch")
                    print("3: Pitch Down")
                    print("Press any other key to exit")
                    while True:
                        command = readchar.readkey()
                        if(command == "w"):
                            velocity = self.controller_node.env._ref_base_lin_vel_H
                            velocity[0] = min(velocity[0] + 0.1, self.max_forward_speed)
                            print(f"vx = {velocity[0]:.2f} m/s")
                        elif(command == "s"):
                            velocity = self.controller_node.env._ref_base_lin_vel_H
                            velocity[0] = max(velocity[0] - 0.1, -self.max_forward_speed)
                            print(f"vx = {velocity[0]:.2f} m/s")
                        elif(command == "a"):
                            velocity = self.controller_node.env._ref_base_lin_vel_H
                            velocity[1] = min(velocity[1] + 0.1, self.max_lateral_speed)
                            print(f"vy = {velocity[1]:.2f} m/s")
                        elif(command == "d"):
                            velocity = self.controller_node.env._ref_base_lin_vel_H
                            velocity[1] = max(velocity[1] - 0.1, -self.max_lateral_speed)
                            print(f"vy = {velocity[1]:.2f} m/s")
                        elif(command == "q"):
                            self.controller_node.env._ref_base_ang_yaw_dot = min(
                                self.controller_node.env._ref_base_ang_yaw_dot + 0.1,
                                self.max_yaw_rate,
                            )
                            print(f"yaw rate = {self.controller_node.env._ref_base_ang_yaw_dot:.2f} rad/s")
                        elif(command == "e"):
                            self.controller_node.env._ref_base_ang_yaw_dot = max(
                                self.controller_node.env._ref_base_ang_yaw_dot - 0.1,
                                -self.max_yaw_rate,
                            )
                            print(f"yaw rate = {self.controller_node.env._ref_base_ang_yaw_dot:.2f} rad/s")
                        elif(command == "0"):
                            self.controller_node.env._ref_base_lin_vel_H[0] = 0
                            self.controller_node.env._ref_base_lin_vel_H[1] = 0
                            self.controller_node.env._ref_base_ang_yaw_dot = 0 
                            print("0")
                        elif(command == "1"):
                            self.pitch_delta -= 0.1
                            print("1")
                        elif(command == "2"):
                            self.pitch_delta = 0
                            print("2")
                        elif(command == "3"):
                            self.pitch_delta += 0.1
                            print("3")
                        else:
                            self.controller_node.env._ref_base_lin_vel_H[0] = 0
                            self.controller_node.env._ref_base_lin_vel_H[1] = 0
                            self.controller_node.env._ref_base_ang_yaw_dot = 0 
                            break
            except Exception as e:
                print("Error: ", e)
                print("Invalid Command")
                self.print_all_commands()


    def print_all_commands(self):
        print("\nAvailable Commands")
        print("help: Display all available messages")
        print("stw: Start Walking")
        print("ooo: Stop Walking")
        print("ictp: Interactive Keyboard Control")
        print("########################")
        print("narrowstance: Narrow Stance")
        print("widestance: Wide Stance")
        print("goUp: The robot goes up")
        print("goDown: The robot goes down")
        print("########################")
        print("setGaitTimer: Set the gait type")
        print("setupGaitTimer: Setup the gait timer")
        print("setupLegsGains: Setup the leg gains")
        print("setupGeneral: Setup general parameters\n")
