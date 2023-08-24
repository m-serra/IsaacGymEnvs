import threading
import math
import time
import numpy as np
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil

from scipy.interpolate import interp1d

from kinova import KinovaArm, MoveJController, DeviceConnection

import torch

# parse arguments
args = gymutil.parse_arguments(description="Joint control Methods Example")

def setup_sim():
   # initialize gym
    gym = gymapi.acquire_gym()

    # create a simulator
    sim_params = gymapi.SimParams()
    sim_params.substeps = 5
    sim_params.dt = 1.0 / 60.0
    sim_params.up_axis = gymapi.UP_AXIS_Z

    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 5
    sim_params.physx.bounce_threshold_velocity = 0.28
    sim_params.physx.contact_offset = 0.035
    sim_params.physx.rest_offset = 0.00001
    sim_params.physx.friction_offset_threshold = 0.01
    sim_params.physx.friction_correlation_distance = 0.05
    sim_params.physx.max_depenetration_velocity = 1000.0

    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

    sim_params.use_gpu_pipeline = False
    if args.use_gpu_pipeline:
        print("WARNING: Forcing CPU pipeline.")

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

    if sim is None:
        print("*** Failed to create sim")
        quit()

    # create viewer using the default camera properties
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise ValueError('*** Failed to create viewer')

    # add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    # set up the env grid
    num_envs = 1
    spacing = 1.5
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, 0.0, spacing)
    env = gym.create_env(sim, env_lower, env_upper, 2)

    # Look at the first env
    cam_pos = gymapi.Vec3(8, 4, 1.5)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.2)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    return gym, sim, viewer, env

def load_hand():
    # add hand urdf asset
    asset_root = "urdf"
    asset_file = "kinova.urdf"

    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.001
    asset_options.fix_base_link = True
    asset_options.thickness = 0.002

    asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

    # Load asset with default control type of position for all joints
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    asset_options.use_mesh_materials = True
    asset_options.override_com = True
    asset_options.override_inertia = True
    asset_options.armature = 0.0001
    print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
    hand_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    # initial root pose for hand actors
    initial_pose = gymapi.Transform()
    initial_pose.p = gymapi.Vec3(0., 0., 0.)
    initial_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    hand = gym.create_actor(env, hand_asset, initial_pose, 'arm_hand', 0, 1)

    return hand, hand_asset

def set_actor_properties(hand):
    # Configure DOF properties
    props = gym.get_actor_dof_properties(env, hand)
    props["driveMode"].fill(gymapi.DOF_MODE_POS)
    props["lower"][0] = -math.pi
    props["upper"][0] = math.pi

    for i in range(7):
        print (props['lower'][i], props['upper'][i])

    gym.set_actor_dof_properties(env, hand, props)
    return len(props)

def set_hand_joint_positions(joint_positions, hand_handle):
    hand_dof_states = gym.get_actor_dof_states(env, hand_handle, gymapi.STATE_POS)

    for j in range(len(hand_dof_states['pos'])):
        hand_dof_states['pos'][j] = joint_positions[j]

    gym.set_actor_dof_states(env, hand_handle, hand_dof_states, gymapi.STATE_POS)
    gym.set_actor_dof_position_targets(env, hand_handle, joint_positions)

def set_hand_joint_target_positions(joint_positions, hand_handle):
    gym.set_actor_dof_position_targets(env, hand_handle, joint_positions)

def remap(value, low1=-math.pi, high1=math.pi, low2=-180, high2=180):
    return low2 + (value - low1) * (high2 - low2) / (high1 - low1)

def center_angles_at_360(x):
    return (x + 360) % 360

def sim2real_joints(sim_angles):
    ranges = np.array([[-math.pi, math.pi, -180, 180],
                       [-2.41, 2.41, -128.93, 128.93],
                       [-math.pi, math.pi, -180, 180],
                       [-2.66, 2.66, -147.83, 147.83],
                       [-math.pi, math.pi, -180, 180],
                       [-2.23, 2.23, -120.32, 120.32],
                       [-math.pi, math.pi, -180, 180]])

    real_angles = torch.zeros_like(sim_angles)
    for i in range(real_angles.shape[0]):
        x = remap(sim_angles[i], ranges[i][0], ranges[i][1], ranges[i][2], ranges[i][3])
        if i in [0, 2, 4, 6]: x = x % 360
        x = center_angles_at_360(x)
        real_angles[i] = x

    return real_angles

def center_angles_at_0(x, angle_range):
    new_x = (x + angle_range) % 360 - angle_range
    assert abs(new_x) <= angle_range, "x is outside the angle range"
    return new_x

def real2sim_joints(real_angles):
    ranges = np.array([[-math.pi, math.pi, -180, 180],
                       [-2.41, 2.41, -128.93, 128.93],
                       [-math.pi, math.pi, -180, 180],
                       [-2.66, 2.66, -147.83, 147.83],
                       [-math.pi, math.pi, -180, 180],
                       [-2.23, 2.23, -120.32, 120.32],
                       [-math.pi, math.pi, -180, 180]])
    
    sim_angles = torch.zeros_like(real_angles)
    for i in range(sim_angles.shape[0]):
        real_angles[i] = center_angles_at_0(real_angles[i], angle_range=ranges[i][3])
        x = remap(real_angles[i], ranges[i][2], ranges[i][3], ranges[i][0], ranges[i][1])
        sim_angles[i] = x
    return sim_angles

def shift_angle(angle, shift):
    if angle <= shift:
        return angle + shift
    elif angle >= (360-shift):
        return (angle - (360-shift)) # % shift
    print("else")
    exit(0)


def get_direction_and_refine_target_bounded(current_angle, target_angle, angle_width):
    # 0: [-math.pi, math.pi, -180, 180],
    # 1: [-2.41, 2.41, -128.93, 128.93],
    # 2: [-math.pi, math.pi, -180, 180],
    # 3: [-2.66, 2.66, -147.83, 147.83],
    # 4: [-math.pi, math.pi, -180, 180],
    # 5: [-2.23, 2.23, -120.32, 120.32],
    # 6: [-math.pi, math.pi, -180, 180]]

    """
    
    """
    if target_angle == 0:
        if current_angle >= (360 - angle_width):
            direction = 1
            target_angle = 360 
        else:
            direction = -1
    elif target_angle == 360:
        if current_angle <= angle_width:
            direction = -1
            target_angle = 0
        else:
            direction = 1
    else:
        print("Current_angle:", current_angle)
        current_shifted = shift_angle(current_angle, angle_width)
        print("Current shifted:", current_shifted)
        target_shifted = shift_angle(target_angle, angle_width)
        print("Target shifted:", target_shifted)
        diff = current_shifted - target_shifted
        direction = -1 if diff >= 0 else 1
        
    return direction, target_angle

def get_direction_and_refine_target_360(current_angle, target_angle):
    """
    Determines the rotation direction that leads to a shorter movement to the target.
    If the target angle is in the transition, we refine it according to the
    direction, to avoid jumps from 0 to 360 and vice-versa
    """
    if target_angle == 0:
        if current_angle == 0 or current_angle == 360:
            direction = 0
        elif current_angle <= 180:
            direction = -1
        else:
            direction = 1
            target_angle = 360 
    elif target_angle == 360:
        if current_angle == 0 or current_angle == 360:
            direction = 0
        if current_angle >= 180:
            direction = 1
        else:
            direction = -1
            target_angle = 0
    else:
        diff = current_angle - target_angle
        if diff < -180:
            direction = -1
        elif diff < 0:
            direction = 1
        elif diff > 180:
            direction = 1
        else: # diff in [0, 180]
            direction = -1
    return direction, target_angle

gym, sim, viewer, env = setup_sim()

hand_handle, hand_asset = load_hand()

real_arm = KinovaArm()

# =====
# Snippet to get new target positions from the real arm (comment after using)
# real_angles = real_arm.get_joint_angles()
# print("Real angles:", real_angles)
# target_angles = real2sim_joints(torch.tensor(real_angles))
# print("Target angles:", target_angles)
# exit()
# =====

# Target joint angles: fixed simulated arm pose
arm_control = [0.] * 7
arm_control = [0.0, 0.247, math.pi, -2.263, 0., 0.967, 1.57]
arm_control = [-0.6739,  0.7431, -0.0231,  1.2184, -0.4662,  1.1644,  0.4668]
arm_control = [0.0,  0.28043, -math.pi, -2.3393, 0.0, 1.0193,  1.57]  # home position
arm_control = [0.0,  0.28043, -math.pi, -2.3393, 0.0, 1.0193,  0.0]
joint_positions = arm_control
set_hand_joint_positions(joint_positions, hand_handle)

gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_X, "x")


handles = {"eef": gym.find_actor_rigid_body_handle(env, hand_handle, "eef")}
_rigid_body_state_tensor = gym.acquire_rigid_body_state_tensor(sim)
_rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(1, -1, 13)
_eef_pos = _rigid_body_state[:, handles["eef"], :3]
_eef_rot = _rigid_body_state[:, handles["eef"], 3:7]

execute = False


joint_positions = torch.tensor(real_arm.get_joint_angles())

while not gym.query_viewer_has_closed(viewer):
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "x" and evt.value > 0:
            execute = not execute
            # execute = True
    
    if execute:
        
        joint_positions = joint_positions.cpu().numpy()
        joint_idx = 1
        joint_positions[joint_idx] = 270.0
        
        initial_diffs = abs(joint_positions - real_arm.get_joint_angles())
        max_speed = 40  # deg/sec
        stop_condition = .02  # difference in deg between target & real
        factor = np.zeros(real_arm.num_joints)
        angle_width = [360, 128.93, 360, 147.83, 360, 120.32, 360]

        # TODO: 
        # - choose direction that leads to the shortest movement.
        # - Fix factor computation
        # - Assert target angles have been converted to correct range
        # - Write get_closes_direction function for joints that are not 360

        # While any of the joint angle errors is above the stop condition
        while np.sum(abs(joint_positions - real_arm.get_joint_angles()) > stop_condition) != 0:
            
            # Angles in [0:360] with transition at 9 o'clock
            real_joint_angles = real_arm.get_joint_angles()  # repeated call...
            print("Real angle:", real_joint_angles[joint_idx])
            
            current_diffs = abs(joint_positions - real_joint_angles)
            print("Current_difs:", current_diffs[joint_idx])
            
            speeds = np.zeros(7)

            # Make smoother function for the factor
            for i in range(real_arm.num_joints):

                if i != joint_idx:  # safety guard to move only one joint during development
                    continue
                target_angle = center_angles_at_360(joint_positions[i])
                
                if angle_width[i] == 360:
                    direction, joint_positions[i] = get_direction_and_refine_target_360(real_joint_angles[i], target_angle)
                else:
                    direction, joint_positions[i] = get_direction_and_refine_target_bounded(real_joint_angles[i], target_angle, angle_width[i])

                if current_diffs[i] > max_speed:
                    speeds[i] = direction * max_speed
                else:
                    speeds[i] = direction * current_diffs[i]

            print("Speed", speeds[joint_idx], "\n")
            real_arm.send_joint_speeds(speeds)

        break
    

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print('Done')

real_arm.disconnect()
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
