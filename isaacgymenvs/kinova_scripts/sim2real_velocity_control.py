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
        real_angles[i] = x

    return real_angles

gym, sim, viewer, env = setup_sim()

hand_handle, hand_asset = load_hand()

real_arm = KinovaArm()

# Target joint angles: fixed simulated arm pose
arm_control = [0.] * 7
arm_control = [0.0, 0.247, math.pi, -2.263, 0., 0.967, 1.57]
arm_control = [-0.6739,  0.7431, -0.0231,  1.2184, -0.4662,  1.1644,  0.4668]
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

#tensor(41.1045) tensor(3.1049) tensor(171.2946)   
print (gymapi.Quat(-0.108886, 0.075947, 0.674842, -0.725923).to_euler_zyx())
                                                  
#tensor(-130.0659) tensor(163.2148) tensor(5.5699) 
print (gymapi.Quat(0.739716, -0.322126, -0.149240, 0.571649).to_euler_zyx())
                                                
# Control the arm to the position set in the simulator
new_pos = _eef_pos[0]
new_rot = _eef_rot[0]
while not gym.query_viewer_has_closed(viewer):
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "x" and evt.value > 0:
            execute = not execute
            # execute = True
    
    if execute:
        
        old_pose = real_arm.get_tool_pose()
        
        # Cartesian Vel
        old_pos = torch.tensor([old_pose[0][0], 
                                old_pose[0][1],
                                old_pose[0][2]])

        vel = torch.zeros((6,))
        vel[0] = 0 # new_pos[1] - old_pos[1]
        vel[1] = 0 # new_pos[2] - old_pos[2]
        vel[2] = 0 # new_pos[0] - old_pos[0]

        # Angular Vel
        old_rot = torch.tensor([old_pose[1][0], 
                                old_pose[1][1],
                                old_pose[1][2]])

        # print(gymapi.Quat().from_euler_zyx(old_rot[0], old_rot[1], old_rot[2]))

        new_rot_zyx = gymapi.Quat(new_rot[0],
                                  new_rot[1], 
                                  new_rot[2],
                                  new_rot[3]).to_euler_zyx()    
        new_rot_zyx = torch.tensor(new_rot_zyx) * (180. / math.pi)
        new_rot_xyz = new_rot_zyx.flip(0)
      
        vel[3] = 0 # new_rot_xyz[0] - old_rot[0]
        vel[4] = 0 # new_rot_xyz[2] - old_rot[2]
        vel[5] = new_rot_xyz[1] - old_rot[1]
        
        # print(f"Vel : {vel}")
        real_arm.move_cartesian_vel(vel, blocking=False)
        
    gym.refresh_rigid_body_state_tensor(sim)

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
