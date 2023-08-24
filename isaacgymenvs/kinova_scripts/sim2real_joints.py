import threading
import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil

from scipy.interpolate import interp1d

from kinova import KinovaArm, MoveJController, DeviceConnection

# parse arguments
args = gymutil.parse_arguments(description="Joint control Methods Example")

def setup_sim():
   # initialize gym
    gym = gymapi.acquire_gym()

    # create a simulator
    sim_params = gymapi.SimParams()
    sim_params.substeps = 5
    sim_params.dt = 1.0 / 60.0

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
    gym.add_ground(sim, gymapi.PlaneParams())

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
    initial_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    hand = gym.create_actor(env, hand_asset, initial_pose, 'arm_hand', 0, 1)

    return hand, hand_asset

def set_actor_properties(hand):
    # Configure DOF properties
    props = gym.get_actor_dof_properties(env, hand)
    props["driveMode"].fill(gymapi.DOF_MODE_POS)
    props["lower"][0] = -3.14
    props["upper"][0] = 3.14

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

gym, sim, viewer, env = setup_sim()

hand_handle, hand_asset = load_hand()

arm_control = [0] * 7
joint_positions = arm_control
set_hand_joint_positions(joint_positions, hand_handle)

gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_X, "x")

def remap(value, low1=-3.14, high1=3.14, low2=-180, high2=180):
    return low2 + (value - low1) * (high2 - low2) / (high1 - low1)

ranges = np.array([[-3.14, 3.14, -180, 180],
                   [-2.41, 2.41, -128, 128],
                   [-3.14, 3.14, -180, 180],
                   [-2.66, 2.66, -147, 147],
                   [-3.14, 3.14, -180, 180],
                   [-2.23, 2.23, -120, 120],
                   [-3.14, 3.14, -180, 180]])


arm = KinovaArm()
# Simulate
while not gym.query_viewer_has_closed(viewer):
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "x" and evt.value > 0:
            j = int(input('give joint: '))
            v = float(input('give angle: '))
            
            arm_control[j] = v

            joint_positions = arm_control
            set_hand_joint_target_positions(joint_positions, hand_handle)

            x = remap(v, ranges[j][0], ranges[j][1], ranges[j][2], ranges[j][3])
            x = x % 360

            print (x)

            # try:
            angles = arm.get_joint_angles()
            print (angles)
            angles[j] = x
            # th_arm = threading.Thread(target=arm.move_angular, args=(angles,))
            # th_arm.start()
            # th_arm.join()
            print (arm.get_tool_pose())
            # arm.move_angular(angles)
            # finally:
            #     arm.disconnect()

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

arm.disconnect()
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

## Actuator 1
# def remap(value, low1=-3.14, high1=3.14, low2=-180, high2=180):
#     return low2 + (value - low1) * (high2 - low2) / (high1 - low1)

## Actuator 2
# def remap(value, low1=-2.41, high1=2.41, low2=-128, high2=128):
#     return low2 + (value - low1) * (high2 - low2) / (high1 - low1)

## Actuator 3
# def remap(value, low1=-3.14, high1=3.14, low2=-180, high2=180):
#     return low2 + (value - low1) * (high2 - low2) / (high1 - low1)

## Actuator 4
# def remap(value, low1=-2.66, high1=2.66, low2=-147, high2=147):
#     return low2 + (value - low1) * (high2 - low2) / (high1 - low1)

## Actuator 5
# def remap(value, low1=-3.14, high1=3.14, low2=-180, high2=180):
#     return low2 + (value - low1) * (high2 - low2) / (high1 - low1)

## Actuator 6
# def remap(value, low1=-2.23, high1=2.23, low2=-120, high2=120):
#     return low2 + (value - low1) * (high2 - low2) / (high1 - low1)

## Actuator 7
# def remap(value, low1=-3.14, high1=3.14, low2=-180, high2=180):
#     return low2 + (value - low1) * (high2 - low2) / (high1 - low1)
