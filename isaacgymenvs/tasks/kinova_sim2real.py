import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.tasks.grasp_sampler import load_model, GraspModel

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import threading
import time

from isaacgymenvs.tasks.kinova import KinovaArm, MoveJController, DeviceConnection

class KinovaTest(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.robot_position_noise = self.cfg["env"]["robotPositionNoise"]
        self.robot_rotation_noise = self.cfg["env"]["robotRotationNoise"]
        self.robot_dof_noise = self.cfg["env"]["robotDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # Create dicts to pass to reward function
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "r_rot_scale": self.cfg["env"]["rotationRewardScale"],
            "r_fintip_scale": self.cfg["env"]["fingertipRewardScale"],
            "r_fintip_dist_scale": self.cfg["env"]["fingertipDistanceScale"],
            "r_lift_scale": self.cfg["env"]["liftRewardScale"],
            "r_lift_height_scale": self.cfg["env"]["liftHeightRewardScale"],
            "r_actions_reg_scale": self.cfg["env"]["actionsRegularizationRewardScale"],
        }

        # Arm controller type
        self.arm_control_type = self.cfg["env"]["armControlType"]
        assert self.arm_control_type in {"osc", "pos"},\
            "Invalid control type specified. Must be one of: {osc, pos}"

        # Hand controller type
        # self.hand_control_type = self.cfg["env"]["handControlType"]
        # assert self.hand_control_type in {"binary", "synergy", "pos"},\
        #     "Invalid control type specified. Must be one of: {binary, synergy, pos}"

        # dimensions
        # obs include: object_pose (7) + eef_pose (7) + relative_object_eef_pos (3) 
        # if hand_pos control: + hand_q (17)
        # if hand_synergy control: + hand_q (2)
        # num_obs = 17
        num_obs = 13
        # if self.hand_control_type == "pos": num_obs += 17
        # elif self.hand_control_type == "synergy": num_obs += 2
        self.cfg["env"]["numObservations"] = num_obs

        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        # if arm_osc control: delta eef (6)
        # if arm_pos control: arm joint angles (7)
        # if hand_binary control: bool gripper (1)
        # if hand_synergy control: latent space (2)
        # if hand_pos control: finger joint angles (17)
        num_actions = 0
        if self.arm_control_type == "osc": num_actions += 6
        elif self.arm_control_type == "pos": num_actions += 7

        # if self.hand_control_type == "binary": num_actions += 1
        # elif self.hand_control_type == "synergy": num_actions += 2
        # else: num_actions += 17

        self.cfg["env"]["numActions"] = num_actions

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed

        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None                     # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None     # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None            # Position actions
        self._effort_control = None         # Torque actions
        self._robot_effort_limits = None        # Actuator effort limits for robot
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array

        self.up_axis = "z"

        super().__init__(config=self.cfg, rl_device=rl_device,
                sim_device=sim_device, graphics_device_id=graphics_device_id,
                headless=headless,
                virtual_screen_capture=virtual_screen_capture,
                force_render=force_render)

        # Kinova + Seed defaults
        self.robot_default_dof_pos_list = [0.19, 0.42, -1.39, 1.04, -1.00, 0.55, 0.00] #+ [0.0] * 17
        # robot_default_dof_pos = [-0.5, 1.2, 0.09, 1.14, -1.7, -1.5, -0.6] + [0.0] * 17
        # robot_default_dof_pos[20] = 1.57
        self.robot_default_dof_pos = to_torch(self.robot_default_dof_pos_list, device=self.device)

        # OSC Gains
        self.kp = to_torch([200.] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.] * 7, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)

        # Set control limits
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
        self.arm_control_type == "osc" else self._robot_effort_limits[:7].unsqueeze(0)

        # if self.hand_control_type == "synergy":
        #     self.synergy_model = load_model('tasks/grasp_sampler', GraspModel)

        self.axes_geom = gymutil.AxesGeometry(0.2)
        
        # print ('\n\n\n\n')
        self.real_arm = KinovaArm()
        # print ('\n\n\n\n')

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()
                
        if self.viewer != None:
            cam_pos = gymapi.Vec3(1.0, 1.0, 1.0)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        robot_asset_file = "urdf/seed_kinova_robot/kinova.urdf"

        # load robot asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        # asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        asset_options.override_com = True
        asset_options.override_inertia = True
        robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, asset_options)

        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(robot_asset)

        print("num robot bodies: ", self.num_robot_bodies)
        print("num robot dofs: ", self.num_robot_dofs)

        # set robot dof properties
        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        self.robot_dof_lower_limits = []
        self.robot_dof_upper_limits = []
        self._robot_effort_limits = []

        for i in range(self.num_robot_dofs):
            self.robot_dof_lower_limits.append(robot_dof_props['lower'][i])
            self.robot_dof_upper_limits.append(robot_dof_props['upper'][i])
            self._robot_effort_limits.append(robot_dof_props['effort'][i])

        # arm properties
        if self.arm_control_type == 'osc':
            robot_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
            robot_dof_props["stiffness"][:7].fill(5.0)
            robot_dof_props["damping"][:7].fill(1.0)
        else:
            robot_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
            robot_dof_props["stiffness"][:7].fill(150.0)
            robot_dof_props["damping"][:7].fill(100.0)
        # robot_dof_props["armature"][:7].fill(0.001)
        # hand properties
        # robot_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        # robot_dof_props["armature"][7:].fill(0.001)
        # robot_dof_props["stiffness"][7:].fill(100.0)
        # robot_dof_props["damping"][7:].fill(50.0)

        self.robot_dof_lower_limits = to_torch(self.robot_dof_lower_limits, device=self.device)
        self.robot_dof_upper_limits = to_torch(self.robot_dof_upper_limits, device=self.device)
        self.robot_dof_lower_limits[[0, 2, 4, 6]] = -3.14
        self.robot_dof_upper_limits[[0, 2, 4, 6]] = 3.14
        self._robot_effort_limits = to_torch(self._robot_effort_limits, device=self.device)

        print ("DoF lower limits: ", self.robot_dof_lower_limits)
        print ("DoF upper limits: ", self.robot_dof_upper_limits)

        # Define start pose for robot
        robot_start_pose = gymapi.Transform()
        robot_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        robot_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        num_robot_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)
        max_agg_bodies = num_robot_bodies #+ num_table_bodies + num_object_bodies
        max_agg_shapes = num_robot_shapes #+ num_table_shapes + num_object_shapes

        self.robots = []
        self.envs = []

        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create robot actor and set properties
            robot_actor = self.gym.create_actor(env_ptr, robot_asset, robot_start_pose, "robot", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, robot_actor, robot_dof_props)

            self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.robots.append(robot_actor)
        
        self.timer = 0
        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        robot_handle = 0

        self.handles = {
            "eef": self.gym.find_actor_rigid_body_handle(env_ptr, robot_handle, "eef"),
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)

        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.handles["eef"], :]

        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "robot")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, robot_handle)['EndEffector']
        self._j_eef = jacobian[:, hand_joint_index, :, :7]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "robot")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :7, :7]

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * 1, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

        self.down_axis = to_torch([0, 0, -1], device=self.device).repeat((self.num_envs, 1))
        self.grasp_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))

        # self.target_pose = random_pos(self.num_envs, self.device)
        pose = torch.tensor([0.4, 0.4, 0.3]).to(self.device)
        # pose = torch.tensor([0.58, 0.01, 0.434]).to(self.device)
        self.target_pose = pose.repeat(self.num_envs, 1)

    def _update_states(self):
        self.states.update({
            "q_arm": self._q[:, :7],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "target": self.target_pose,
            "dist": self._eef_state[:, :3] - self.target_pose,
        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], reward_dict = compute_robot_reward(
                self.reset_buf, self.progress_buf, self.actions, self.states,
                self.grasp_up_axis, self.down_axis,
                self.num_envs,
                self.reward_settings, self.max_episode_length)

    def compute_observations(self):
        self._refresh()
        obs = ["eef_pos", "eef_quat", "target", "dist"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        maxs = {ob: torch.max(self.states[ob]).item() for ob in obs}

        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.timer = 0

        # self.real_arm.zero(blocking=True)
        # self.real_arm.hom(blocking=True)
        default_angles = sim2real_joints(self.robot_default_dof_pos)
        self.real_arm.move_angular(default_angles)
        
        pos = tensor_clamp(self.robot_default_dof_pos.unsqueeze(0),
            self.robot_dof_lower_limits.unsqueeze(0), self.robot_dof_upper_limits)

        # # Reset the internal obs accordingly
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)
        self._dof_state[env_ids, :, 0] = pos

        # self.target_pose[env_ids] = random_pos(env_ids.shape[0], self.device)

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _compute_osc_torques(self, dpose):
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[:, :7], self._qd[:, :7]
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
                self.kp * dpose - self.kd * self.states["eef_vel"]).unsqueeze(-1)

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        # j_eef_inv = m_eef @ self._j_eef @ mm_inv
        # u_null = self.kd_null * -qd + self.kp_null * (
        #         (self.robot_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi)
        # u_null[:, 7:] *= 0
        # u_null = self._mm @ u_null.unsqueeze(-1)
        # u += (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(u.squeeze(-1),
                         -self._robot_effort_limits[:7].unsqueeze(0), self._robot_effort_limits[:7].unsqueeze(0))

        return u

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        

        # Split arm and gripper command
        if self.arm_control_type == "osc":
            # u_arm, u_hand = self.actions[:, :6], self.actions[:, 6:]
            u_arm = self.actions[:, :6]
            u_arm = u_arm * self.cmd_limit / self.action_scale
            #print (u_arm)
            u_arm = self._compute_osc_torques(dpose=u_arm)

            self._effort_control = u_arm 
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))
        else:
            u_arm = self.actions[:, :7]
            u_arm = unscale_transform(u_arm,
                                      self.robot_dof_lower_limits[:7],
                                      self.robot_dof_upper_limits[:7])
        
            u_arm = tensor_clamp(u_arm,
                                 self.robot_dof_lower_limits[:7],
                                 self.robot_dof_upper_limits[:7])

            #real_angles = sim2real_joints(u_arm[0]).cpu()
            #real_angles = real_angles.tolist()
            
            #if self.timer % 15 == 0 and self.timer > 1:
            if self.timer > 1:
                print ()
                old_pose = self.real_arm.get_tool_pose()
                
                old_pos = torch.tensor([[old_pose[0][0], 
                                         old_pose[0][1],
                                         old_pose[0][2]]])
                
                print (f"Old pos: {old_pos[0]}")
                print (f"New pos: {self.new_pos[0]}")

                vel = torch.zeros((6,))
                vel[0] = self.new_pos[0][1] - old_pos[0][1]
                vel[1] = self.new_pos[0][2] - old_pos[0][2]
                vel[2] = self.new_pos[0][0] - old_pos[0][0]

                old_rot = torch.tensor([old_pose[1][0], 
                                         old_pose[1][1],
                                         old_pose[1][2]])
        
                new_rot_zyx = gymapi.Quat(self.new_rot[0][0],
                                          self.new_rot[0][1], 
                                          self.new_rot[0][2],
                                          self.new_rot[0][3]).to_euler_zyx()

                new_rot_zyx = torch.tensor(new_rot_zyx) * (180. / 3.1415927410125732)

                new_rot_xyz = new_rot_zyx.flip(0)
                print (f"Old rot: {old_rot}")
                print (f"New rot: {new_rot_xyz}")                
                
                vel[3] = new_rot_xyz[0] - old_rot[0]
                vel[4] = new_rot_xyz[2] - old_rot[2]
                vel[5] = new_rot_xyz[1] - old_rot[1]
                
                print(f"Vel: {vel}")

                #self.real_arm.move_cartesian_vel(vel, blocking=False)
                print (self.states['q_arm   '])
                #if self.timer > 15:
                #    vel = [0, 0, 0]
                #    self.real_arm.move_cartesian_vel(vel, blocking=False)
                #    exit()
                #self.real_arm.move_angular(real_angles, blocking=True)
                #print (u_arm[0].cpu().tolist())
                #print (real_angles)
                #self.real_arm.move_angular(real_angles, blocking=False)
            
            #time.sleep(0.1)

            self._pos_control = u_arm 
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
            self.timer += 1

    def post_physics_step(self):
        self.progress_buf += 1
     
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()

        self.new_pos = self.states["eef_pos"].clone()
        self.new_rot = self.states["eef_quat"].clone()
        
        self.compute_reward(self.actions)
  
        if self.viewer:
            self.gym.clear_lines(self.viewer)
            for i in range(self.num_envs):
                env = self.envs[i]
                pose = self.target_pose[i]
                tpose = get_transform(pose)
                gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, env, tpose)
        # print ('POST')

def remap(value, low1=-3.14, high1=3.14, low2=-180, high2=180):
    return low2 + (value - low1) * (high2 - low2) / (high1 - low1)

def sim2real_joints(sim_angles):
    ranges = np.array([[-3.14, 3.14, -180, 180],
                       [-2.41, 2.41, -128, 128],
                       [-3.14, 3.14, -180, 180],
                       [-2.66, 2.66, -147, 147],
                       [-3.14, 3.14, -180, 180],
                       [-2.23, 2.23, -120, 120],
                       [-3.14, 3.14, -180, 180]])

    real_angles = torch.zeros_like(sim_angles)
    for i in range(real_angles.shape[0]):
        x = remap(sim_angles[i], ranges[i][0], ranges[i][1], ranges[i][2], ranges[i][3])
        if i in [0, 2, 4, 6]: x = x % 360
        real_angles[i] = x

    return real_angles

def get_transform(pose):
    t = gymapi.Transform()
    t.p.x = pose[0]
    t.p.y = pose[1]
    t.p.z = pose[2]
    return t
#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def compute_robot_reward(
    reset_buf, progress_buf, actions, states, grasp_up_axis, down_axis,
    num_envs, reward_settings, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Tensor, Tensor, int, Dict[str, float], float) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]

    # distance from grasp link to the object
    d_eef = torch.norm(states["dist"], dim=-1)
    dist_reward = 1 - torch.tanh(d_eef)
    # dist_reward = 1 - torch.tanh(2 * d_eef)

    # print (states["dist"])
    # print (dist_reward)
    # print (dist_reward.shape)
    # distance from fingertips to the object
    # d_fftip = torch.norm(states["fftip_pos"] - states["object_pos"], dim=-1)
    # d_thtip = torch.norm(states["thtip_pos"] - states["object_pos"], dim=-1)
    # fintip_reward = 1 - torch.tanh(reward_settings["r_fintip_dist_scale"] * (d_fftip + d_thtip) / 2.)

    ## grasp axis should look down
    grasp_axis = tf_vector(states["eef_quat"], grasp_up_axis)
    dot = torch.bmm(grasp_axis.view(num_envs, 1, 3), down_axis.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
    rot_reward = torch.sign(dot) * dot ** 2

    # # reward for lifting object
    # object_height = states["object_pos"][:, 2]
    # object_lifted = object_height > 0.45
    # lift_reward = object_lifted
    # lift_height = object_height - 0.3

    # # Regularization on the actions
    # action_penalty = torch.sum(actions ** 2, dim=-1)

    rewards = reward_settings["r_dist_scale"] * dist_reward \
            + reward_settings["r_rot_scale"] * rot_reward 
            # + reward_settings["r_fintip_scale"] * fintip_reward \
            # + reward_settings["r_lift_scale"] * lift_reward \
            # + reward_settings["r_lift_height_scale"] * lift_height \
            # + reward_settings["r_actions_reg_scale"] * action_penalty

    # Compute resets
    # reset_buf = torch.where((progress_buf >= max_episode_length - 1) | (lift_reward > 0), torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)

    # Return rewards in dict for debugging
    reward_dict = {"Distance Reward": reward_settings["r_dist_scale"] * dist_reward,
                   "Rotation Reward": reward_settings["r_rot_scale"] * rot_reward}
                   # "Fingertips Reward": reward_settings["r_fintip_scale"] * fintip_reward,
                   # "Lift Reward": reward_settings["r_lift_scale"] * lift_reward,
                   # "Lift Height Reward": reward_settings["r_lift_height_scale"] * lift_reward,
                   # "Action Regularization Reward": reward_settings["r_actions_reg_scale"] * action_penalty,}

    return rewards, reset_buf, reward_dict

@torch.jit.script
def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

@torch.jit.script
def random_pos(num: int, device: str) -> torch.Tensor:
    radius = 0.6
    height = 0.3
    # sample theta of point
    theta = 2 * np.pi * torch.rand(num, dtype=torch.float, device=device)

    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    z = torch.tensor([height], device=device).repeat((num, 1))

    return torch.cat((x[:, None], y[:, None], z), dim=-1)

@torch.jit.script
def remap(x: torch.Tensor, l1: float, h1: float, l2: float, h2: float) -> torch.Tensor:
    return l2 + (x - l1) * (h2 - l2) / (h1 - l1)
    
@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat

@torch.jit.script
def to_rads(x):
    return (x * 3.14159265359) / 180.
