# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class DefaultQuadcopterStrategy:
    """Default strategy implementation for quadcopter environment."""

    def __init__(self, env: QuadcopterEnv):
        """Initialize the default strategy.

        Args:
            env: The quadcopter environment instance.
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        # Initialize episode sums for logging if in training mode
        if self.cfg.is_train and hasattr(env, 'rew'):
            keys = [key.split("_reward_scale")[0] for key in env.rew.keys() if key != "death_cost"]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in keys
            }

        # Initialize fixed parameters once (no domain randomization)
        # These parameters remain constant throughout the simulation
        # Aerodynamic drag coefficients
        self.env._K_aero[:, :2] = self.env._k_aero_xy_value
        self.env._K_aero[:, 2] = self.env._k_aero_z_value

        # PID controller gains for angular rate control
        # Roll and pitch use the same gains
        self.env._kp_omega[:, :2] = self.env._kp_omega_rp_value
        self.env._ki_omega[:, :2] = self.env._ki_omega_rp_value
        self.env._kd_omega[:, :2] = self.env._kd_omega_rp_value

        # Yaw has different gains
        self.env._kp_omega[:, 2] = self.env._kp_omega_y_value
        self.env._ki_omega[:, 2] = self.env._ki_omega_y_value
        self.env._kd_omega[:, 2] = self.env._kd_omega_y_value

        # Motor time constants (same for all 4 motors)
        self.env._tau_m[:] = self.env._tau_m_value

        # Thrust to weight ratio
        self.env._thrust_to_weight[:] = self.env._twr_value

    def get_rewards(self) -> torch.Tensor:
        """get_rewards() is called per timestep. Reward structure for racing the Powerloop track:
        directional gate-pass detection through the actual opening (not just proximity to the gate),
        distance-based progress shaping toward the current gate, and a contact-sensor-driven crash
        penalty. See EXPERIMENT_LOG.md for the design reasoning behind each piece."""

        # Gate-pass detection: sign change of the drone's position along the *current target gate's*
        # own local forward axis (_pose_drone_wrt_gate[:, 0] -- this is in the same rotation
        # convention as _normal_vectors, since both come from _waypoints_quat), combined with
        # lateral/vertical bounds so it counts as passing through the opening, not just crossing the
        # infinite gate plane. _prev_x_drone_wrt_gate starts at +1.0 on reset (drone spawns on the
        # positive side, per reset_idx's spawn-position math), so a forward pass is a + -> - crossing.
        x_now = self.env._pose_drone_wrt_gate[:, 0]
        y_now = self.env._pose_drone_wrt_gate[:, 1]
        z_now = self.env._pose_drone_wrt_gate[:, 2]
        x_prev = self.env._prev_x_drone_wrt_gate

        gate_half_extent = self.env._gate_model_cfg_data.gate_side / 2.0 * 0.8
        crossed_plane = (x_prev > 0) & (x_now <= 0)
        within_opening = (torch.abs(y_now) < gate_half_extent) & (torch.abs(z_now) < gate_half_extent)
        gate_passed = crossed_plane & within_opening
        ids_gate_passed = torch.where(gate_passed)[0]

        # Distance from the gate's centerline at the moment of passing, for the quality-scaled bonus.
        pass_offset_from_center = torch.sqrt(y_now**2 + z_now**2)

        self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % self.env._waypoints.shape[0]
        self.env._n_gates_passed[ids_gate_passed] += 1

        # set desired positions in the world frame
        self.env._desired_pos_w[ids_gate_passed, :2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], :2]
        self.env._desired_pos_w[ids_gate_passed, 2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], 2]

        # Cache this step's gate-relative x for next step's sign-change check. For envs that just
        # passed, _idx_wp changed above, so next step's _pose_drone_wrt_gate will already be in the
        # *new* gate's frame -- recompute against that new target now, before caching, so the
        # comparison next step stays apples-to-apples instead of comparing two unrelated gates' frames.
        self.env._prev_x_drone_wrt_gate = x_now.clone()
        if len(ids_gate_passed) > 0:
            new_pose_wrt_gate, _ = subtract_frame_transforms(
                self.env._waypoints[self.env._idx_wp[ids_gate_passed], :3],
                self.env._waypoints_quat[self.env._idx_wp[ids_gate_passed], :],
                self.env._robot.data.root_link_pos_w[ids_gate_passed],
            )
            self.env._prev_x_drone_wrt_gate[ids_gate_passed] = new_pose_wrt_gate[:, 0]

        # Progress reward: potential-based shaping on the *decrease* in distance to the current
        # goal, not raw closeness -- rewards actually moving toward the gate rather than just being
        # near it (the Milestone-1 checkpoint settled for "close enough" near a gate corner under the
        # old closeness-bonus form, instead of flying to the goal -- see EXPERIMENT_LOG.md).
        distance_to_goal = torch.linalg.norm(self.env._desired_pos_w - self.env._robot.data.root_link_pos_w, dim=1)
        progress = self.env._last_distance_to_goal - distance_to_goal
        progress = torch.clamp(progress, min=-1.0, max=1.0)  # defensive bound; shouldn't bind at realistic speeds
        # Zero out on the exact step the target gate switched: comparing distance-to-old-target
        # against distance-to-new-target isn't a meaningful measure of that step's actual progress.
        progress[ids_gate_passed] = 0.0
        self.env._last_distance_to_goal = distance_to_goal.clone()

        # compute crashed environments if contact detected for 100 timesteps
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask

        if self.cfg.is_train:
            gate_pass_bonus = torch.zeros(self.num_envs, device=self.device)
            gate_pass_bonus[ids_gate_passed] = torch.clamp(1.0 - pass_offset_from_center[ids_gate_passed], min=0.0)

            rewards = {
                "progress_goal": progress * self.env.rew['progress_goal_reward_scale'],
                "gate_pass": gate_pass_bonus * self.env.rew['gate_pass_reward_scale'],
                "crash": crashed * self.env.rew['crash_reward_scale'],
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            reward = torch.where(self.env.reset_terminated,
                                torch.ones_like(reward) * self.env.rew['death_cost'], reward)

            # Logging
            for key, value in rewards.items():
                self._episode_sums[key] += value
        else:   # This else condition implies eval is called with play_race.py. Can be useful to debug at test-time
            reward = torch.zeros(self.num_envs, device=self.device)

        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Fully egocentric observation vector (19-dim): body-frame velocities, a body-frame gravity
        vector for attitude (avoids the world-frame quaternion's q/-q double-cover ambiguity), and
        gate-relative position -- expressed in the *drone's* body frame, for both the current and
        next gate (lookahead for the powerloop/chicane sequences) -- plus the previous action. See
        EXPERIMENT_LOG.md for the frame-choice reasoning behind each component."""

        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b
        drone_ang_vel_b = self.env._robot.data.root_ang_vel_b
        projected_gravity_b = self.env._robot.data.projected_gravity_b

        current_gate_pos_w = self.env._waypoints[self.env._idx_wp, :3]
        next_idx_wp = (self.env._idx_wp + 1) % self.env._waypoints.shape[0]
        next_gate_pos_w = self.env._waypoints[next_idx_wp, :3]

        current_gate_pos_b, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_pos_w, self.env._robot.data.root_quat_w, current_gate_pos_w
        )
        next_gate_pos_b, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_pos_w, self.env._robot.data.root_quat_w, next_gate_pos_w
        )

        obs = torch.cat(
            [
                drone_lin_vel_b,             # body-frame linear velocity (3)
                drone_ang_vel_b,              # body-frame angular velocity / body rates (3)
                projected_gravity_b,          # body-frame gravity direction -- attitude (3)
                current_gate_pos_b,           # current target gate, in the drone's body frame (3)
                next_gate_pos_b,               # next gate after that -- lookahead (3)
                self.env._previous_actions,   # previous action (4)
            ],
            dim=-1,
        )
        observations = {"policy": obs}

        return observations

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # Logging for training mode
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            self.env.extras["log"].update(extras)

        # Call robot reset first
        self.env._robot.reset(env_ids)

        # Initialize model paths if needed
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]

            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)

            self.env._models_paths_initialized = True

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # Reset action buffers
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        # Reset joints state
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids]

        # TODO ----- START ----- Define the initial state during training after resetting an environment.
        # This example code initializes the drone 2m behind the first gate. You should delete it or heavily
        # modify it once you begin the racing task.

        # start from the zeroth waypoint (beginning of the race)
        waypoint_indices = torch.zeros(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)

        # get starting poses behind waypoints
        x0_wp = self.env._waypoints[waypoint_indices][:, 0]
        y0_wp = self.env._waypoints[waypoint_indices][:, 1]
        theta = self.env._waypoints[waypoint_indices][:, -1]
        z_wp = self.env._waypoints[waypoint_indices][:, 2]

        x_local = -2.0 * torch.ones(n_reset, device=self.device)
        y_local = torch.zeros(n_reset, device=self.device)
        z_local = torch.zeros(n_reset, device=self.device)

        # rotate local pos to global frame
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_rot = cos_theta * x_local - sin_theta * y_local
        y_rot = sin_theta * x_local + cos_theta * y_local
        initial_x = x0_wp - x_rot
        initial_y = y0_wp - y_rot
        initial_z = z_local + z_wp

        default_root_state[:, 0] = initial_x
        default_root_state[:, 1] = initial_y
        default_root_state[:, 2] = initial_z

        # point drone towards the zeroth gate
        initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
        quat = quat_from_euler_xyz(
            torch.zeros(1, device=self.device),
            torch.zeros(1, device=self.device),
            initial_yaw + torch.empty(1, device=self.device).uniform_(-0.15, 0.15)
        )
        default_root_state[:, 3:7] = quat
        # TODO ----- END -----

        # Handle play mode initial position
        if not self.cfg.is_train:
            # x_local and y_local are randomly sampled
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]

            # rotate local pos to global frame
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = 0.05

            # point drone towards the zeroth gate
            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)

            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0)
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0

            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0
            )
            default_root_state[:, 3:7] = quat
            waypoint_indices = self.env._initial_wp

        # Set waypoint indices and desired positions
        self.env._idx_wp[env_ids] = waypoint_indices

        self.env._desired_pos_w[env_ids, :2] = self.env._waypoints[waypoint_indices, :2].clone()
        self.env._desired_pos_w[env_ids, 2] = self.env._waypoints[waypoint_indices, 2].clone()

        # Full 3D (not just XY) to match get_rewards()'s progress-delta calculation -- this buffer
        # was dead/unread code before Phase 2b started using it, so there's no prior convention to
        # preserve, and this track has real z-variation between gates (0.75m vs 2.0m).
        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids, :] - self.env._robot.data.root_link_pos_w[env_ids, :], dim=1
        )
        self.env._n_gates_passed[env_ids] = 0

        # Write state to simulation
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset variables
        self.env._yaw_n_laps[env_ids] = 0

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3]
        )

        self.env._prev_x_drone_wrt_gate[env_ids] = 1.0

        self.env._crashed[env_ids] = 0