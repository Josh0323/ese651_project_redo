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

        # Per-episode metric accumulators (reset in reset_idx)
        self._episode_gate_pass_errors = torch.zeros(self.num_envs, device=self.device)
        self._episode_speed_at_pass    = torch.zeros(self.num_envs, device=self.device)
        self._episode_backwards_count  = torch.zeros(self.num_envs, device=self.device)

        if self.cfg.is_train and hasattr(env, 'rew'):
            keys = [key.split("_reward_scale")[0] for key in env.rew.keys() if key != "death_cost"]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in keys
            }

    def get_rewards(self) -> torch.Tensor:
        """Dense progress (delta distance), sparse gate pass, crash penalty; strict plane + aperture gate crossing."""

        n_gates = self.env._waypoints.shape[0]
        drone_pos_w = self.env._robot.data.root_link_pos_w

        # p_now, _ = subtract_frame_transforms(wp_pos, wp_quat, drone_pos_w)
        # pose_prev = self._pose_gate_prev

        current_x = self.env._pose_drone_wrt_gate[:, 0]
        current_y = self.env._pose_drone_wrt_gate[:, 1]
        current_z = self.env._pose_drone_wrt_gate[:, 2]
        prev_x    = self.env._prev_x_drone_wrt_gate

        half_side     = 0.5   # gate opening is 1 m × 1 m
        within_bounds = (torch.abs(current_y) < half_side) & (torch.abs(current_z) < half_side)
        gate_passed   = (prev_x > 0) & (current_x <= 0) & within_bounds
        backwards     = (prev_x < 0) & (current_x >= 0) & within_bounds

        ids_gate_passed = torch.where(gate_passed)[0]
        ids_backwards   = torch.where(backwards)[0]

        # Advance waypoint index and counters for envs that passed a gate
        if len(ids_gate_passed) > 0:
            self.env._idx_wp[ids_gate_passed] = (
                self.env._idx_wp[ids_gate_passed] + 1
            ) % n_gates
            self.env._n_gates_passed[ids_gate_passed] += 1

            # Update desired position to new gate
            new_idx = self.env._idx_wp[ids_gate_passed]
            self.env._desired_pos_w[ids_gate_passed, :3] = self.env._waypoints[new_idx, :3]

            # Update last-distance for progress reward (reset to distance to NEW gate)
            new_gate_pos = self.env._waypoints[new_idx, :3]
            self.env._last_distance_to_goal[ids_gate_passed] = torch.linalg.norm(
                new_gate_pos - drone_pos_w[ids_gate_passed], dim=1
            )

            # Set prev_x relative to NEW gate (so detection works immediately next step)
            new_gate_quat = self.env._waypoints_quat[new_idx, :]
            new_pose, _ = subtract_frame_transforms(
                new_gate_pos, new_gate_quat, drone_pos_w[ids_gate_passed]
            )
            self.env._prev_x_drone_wrt_gate[ids_gate_passed] = new_pose[:, 0]

            # Accumulate WandB metrics at gate passage
            gate_pass_err = torch.sqrt(current_y[ids_gate_passed] ** 2 +
                                       current_z[ids_gate_passed] ** 2)
            speed_now = torch.linalg.norm(
                self.env._robot.data.root_com_lin_vel_b[ids_gate_passed], dim=1
            )
            self._episode_gate_pass_errors[ids_gate_passed] += gate_pass_err
            self._episode_speed_at_pass[ids_gate_passed]    += speed_now
        # Update prev_x for envs that did NOT pass a gate this step
        not_passed_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        not_passed_mask[ids_gate_passed] = False
        self.env._prev_x_drone_wrt_gate[not_passed_mask] = current_x[not_passed_mask]

        # Handle backwards traversal — force termination next step
        if len(ids_backwards) > 0:
            self.env._crashed[ids_backwards] = 200   # exceeds _get_dones threshold of 100
            self._episode_backwards_count[ids_backwards] += 1.0

        # ------------------------------------------------------------------ #
        # r_prog: progress reward (arxiv 2406.12505, λ₁ = 0.5)              #
        # r_prog = d_{t-1} - d_t  (positive = moved closer to gate)         #
        # ------------------------------------------------------------------ #
        dist_to_gate = torch.linalg.norm(
            self.env._waypoints[self.env._idx_wp, :3] - drone_pos_w, dim=1
        )
        r_prog = self.env._last_distance_to_goal - dist_to_gate
        # Clamp to avoid large negative spike when target gate switches
        r_prog = torch.clamp(r_prog, min=-0.5, max=5.0)
        self.env._last_distance_to_goal = dist_to_gate.clone()

        # ------------------------------------------------------------------ #
        # r_pass: sparse gate-pass bonus                                      #
        # ------------------------------------------------------------------ #
        r_pass = gate_passed.float()

        # ------------------------------------------------------------------ #
        # r_crash: per-step contact penalty + crash accumulator               #
        # ------------------------------------------------------------------ #
        contact_forces = self.env._contact_sensor.data.net_forces_w  # (N, 1, 3)
        crashed_now    = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask           = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed_now * mask
        r_crash = crashed_now.float()

        if self.cfg.is_train:
            # TODO ----- START ----- Compute per-timestep rewards by multiplying with your reward scales (in train_race.py)
            rewards = {
                "prog_delta": r_prog * self.env.rew["prog_delta_reward_scale"],
                "gate_pass": r_pass * self.env.rew["gate_pass_reward_scale"],
                "crash": r_crash * self.env.rew["crash_reward_scale"],
            }
            reward = r_prog + r_pass + r_crash
            reward = torch.where(
                self.env.reset_terminated,
                torch.ones_like(reward) * self.env.rew["death_cost"],
                reward,
            )
            for key, value in rewards.items():
                self._episode_sums[key] += value
        else:
            reward = torch.zeros(self.num_envs, device=self.device)
            # TODO ----- END -----

        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations. Read reset_idx() and quadcopter_env.py to see which drone info is extracted from the sim.
        The following code is an example. You should delete it or heavily modify it once you begin the racing task."""

        # TODO ----- START ----- Define tensors for your observation space. Be careful with frame transformations
        #### Basic drone states, modify for your needs)
        drone_pose_w = self.env._robot.data.root_link_pos_w
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b
        drone_quat_w = self.env._robot.data.root_quat_w

        ##### Some example observations you may want to explore using
        # Angular velocities (referred to as body rates)
        # drone_ang_vel_b = self.env._robot.data.root_ang_vel_b  # [roll_rate, pitch_rate, yaw_rate]

        # Current target gate information
        # current_gate_idx = self.env._idx_wp
        # current_gate_pos_w = self.env._waypoints[current_gate_idx, :3]  # World position of current gate
        # current_gate_yaw = self.env._waypoints[current_gate_idx, -1]    # Yaw orientation of current gate

        # Relative position to current gate in gate frame
        drone_pos_gate_frame = self.env._pose_drone_wrt_gate

        # Relative position to current gate in body frame
        # gate_pos_b, _ = subtract_frame_transforms(
        #     self.env._robot.data.root_link_pos_w,
        #     self.env._robot.data.root_quat_w,
        #     current_gate_pos_w
        # )

        # Previous actions
        # prev_actions = self.env._previous_actions  # Shape: (num_envs, 4)

        # Number of gates passed
        # gates_passed = self.env._n_gates_passed.unsqueeze(1).float()

        # TODO ----- END -----

        obs = torch.cat(
            # TODO ----- START ----- List your observation tensors here to be concatenated together
            [
                drone_pose_w,       # position in the world frame (3 dims)
                drone_lin_vel_b,    # velocity in the body frame (3 dims)
                drone_quat_w,       # quaternion in the world frame (4 dims)
                drone_pos_gate_frame
            ],
            # TODO ----- END -----
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

        self._episode_gate_pass_errors[env_ids] = 0.0
        self._episode_speed_at_pass[env_ids]    = 0.0
        self._episode_backwards_count[env_ids]  = 0.0

        default_root_state = self.env._robot.data.default_root_state[env_ids]

        # TODO ----- START ----- Define the initial state during training after resetting an environment.
        # Random logical gate per env; spawn 2 m behind that gate (local -x), facing the gate opening.

        n_wp = self.env._waypoints.shape[0]
        waypoint_indices = torch.randint(
            0, n_wp, (n_reset,), device=self.device, dtype=self.env._idx_wp.dtype
        )

        # get starting poses behind waypoints
        x0_wp = self.env._waypoints[waypoint_indices][:, 0]
        y0_wp = self.env._waypoints[waypoint_indices][:, 1]
        theta = self.env._waypoints[waypoint_indices][:, -1]
        z_wp = self.env._waypoints[waypoint_indices][:, 2]

        x_local = -torch.empty(n_reset, device=self.device).uniform_(0.5, 3.0)
        y_local =  torch.empty(n_reset, device=self.device).uniform_(-0.4, 0.4)
        z_local =  torch.empty(n_reset, device=self.device).uniform_(-0.15, 0.15)

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

        # point drone towards the gate with small per-env yaw noise
        yaw_noise = torch.empty(n_reset, device=self.device).uniform_(-0.15, 0.15)
        initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x) + yaw_noise
        quat = quat_from_euler_xyz(
            torch.zeros(n_reset, device=self.device),
            torch.zeros(n_reset, device=self.device),
            initial_yaw,
        )
        default_root_state[:, 3:7] = quat
        # TODO ----- END -----
    # ---- Starting velocity: random speed toward gate center ----
        gate_pos_tgt = self.env._waypoints[waypoint_indices, :3]  # (N, 3)
        start_pos    = torch.stack([initial_x, initial_y, initial_z], dim=1)
        direction    = gate_pos_tgt - start_pos
        direction    = direction / (torch.linalg.norm(direction, dim=1, keepdim=True) + 1e-6)
        if self.cfg.is_train:
            speed = torch.empty(n_reset, device=self.device).uniform_(0.0, 2.0).unsqueeze(1)
        else:
            speed = torch.zeros(n_reset, 1, device=self.device)
        initial_vel = direction * speed          # (N, 3)

        default_root_state[:, 7:10]  = initial_vel
        default_root_state[:, 10:13] = 0.0       # zero angular velocity

        # TODO ----- START ----- Define the initial state during play mode after resetting an environment.
        # Handle play mode initial position
        if not self.cfg.is_train:
            # x_local and y_local are randomly sampled
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0,  1.0)
            z_local = torch.zeros(1, device=self.device)

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

        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids] - self.env._robot.data.root_link_pos_w[env_ids], dim=1
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

        # self.env._prev_x_drone_wrt_gate[env_ids] = 1.0
        # ------------------------------------------------------------------ #
        # _prev_x_drone_wrt_gate                                              #
        # Sign convention: x_local < 0 → gate_frame_x ≈ +|x_local| > 0      #
        # Using -x_local directly avoids stale sim readback issues            #
        # (root_link_state_w may not reflect the newly written position yet). #
        # ------------------------------------------------------------------ #
        self.env._prev_x_drone_wrt_gate[env_ids] = -x_local  # guaranteed positive

        self.env._crashed[env_ids] = 0