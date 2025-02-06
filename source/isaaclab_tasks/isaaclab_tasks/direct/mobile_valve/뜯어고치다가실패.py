# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom , Gf

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import sample_uniform


@configclass
class MobileValveEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    action_space = 12
    observation_space = 29
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/vision/Downloads/floating_franka_original.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                # 모바일 베이스
                "base_joint_x": 0.0,
                "base_joint_y": 0.0,
                "base_joint_z": 0.0,
                # Franka Arm
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.747,
                "panda_joint5": 0.0,
                "panda_joint6": 2.0,
                "panda_joint7": 0.741,
                # 그립퍼
                "panda_finger_joint1": 0.02,
                "panda_finger_joint2": 0.02,
            },
            pos=(0, 0.0, 0.0), 
            rot=(0.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            # 모바일 베이스
            "base": ImplicitActuatorCfg(
                joint_names_expr=["base_joint_.*"], 
                stiffness=1e4,
                damping=1e4,
            ),
            # 프랑카 어깨 관절
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=800.0,
                damping=40.0,
            ),
            # 프랑카 팔 관절
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=800.0,
                damping=40.0,
            ),
            # 그립퍼
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=1e5,
                damping=1e3,
            ),
        },
    )

    # cabinet
    valve = ArticulationCfg(
        prim_path="/World/envs/env_.*/Valve",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/vision/Downloads/valves/round_valve/round_valve_main.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(2.0, 0, 0.5),
            rot=(0.0, 0.0, 0.0, 0.0),
            joint_pos={
                "valve_handle_joint": 0.0,
            },
        ),
        actuators={
            "valve": ImplicitActuatorCfg(
                joint_names_expr=["valve_handle_joint"],
                effort_limit=100.0,
                velocity_limit=10.0,
                stiffness=20.0,
                damping=10.0,
            ),
        },
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    action_scale = 7.5
    dof_velocity_scale = 0.1


class MobileValveEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: MobileValveEnvCfg

    def __init__(self, cfg: MobileValveEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # Create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        # Base joints 설정
        base_joints = self._robot.find_joints("base_joint_.*")[0]
        print(f"Base Joints: {base_joints}")
        if isinstance(base_joints, list):
            for joint in base_joints:
                self.robot_dof_speed_scales[joint] = 1.0
        else:
            self.robot_dof_speed_scales[base_joints] = 1.0

        # Gripper joints 설정
        gripper_joints = self._robot.find_joints("panda_finger_joint.*")[0]
        print(f"Gripper Joints: {gripper_joints}")
        if isinstance(gripper_joints, list):
            for joint in gripper_joints:
                self.robot_dof_speed_scales[joint] = 0.1
        else:
            self.robot_dof_speed_scales[gripper_joints] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")),
            self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_leftfinger")),
            self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_rightfinger")),
            self.device,
        )

        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        robot_local_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        # 밸브 핸들 관련 로컬 포즈 (예시 값, 실제 상황에 맞게 수정 필요)
        valve_local_grasp_pose = torch.tensor([0.1150, 0.15, 0.0751, 0, 0, -0.707, -0.707], device=self.device)
        self.valve_local_grasp_pos = valve_local_grasp_pose[0:3].repeat((self.num_envs, 1))
        self.valve_local_grasp_rot = valve_local_grasp_pose[3:7].repeat((self.num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        # 밸브를 회전시키는 방향이나 축은 실제 USD에 맞게 조정
        self.valve_inward_axis = torch.tensor([0, 0, -1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.valve_up_axis = torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )

        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]

        # 밸브 핸들 바디 인덱스를 찾는다.
        self.valve_link_idx = self._valve.find_bodies("valve_handle")[0][0]
        self.base_link_idx = self._robot.find_bodies("robot_base_link")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.valve_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.valve_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)


    def _setup_scene(self):
        # 로봇과 밸브 아티큘레이션을 초기화
        self._robot = Articulation(self.cfg.robot)
        self._valve = Articulation(self.cfg.valve)  # 'cabinet'을 'valve'로 변경

        # 씬에 아티큘레이션 추가
        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["valve"] = self._valve  # 'cabinet'을 'valve'로 변경

        # 지형 설정
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # 환경 복제, 필터링 및 복제
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # 조명 추가
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        self.robot_dof_targets[:, 2] = 0.0 
        
    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        truncated = self.episode_length_buf >= self.max_episode_length - 1

                # 거리 조건: 목표와의 거리
        d = torch.norm(self.robot_grasp_pos - self.valve_grasp_pos, p=2, dim=-1)
        close_condition = d <= 0.025

        # 손가락 닫힘 조건: 손가락 간 거리가 충분히 작은 경우
        finger_distance = torch.norm(self._robot.data.body_pos_w[:, self.left_finger_link_idx] -
                                    self._robot.data.body_pos_w[:, self.right_finger_link_idx], p=2, dim=-1)
        finger_close_condition = finger_distance < 0.01

        # 성공 조건: 목표 근처에서 손가락이 닫혀야 함
        terminated = close_condition & finger_close_condition
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        # 왼손가락과 오른손가락 위치 가져오기
        robot_left_finger_pos = self._robot.data.body_pos_w[:, self.left_finger_link_idx]
        robot_right_finger_pos = self._robot.data.body_pos_w[:, self.right_finger_link_idx]

        # 모바일 베이스와 밸브 위치 가져오기
        robot_base_pos = self._robot.data.body_pos_w[:, self.base_link_idx]
        valve_base_pos = self._valve.data.body_pos_w[:, self.valve_link_idx]

        # 보상 계산 호출
        return self._compute_rewards(
            self.actions,
            self._valve.data.joint_pos,             # valve의 joint position
            self.robot_grasp_pos,
            self.valve_grasp_pos,                   # valve grasp position
            self.robot_grasp_rot,
            self.valve_grasp_rot,                   # valve grasp rotation
            robot_left_finger_pos,
            robot_right_finger_pos,
            self.gripper_forward_axis,
            self.valve_inward_axis,                 # valve 방향 축
            self.gripper_up_axis,
            self.valve_up_axis,                     # valve의 위쪽 축
            robot_base_pos,                         # 모바일 베이스 위치
            valve_base_pos,                         # 밸브의 기준 위치
            self.num_envs,     
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # valve state
        zeros = torch.zeros((len(env_ids), self._valve.num_joints), device=self.device)
        self._valve.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict:
        # 조인트 위치를 스케일링하여 [-1, 1] 범위로 정규화
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        
        # 로봇 그립 위치와 밸브 그립 위치 간의 차이 계산
        to_target = self.valve_grasp_pos - self.robot_grasp_pos

        # 관측 벡터 구성
        obs = torch.cat(
            (
                dof_pos_scaled,                              # 조인트 위치 스케일링
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,  # 조인트 속도 스케일링
                to_target,                                    # 목표 지점과의 거리
                self._valve.data.joint_pos[:, 0].unsqueeze(-1),  # 밸브 핸들의 조인트 위치 (예: valve_handle_joint)
                self._valve.data.joint_vel[:, 0].unsqueeze(-1),  # 밸브 핸들의 조인트 속도
            ),
            dim=-1,
        )
        
        # 관측 벡터를 [-5.0, 5.0] 범위로 클램프하여 반환
        return {"policy": torch.clamp(obs, -5.0, 5.0)}


    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        # 로봇의 손 위치 및 회전 정보
        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]

        # 밸브의 핸들 위치 및 회전 정보
        valve_pos = self._valve.data.body_pos_w[env_ids, self.valve_link_idx]
        valve_rot = self._valve.data.body_quat_w[env_ids, self.valve_link_idx]

        # 그랩 포즈 계산
        (
            self.robot_grasp_rot[env_ids],
            self.robot_grasp_pos[env_ids],
            self.valve_grasp_rot[env_ids],
            self.valve_grasp_pos[env_ids],
        ) = self._compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.robot_local_grasp_rot[env_ids],
            self.robot_local_grasp_pos[env_ids],
            valve_rot,
            valve_pos,
            self.valve_local_grasp_rot[env_ids],
            self.valve_local_grasp_pos[env_ids],
        )

        # 선택된 환경 ID에 맞게 valve_grasp_pos 가져오기
        valve_grasp_pos = self.valve_grasp_pos.index_select(0, env_ids)

        # X, Y 좌표만 고려하여 벡터 계산 (Z축 값은 무시)
        look_at_vector_xy = valve_pos[:, :2] - valve_grasp_pos[:, :2]

        # Z축 값을 0으로 설정
        z_values = torch.zeros((look_at_vector_xy.shape[0], 1), device=self.device)

        # look_at_vector 생성
        look_at_vector = torch.cat((look_at_vector_xy, z_values), dim=-1)

        # 벡터 정규화 (0으로 나누는 것을 방지)
        norms = torch.norm(look_at_vector, p=2, dim=-1, keepdim=True)
        norms = torch.clamp(norms, min=1e-6)
        look_at_vector = look_at_vector / norms

        # axis4 업데이트
        self.axis4 = look_at_vector



    def _compute_rewards(
        self,
        actions,
        valve_dof_pos,
        robot_grasp_pos,
        valve_grasp_pos,
        robot_grasp_rot,
        valve_grasp_rot,
        robot_lfinger_pos,
        robot_rfinger_pos,
        gripper_forward_axis,
        valve_inward_axis,
        gripper_up_axis,
        valve_up_axis,
        robot_base_pos,
        valve_base_pos,
        num_envs,
    ):
        # 1. Franka와 Valve 간 거리 계산 보상
        d = torch.norm(robot_grasp_pos - valve_grasp_pos, p=2, dim=-1)
        franka_distance_reward = 1.0 / (1.0 + d**2)
        franka_distance_reward = franka_distance_reward.pow(2)
        franka_distance_reward = torch.where(d <= 0.02, franka_distance_reward * 2, franka_distance_reward)

        # 2. Franka의 축 정렬 계산
        axis1 = tf_vector(robot_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(valve_grasp_rot, valve_inward_axis)
        axis3 = tf_vector(robot_grasp_rot, gripper_up_axis)
        axis4 = self.axis4

        dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)

        # Franka의 축 정렬 보상
        rot_reward = torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2

        # 3. 행동 제어 패널티 계산
        action_penalty = torch.sum(actions**2, dim=-1)

        # 4. 총 보상 계산
        rewards = (
            2.0 * franka_distance_reward  # Franka의 거리 보상
            + 1.5 * rot_reward            # Franka의 축 정렬 보상
            - 0.1 * action_penalty       # 행동 제어 패널티
        )
        print("robot_dof_targets:", self.robot_dof_targets[:, :2])
        # 로깅 정보
        self.extras["log"] = {
            "franka_distance_reward": franka_distance_reward.mean(),
            "rot_reward": rot_reward.mean(),
            "action_penalty": -action_penalty.mean(),
            "d": d.mean(),
        }

        #self.visualize_axes()

        return rewards


    def _compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        valve_rot,
        valve_pos,
        valve_local_grasp_rot,
        valve_local_grasp_pos,
    ):
        # 로봇의 그랩 포즈를 글로벌 좌표계로 변환
        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        
        # 밸브의 그랩 포즈를 글로벌 좌표계로 변환
        global_valve_rot, global_valve_pos = tf_combine(
            valve_rot, valve_pos, valve_local_grasp_rot, valve_local_grasp_pos
        )

        return global_franka_rot, global_franka_pos, global_valve_rot, global_valve_pos


    def _post_physics_step(self):
        self.visualize_axes()  # 매 스텝마다 축을 시각적으로 업데이트

    def visualize_axes(self):
        stage = get_current_stage()

        # 축 시각화를 위한 설정 함수
        def create_or_update_axis_line(start_pos, direction, length, path, color):
            end_pos = start_pos + direction * length
            line_prim = stage.GetPrimAtPath(path)

            if not line_prim:  # Prim이 없을 때 생성
                line = UsdGeom.BasisCurves.Define(stage, path)
                line.CreateWidthsAttr([0.01, 0.01])
                line.CreateTypeAttr().Set("linear")
                line.CreateDisplayColorAttr([color])
                line.GetCurveVertexCountsAttr().Set([2])
            else:
                line = UsdGeom.BasisCurves(line_prim)

            # 강제로 최신 좌표로 업데이트 (USD에서 값 반영 보장)
            points = [Gf.Vec3f(*start_pos.cpu().numpy().tolist()), Gf.Vec3f(*end_pos.cpu().numpy().tolist())]
            line.GetPointsAttr().Set(points)
            line.GetPointsAttr().Set(points)  # 두 번 설정하여 강제 갱신

        # axis1, axis2, axis3, axis4 계산
        axis1 = tf_vector(self.robot_grasp_rot, self.gripper_forward_axis)  # EE의 "앞" 방향
        axis2 = tf_vector(self.valve_grasp_rot, self.valve_inward_axis)    # 서랍 손잡이의 "안쪽" 방향
        axis3 = tf_vector(self.robot_grasp_rot, self.gripper_up_axis)       # EE의 "위쪽" 방향
        #axis4 = tf_vector(self.valve_grasp_rot, self.valve_up_axis)        # 서랍의 "위쪽" 방향

        # 각 축에 대한 시각화
        gripper_start = self.robot_grasp_pos[0]

        valve_start = self.valve_grasp_pos[0]


        # axis1, axis2, axis3, axis4 시각화 추가
        create_or_update_axis_line(gripper_start, axis1[0], 0.1, "/World/Visuals/axis1_line", (0, 0, 1))
        create_or_update_axis_line(valve_start, axis2[0], 0.1, "/World/Visuals/axis2_line", (1, 0, 0))
        create_or_update_axis_line(gripper_start, axis3[0], 0.1, "/World/Visuals/axis3_line", (0, 1, 0))
        # axis4 시각화 (look_at_vector 사용)
        create_or_update_axis_line(valve_start, self.axis4[0], 0.1, "/World/Visuals/axis4_line", (1, 1, 0))

