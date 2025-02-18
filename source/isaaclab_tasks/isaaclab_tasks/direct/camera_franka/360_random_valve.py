# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform


@configclass
class CameraFrankaEnvCfg(DirectRLEnvCfg):
    """
    모바일 베이스 + 프랑카(Franka) 로봇 팔 + 밸브(Valve) 환경을 위한 환경 설정 클래스입니다.
    시뮬레이션 시간 간격, 에피소드 길이, 동작 스케일(action_scale), 보상 스케일 등을 정의합니다.
    """

    # 환경
    episode_length_s = 8.3333
    decimation = 2
    action_space = 12   # (Mobile Base 3개 DOF + Franka Arm 9개 DOF)
    observation_space = 30
    state_space = 0

    # 시뮬레이션
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        disable_contact_processing=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # 씬
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=100.0, replicate_physics=True)

    # 로봇 설정
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/vision/Downloads/floating_franka_original.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=3.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=20,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                # 모바일 베이스
                "base_joint_x": 0.0,
                "base_joint_y": 0.0,
                "base_joint_z": 0.0,
                # 프랑카 암
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
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "base": ImplicitActuatorCfg(
                joint_names_expr=["base_joint_.*"],
                stiffness=1e4,
                damping=1e4,
            ),
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=800.0,
                damping=40.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=800.0,
                damping=40.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # 밸브(Valve) 설정
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
                velocity_limit=0.0,
                stiffness=20.0,
                damping=10.0,
            ),
        },
    )

    # 바닥 지형(Plane)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=5.0,
            dynamic_friction=3.0,
            restitution=0.0,
        ),
    )

    # 동작 및 보상 스케일
    action_scale = 2.0
    dof_velocity_scale = 0.1
    dist_reward_scale = 3
    rot_reward_scale = 5
    action_penalty_scale = 0.1
    finger_reward_scale = 2.0
    base_reward_scale = 1.0
    base_penalty_scale = 0.1


class CameraFrankaEnv(DirectRLEnv):
    """
    모바일 베이스와 프랑카 로봇 암, 그리고 밸브(Valve)가 있는 환경을 시뮬레이션하는 클래스입니다.
    Isaac Sim의 DirectRLEnv를 상속하며, RL 훈련을 위한 reset, step, reward 계산 등을 수행합니다.
    """

    cfg: CameraFrankaEnvCfg

    VIRTUAL_CAMERA_FOCAL_LENGTH = 24.0
    VIRTUAL_CAMERA_HORIZONTAL_APERTURE = 20.955
    VIRTUAL_CAMERA_OFFSET = (-0.235, 0, 1.5)     # 로봇 베이스 중심 기준 오프셋
    VIRTUAL_LOCAL_CAMERA_FORWARD = (-1, 0, 0)    # 로컬 카메라 forward 방향

    def __init__(self, cfg: CameraFrankaEnvCfg, render_mode: str | None = None, **kwargs):
        """
        초기화에서는:
          - 로봇의 조인트 제한, 속도 스케일 설정
          - 환경마다 밸브와 로봇 초기화
          - 그립(Grasp) 포즈 설정 등에 필요한 참조 값들을 미리 계산합니다.
        """
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """
            주어진 프림(xformable)의 월드 변환에서, 현재 환경(env_pos)을 기준으로 한 로컬 위치/쿼터니언을 반환합니다.
            """
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

        # 시뮬레이션 dt (config에서 가져온 dt * decimation)
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # 로봇 정보 설정
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        # 각 환경(env)별 밸브 검출 여부 플래그
        self.valve_detected_results = [False] * self.cfg.scene.num_envs

        # 이전 타겟 DOF (특히 모바일 베이스 Z축 등)를 저장하는 텐서
        self.prev_robot_dof_targets = torch.zeros((self.cfg.scene.num_envs, 3), device=self.device)

        # 타임스텝 카운터
        self.timestep = 0

        # 모바일 베이스와 그립퍼 조인트 속도 스케일 설정
        base_joints = self._robot.find_joints("base_joint_.*")[0]
        if isinstance(base_joints, list):
            for joint in base_joints:
                self.robot_dof_speed_scales[joint] = 1.0
        else:
            self.robot_dof_speed_scales[base_joints] = 1.0

        gripper_joints = self._robot.find_joints("panda_finger_joint.*")[0]
        if isinstance(gripper_joints, list):
            for joint in gripper_joints:
                self.robot_dof_speed_scales[joint] = 0.1
        else:
            self.robot_dof_speed_scales[gripper_joints] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.base_link_idx = self._robot.find_bodies("robot_base_link")[0][0]

        # 스테이지(USD Stage)에서 손과 손가락의 실제 위치를 추출해 로컬 그립 포즈 계산
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

        # 손가락 사이의 중앙 위치로 그립(Grasp) 포즈를 설정
        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]  # 회전은 왼손가락 기준

        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])
        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        robot_local_pose_pos += torch.tensor([0, 0.0, 0.0225], device=self.device)

        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        # 밸브(Valve) 핸들을 잡기 위한 로컬 포즈 설정 (환경마다 동일)
        valve_local_grasp_pose = torch.tensor(
            [0.1150, 0.15, 0.0751, 0, 0, -0.707, -0.707], device=self.device
        )
        self.valve_local_grasp_pos = valve_local_grasp_pose[0:3].repeat((self.num_envs, 1))
        self.valve_local_grasp_rot = valve_local_grasp_pose[3:7].repeat((self.num_envs, 1))

        # 축(오리엔테이션) 관련 설정
        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.valve_inward_axis = torch.tensor([0, 0, -1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.valve_up_axis = torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )

        # 링크(Hand, Finger, Valve) 인덱스
        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        self.valve_link_idx = self._valve.find_bodies("valve_handle")[0][0]

        # (N,4), (N,3) 형태로 로봇/밸브 그립 포즈를 저장할 텐서
        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.valve_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.valve_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)


    def _quat_rotate(self, quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """
        쿼터니언(quat: [w, x, y, z])을 이용해 벡터(vec)를 회전시키는 함수입니다.
        v' = v + 2 * cross(q_vec, cross(q_vec, v) + q_w * v)
        """
        q_w = quat[0]
        q_vec = quat[1:]
        t = 2 * torch.cross(q_vec, vec, dim=0)
        rotated_vec = vec + q_w * t + torch.cross(q_vec, t, dim=0)
        return rotated_vec

    def _update_valve_detection_fov(self):
        """
        (카메라 사용 없이) 로봇 베이스의 pose와 설정해둔 카메라 FOV값을 통해,
        밸브가 카메라 시야(FOV) 각도 안에 들어왔는지 간단히 계산합니다.

        각 env 별로 한 번이라도 밸브가 확인되면(True) 그대로 유지합니다(스티키).
        """
        half_fov = math.atan((self.VIRTUAL_CAMERA_HORIZONTAL_APERTURE / 2) / self.VIRTUAL_CAMERA_FOCAL_LENGTH)

        camera_offset = torch.tensor(self.VIRTUAL_CAMERA_OFFSET, device=self.device, dtype=torch.float32)
        local_camera_forward = torch.tensor(self.VIRTUAL_LOCAL_CAMERA_FORWARD, device=self.device, dtype=torch.float32)

        for env_id in range(self.num_envs):
            robot_base_pos = self._robot.data.body_pos_w[env_id, self.base_link_idx]
            robot_base_quat = self._robot.data.body_quat_w[env_id, self.base_link_idx]

            camera_pos = robot_base_pos + self._quat_rotate(robot_base_quat, camera_offset)
            camera_forward = -self._quat_rotate(robot_base_quat, local_camera_forward)

            valve_pos = self._valve.data.body_pos_w[env_id, self.valve_link_idx]
            rel_vec = valve_pos - camera_pos

            # x-y 평면에서 시야각 비교
            camera_forward_xy = camera_forward.clone()
            rel_vec_xy = rel_vec.clone()
            camera_forward_xy[2] = 0
            rel_vec_xy[2] = 0

            if torch.norm(rel_vec_xy) < 1e-6:
                current_detected = False
            else:
                camera_forward_xy = camera_forward_xy / torch.norm(camera_forward_xy)
                rel_norm_xy = rel_vec_xy / torch.norm(rel_vec_xy)
                if torch.dot(camera_forward_xy, rel_norm_xy) > 0:
                    dot_val = torch.dot(camera_forward_xy, rel_norm_xy).clamp(-1.0, 1.0)
                    angle = torch.acos(dot_val)
                    current_detected = angle < half_fov
                else:
                    current_detected = False

            # 스티키 검출
            self.valve_detected_results[env_id] = self.valve_detected_results[env_id] or current_detected

    def _setup_scene(self):
        """
        로봇, 밸브, 지형(Plane) 등을 씬(Scene)에 추가 후, 환경 복제와 물리 충돌 필터링을 설정합니다.
        그리고 라이트(조명) 등을 추가한 뒤 부모 클래스의 _setup_scene를 호출합니다.
        """
        self._robot = Articulation(self.cfg.robot)
        self._valve = Articulation(self.cfg.valve)
        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["valve"] = self._valve

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # 조명 추가 (DomeLight)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        super()._setup_scene()

    def _pre_physics_step(self, actions: torch.Tensor):
        """
        매 타임스텝(step)마다 물리 시뮬레이션 전 적용되는 함수:
          1) 행동(actions)을 받아서 제한(clamp) 후 로봇 DOF 타겟 계산
          2) 밸브 FOV 검출 업데이트
          3) 타임스텝 증가
        """
        self.actions = actions.clone().clamp(-1.0, 1.0)

        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt \
                  * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        self.timestep += 1
        self._update_valve_detection_fov()

    def _apply_action(self):
        """
        계산된 self.robot_dof_targets 값을 기반으로 물리 시뮬레이터 내에서
        로봇의 각 조인트에 position target을 적용합니다.
        """
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        에피소드 종료/트렁케이트 여부를 반환합니다.
          - truncated: 최대 에피소드 길이 도달
          - done: 목표 도달 여부(간단히 그립과 밸브가 충분히 가까운지),
                  또는 모바일 베이스가 밸브와 너무 멀어진 경우
        """
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        d = torch.norm(self.robot_grasp_pos - self.valve_grasp_pos, p=2, dim=-1)
        close_condition = d <= 0.025

        finger_distance = torch.norm(
            self._robot.data.body_pos_w[:, self.left_finger_link_idx]
            - self._robot.data.body_pos_w[:, self.right_finger_link_idx],
            p=2,
            dim=-1
        )
        finger_close_condition = finger_distance < 0.01

        success_condition = close_condition & finger_close_condition

        mobile_xy = self._robot.data.body_pos_w[:, self.base_link_idx][:, :2]
        valve_xy = self._valve.data.body_pos_w[:, self.valve_link_idx][:, :2]
        mobile_dist = torch.norm(mobile_xy - valve_xy, p=2, dim=-1)
        mobile_far_condition = mobile_dist > 5.0

        done = success_condition | mobile_far_condition
        return done, truncated

    def _get_rewards(self) -> torch.Tensor:
        """
        매 스텝마다 보상을 계산하여 반환합니다.
        """
        self._compute_intermediate_values()
        return self._compute_rewards(
            self.actions,
            self._valve.data.joint_pos,
            self.robot_grasp_pos,
            self.valve_grasp_pos,
            self.robot_grasp_rot,
            self.valve_grasp_rot,
            self._robot.data.body_pos_w[:, self.left_finger_link_idx],
            self._robot.data.body_pos_w[:, self.right_finger_link_idx],
            self.gripper_forward_axis,
            self.valve_inward_axis,
            self.gripper_up_axis,
            self.valve_up_axis,
            self._robot.data.body_pos_w[:, self.base_link_idx],
            self._valve.data.body_pos_w[:, self.valve_link_idx],
            self.num_envs,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.finger_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.base_reward_scale,
            self.cfg.base_penalty_scale,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """
        특정 환경(env_ids)들에 대해 초기화(reset)합니다.
          - 로봇 상태(베이스 위치 및 관절)
          - 밸브 조인트 및 루트 상태
          - 밸브를 임의 위치에 스폰
        """
        super()._reset_idx(env_ids)

        # ---------------------------
        # 1) 로봇
        # ---------------------------
        base_joint_indices = self._robot.find_joints("base_joint.*")[0]
        base_joint_indices = torch.tensor(base_joint_indices, dtype=torch.long, device=self.device)
        initial_positions = torch.zeros((len(env_ids), 3), device=self.device)
        self.robot_dof_targets[env_ids, :3] = initial_positions
        self._robot.set_joint_position_target(
            self.robot_dof_targets[env_ids, :3],
            joint_ids=base_joint_indices.tolist(),
            env_ids=env_ids,
        )

        joint_pos = (
            self._robot.data.default_joint_pos[env_ids]
            + sample_uniform(-0.125, 0.125, (len(env_ids), self._robot.num_joints), self.device)
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # ---------------------------
        # 2) 밸브 내부조인트 초기화
        # ---------------------------
        zeros = torch.zeros((len(env_ids), self._valve.num_joints), device=self.device)
        self._valve.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)

        # ---------------------------
        # 3) 밸브 루트 상태
        # ---------------------------
        pos = self._valve.data.root_pos_w.clone()
        quat = self._valve.data.root_quat_w.clone()
        lin_vel = self._valve.data.root_lin_vel_w.clone()
        ang_vel = self._valve.data.root_ang_vel_w.clone()
        root_state = torch.cat([pos, quat, lin_vel, ang_vel], dim=-1)

        # ---------------------------
        # 4) 밸브를 랜덤 x,y 위치에 스폰
        # ---------------------------
        random_x = sample_uniform(-2.0, 2.0, (len(env_ids), 1), device=self.device)
        random_y = sample_uniform(-2.0, 2.0, (len(env_ids), 1), device=self.device)

        new_valve_pos_local = torch.zeros((len(env_ids), 3), device=self.device)
        new_valve_pos_local[:, 0] = random_x[:, 0]
        new_valve_pos_local[:, 1] = random_y[:, 0]
        new_valve_pos_local[:, 2] = 0.5

        origins = self.scene.env_origins[env_ids]
        new_valve_pos_world = new_valve_pos_local + origins

        new_valve_quat = torch.zeros((len(env_ids), 4), device=self.device)
        new_valve_quat[:, 0] = 1.0

        root_state[env_ids, 0:3] = new_valve_pos_world
        root_state[env_ids, 3:7] = new_valve_quat
        root_state[env_ids, 7:10] = 0.0
        root_state[env_ids, 10:13] = 0.0

        self._valve.write_root_state_to_sim(root_state)

        # 밸브 검출 상태/추적 값 초기화
        for env_id in env_ids:
            self.valve_detected_results[env_id] = False

        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict:
        """
        관측(observations)을 구성하여 딕셔너리 형태로 반환합니다.
        관절 위치/속도, 로봇 그립 포즈와 밸브 포즈 차이, 모바일 베이스와 밸브 간 거리 등을 포함합니다.
        """
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        to_target = self.valve_grasp_pos - self.robot_grasp_pos

        mobile_xy = self._robot.data.body_pos_w[:, self.base_link_idx]
        valve_xy = self.valve_grasp_pos
        mobile_dist = torch.norm(mobile_xy - valve_xy, p=2, dim=-1).unsqueeze(-1)

        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                to_target,
                self._valve.data.joint_pos[:, 0].unsqueeze(-1),
                self._valve.data.joint_vel[:, 0].unsqueeze(-1),
                mobile_dist,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        """
        보상/종료 계산에 필요한 중간값들을 매 타임스텝 업데이트합니다.
          - 로봇/밸브의 그립 포즈 및 회전
          - FOV 검출 결과에 따른 로봇 DOF 타겟 수정 등
        """
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        # 로봇 손, 밸브 핸들 위치/회전
        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]
        valve_pos = self._valve.data.body_pos_w[env_ids, self.valve_link_idx]
        valve_rot = self._valve.data.body_quat_w[env_ids, self.valve_link_idx]

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

        # XY 평면에서 로봇 그립과 밸브 간 방향 벡터(look_at_vector) 계산
        valve_grasp_pos = self.valve_grasp_pos.index_select(0, env_ids)
        look_at_vector_xy = valve_pos[:, :2] - valve_grasp_pos[:, :2]
        z_values = torch.zeros((look_at_vector_xy.shape[0], 1), device=self.device)
        look_at_vector = torch.cat((look_at_vector_xy, z_values), dim=-1)
        look_at_vector = look_at_vector / torch.norm(look_at_vector, p=2, dim=-1, keepdim=True)
        self.axis4 = look_at_vector

        # 모바일 베이스와 밸브 간 거리
        robot_base_pos = self._robot.data.body_pos_w[:, self.base_link_idx]
        valve_base_pos = self._valve.data.body_pos_w[:, self.valve_link_idx]
        mobile_dist = torch.norm(robot_base_pos[:, :2] - valve_base_pos[:, :2], p=2, dim=-1)

        # FOV 검출(plausible_detected) 여부에 따라 로봇 DOF 타겟 제어 로직
        detected = torch.tensor(self.valve_detected_results, device=self.device)
        not_detected = ~detected

        # (1) 아직 밸브를 못 봤으면(= not_detected) 프랑카 관절 일부 초기화(또는 특정 값)
        self.robot_dof_targets[not_detected, 3:10] = 0.0
        self.robot_dof_targets[not_detected, 4:5] = 70.0
        self.robot_dof_targets[not_detected, 5:6] = 1.5
        self.robot_dof_targets[not_detected, 6:7] = -160.0
        self.robot_dof_targets[not_detected, 8:9] = 100.0
        self.robot_dof_targets[not_detected, 0:2] = 0.0

        # (2) 모바일 베이스가 밸브 주변 일정 거리(0.75~0.85) 범위에 있지 않으면,
        #     일단 이동 단계로 간주하여 프랑카 관절들은 단순 유지
        mask = (mobile_dist < 0.75) | (mobile_dist > 0.85)
        combined_mask = detected & mask
        self.robot_dof_targets[combined_mask, 3:7] = 0.0

        # (3) 손 높이와 밸브 높이를 비교하여 그립퍼 벌림/닫힘 제어
        z_diff = torch.abs(self.robot_grasp_pos[:, 2] - self.valve_grasp_pos[:, 2])
        finger_mask_open = z_diff > 0.01
        finger_mask_close = z_diff <= 0.01
        self.robot_dof_targets[finger_mask_open, 10:12] = 0.04
        self.robot_dof_targets[finger_mask_close, 10:12] = 0.0

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
        dist_reward_scale,
        rot_reward_scale,
        finger_reward_scale,
        action_penalty_scale,
        base_reward_scale,
        base_penalty_scale, 
    ):
        """
        보상 계산 로직을 담당합니다.
        (FOV를 통한 밸브 검출, 모바일 베이스 이동 보상, 로봇 팔 정렬/접근 보상, 행동 패널티 등을 모두 종합)
        """
        valve_detected_tensor = torch.tensor(self.valve_detected_results, device=self.device, dtype=torch.float32)
        fov_reward = torch.where(valve_detected_tensor > 0, 1.0, -1.0)

        # 모바일 베이스 - 밸브 거리 보상
        mobile_xy = robot_base_pos[:, :2]
        valve_xy = valve_base_pos[:, :2]
        mobile_dist = torch.norm(mobile_xy - valve_xy, p=2, dim=-1)

        target_dist = 0.8
        max_reward = 5.0
        reward_slope = max_reward / 0.05
        penalty_slope = -2.0

        # 범위(0.75~0.85) 내면 점차 보상을 극대화, 아니면 패널티
        mobile_distance_reward = torch.where(
            (mobile_dist >= 0.75) & (mobile_dist <= 0.85),
            max_reward - reward_slope * torch.abs(mobile_dist - target_dist),
            penalty_slope * torch.abs(mobile_dist - target_dist),
        )

        # YOLO 탐지 부분과 유사하게 쓰지만 실제론 FOV 결과 사용
        is_mobile_phase = (mobile_dist < 0.75) | (mobile_dist > 0.85)
        yolo_reward = []
        for env_id in range(num_envs):
            if valve_detected_tensor[env_id] or (not is_mobile_phase[env_id]):
                yolo_reward.append(1.0)
            else:
                yolo_reward.append(-1.0)
        yolo_reward = torch.tensor(yolo_reward, device=self.device)

        # Franka(팔) 보상
        xy_distance = torch.norm(robot_grasp_pos[:, :2] - valve_grasp_pos[:, :2], p=2, dim=-1)
        xy_alignment_reward = 10.0 * torch.exp(-3.0 * xy_distance) * (~is_mobile_phase).float()
        xy_alignment_penalty = torch.where(
            (xy_distance > 0.08) & (~is_mobile_phase),
            torch.tensor(-5.0, device=self.device),
            torch.tensor(0.0, device=self.device),
        )

        is_xy_aligned = (xy_distance < 0.05) & (~is_mobile_phase)
        z_diff = torch.abs(robot_grasp_pos[:, 2] - valve_grasp_pos[:, 2])
        z_reward = torch.where(is_xy_aligned, 10.0 * torch.exp(-3.0 * z_diff), 0.0)

        axis1 = tf_vector(robot_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(valve_grasp_rot, valve_inward_axis)
        axis3 = tf_vector(robot_grasp_rot, gripper_up_axis)
        axis4 = self.axis4

        dot1 = (axis1 * axis2).sum(dim=-1)
        dot2 = (axis3 * axis4).sum(dim=-1)
        rot_reward = 5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2) * (~is_mobile_phase).float()

        is_fully_aligned = is_xy_aligned & (z_diff < 0.025)
        final_reward = torch.where(is_fully_aligned, 100.0 * rot_reward / 10.0, 0.0)

        mobile_action_penalty = torch.sum(actions[:, :2] ** 2, dim=-1) / actions[:, :2].shape[-1]
        franka_action_penalty = torch.sum(actions[:, 3:10] ** 2, dim=-1) / actions[:, 3:10].shape[-1]
        action_penalty = mobile_action_penalty + franka_action_penalty

        rewards = (
            fov_reward
            + mobile_distance_reward
            + xy_alignment_reward
            + xy_alignment_penalty
            + z_reward
            + final_reward
            + rot_reward
            - action_penalty
        )

        self.extras["log"] = {
            "total_rewards": rewards.mean().item(),
            "fov_reward": fov_reward.mean().item(),
            "mobile_distance_reward": mobile_distance_reward.mean().item(),
            "xy_alignment_reward": xy_alignment_reward.mean().item(),
            "z_reward": z_reward.mean().item(),
            "final_reward": final_reward.mean().item(),
            "rot_reward": rot_reward.mean().item(),
            "action_penalty": -action_penalty.mean().item(),
        }
        return rewards

    def _compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        robot_local_grasp_rot,
        robot_local_grasp_pos,
        valve_rot,
        valve_pos,
        valve_local_grasp_rot,
        valve_local_grasp_pos,
    ):
        """
        로봇 손(Grasp 포인트)과 밸브(핸들) 각각의 로컬 그립 포즈를 월드 좌표계로 변환합니다.
        """
        robot_global_grasp_rot, robot_global_grasp_pos = tf_combine(
            hand_rot, hand_pos, robot_local_grasp_rot, robot_local_grasp_pos
        )
        valve_global_grasp_rot, valve_global_grasp_pos = tf_combine(
            valve_rot, valve_pos, valve_local_grasp_rot, valve_local_grasp_pos
        )
        return robot_global_grasp_rot, robot_global_grasp_pos, valve_global_grasp_rot, valve_global_grasp_pos

