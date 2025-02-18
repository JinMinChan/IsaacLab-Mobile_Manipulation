# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import cv2
import torch
import matplotlib.pyplot as plt

from pxr import UsdGeom
from ultralytics import YOLO

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import (
    tf_combine,
    tf_inverse,
    tf_vector,
)


@configclass
class CameraFrankaEnvCfg(DirectRLEnvCfg):
    """
    카메라를 사용하는 모바일 베이스(Ridgeback) + 프랑카(Franka) 로봇 암 환경 설정 클래스입니다.
    episode_length_s, decimation, action_space, observation_space 등의
    환경 파라미터와 로봇/밸브/카메라 초기 설정을 포함합니다.
    """

    episode_length_s = 8.3333
    decimation = 2
    action_space = 12
    observation_space = 32
    state_space = 0

    # 시뮬레이션 설정
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

    # 씬 설정
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=100.0, replicate_physics=True)

    # 로봇(Floating Franka + Mobile Base)
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
                "base_joint_x": 0.0,
                "base_joint_y": 0.0,
                "base_joint_z": 0.0,
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.747,
                "panda_joint5": 0.0,
                "panda_joint6": 2.0,
                "panda_joint7": 0.741,
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
                stiffness=1e5,
                damping=1e3,
            ),
        },
    )

    # 카메라 설정
    camera = CameraCfg(
        data_types=["rgb"],
        prim_path="/World/envs/env_.*/Robot/summit_xl_top_structure/Camera",
        height=320,
        width=320,
        update_period=0.1,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.235, 0.0, 1.5),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
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
            joint_pos={"valve_handle_joint": 0.0},
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

    # 바닥 평면 설정
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

    # 동작 스케일
    action_scale = 2.0
    dof_velocity_scale = 0.1

    # 보상 스케일
    dist_reward_scale = 3
    rot_reward_scale = 5
    action_penalty_scale = 0.1
    finger_reward_scale = 2.0
    base_reward_scale = 1.0
    base_penalty_scale = 0.1


class CameraFrankaEnv(DirectRLEnv):
    """
    카메라를 장착한 모바일 베이스 + 프랑카 로봇 팔이 밸브(Valve)를 찾고 조작하는 환경 클래스입니다.
    Isaac Sim의 DirectRLEnv를 상속하여, RL 훈련에 필요한 reset, step, reward 계산 등을 구현합니다.
    """

    cfg: CameraFrankaEnvCfg

    def __init__(self, cfg: CameraFrankaEnvCfg, render_mode: str | None = None, **kwargs):
        """
        환경 초기화:
          - 로봇/밸브 아티큘레이션, 카메라 설정
          - 조인트 제한 및 속도 스케일
          - 그립(Grasp) 포즈 계산
          - YOLO 모델 로딩
          - 밸브 검출 여부 플래그 초기화
        """
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """
            주어진 프림(xformable)의 월드 변환을, 현재 환경(env_pos) 기준 로컬 좌표로 변환해
            [px, py, pz, qw, qx, qy, qz] 형태의 텐서로 반환합니다.
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

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # 로봇 관절 제한, 속도 스케일 초기화
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        # 밸브 YOLO 탐지 여부 (환경별)
        self.valve_detected_results = [False] * self.cfg.scene.num_envs

        # 모바일 베이스 xy 기록 텐서 (추가로 활용 가능)
        self.prev_robot_dof_targets = torch.zeros((self.cfg.scene.num_envs, 2), device=self.device)

        # 타임스텝
        self.timestep = 0

        # YOLO 모델 로드
        self.yolo_model = YOLO("/home/vision/Downloads/minchan_yolo_320/train_franka2/weights/best.pt", verbose=False)

        # 모바일 베이스 조인트 속도 스케일
        base_joints = self._robot.find_joints("base_joint_.*")[0]
        if isinstance(base_joints, list):
            for joint in base_joints:
                self.robot_dof_speed_scales[joint] = 1.0
        else:
            self.robot_dof_speed_scales[base_joints] = 1.0

        # 그립퍼 조인트 속도 스케일
        gripper_joints = self._robot.find_joints("panda_finger_joint.*")[0]
        if isinstance(gripper_joints, list):
            for joint in gripper_joints:
                self.robot_dof_speed_scales[joint] = 0.1
        else:
            self.robot_dof_speed_scales[gripper_joints] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.base_link_idx = self._robot.find_bodies("robot_base_link")[0][0]

        # 초기 손, 손가락의 실제 위치를 가져와 "로컬 그립(Grasp) 포즈" 계산
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

        # 손가락 사이 중앙값으로 그립 포즈 설정
        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]  # 왼손가락의 회전 값을 사용

        # "손"과 "손가락 중앙" 사이 변환
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])
        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        # 그립 포인트 오프셋
        robot_local_pose_pos += torch.tensor([0, 0.0, 0.0225], device=self.device)

        # (N,3) / (N,4) 형태로 생성
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        # 밸브를 잡기 위한 핸들 로컬 포즈 (실험적)
        valve_local_grasp_pose = torch.tensor([0.1150, 0.15, 0.0751, 0, 0, -0.707, -0.707], device=self.device)
        self.valve_local_grasp_pos = valve_local_grasp_pose[0:3].repeat((self.num_envs, 1))
        self.valve_local_grasp_rot = valve_local_grasp_pose[3:7].repeat((self.num_envs, 1))

        # 로봇/밸브 축 설정 (그립퍼 전방, 밸브 내부 방향 등)
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

        # 링크 인덱스
        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        self.valve_link_idx = self._valve.find_bodies("valve_handle")[0][0]

        # 로봇/밸브 그립 포즈를 매 타임스텝 저장할 텐서
        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.valve_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.valve_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

    def save_camera_image(self):
        """
        카메라에서 얻은 최신 이미지를 파일로 저장합니다.
        (필요 시 외부에서 호출 가능)
        """
        self.camera.update(self.dt)
        if self.camera.data.output["rgb"] is not None:
            rgb_image = self.camera.data.output["rgb"][0, ..., :3]
            filename = os.path.join(self.output_dir, "rgb", f"{self.timestep:04d}.jpg")
            img = rgb_image.detach().cpu().numpy()
            plt.imshow(img)
            plt.axis("off")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename)
            plt.close()

    def detect_valve_with_yolo(self, env_id: int) -> bool:
        """
        특정 env_id에 대해 YOLO로 밸브 검출 시도.
        """
        self.camera.update(self.dt * 12)
        if self.camera.data.output["rgb"] is not None:
            rgb_image = self.camera.data.output["rgb"][env_id, ..., :3].cpu().numpy()
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            results = self.yolo_model.predict(bgr_image, conf=0.8, verbose=False)
            for box in results[0].boxes.data:
                class_id = int(box[-1])
                class_name = self.yolo_model.names[class_id]
                confidence = box[-2]
                if class_name == "valve" and confidence >= 0.8:
                    return True
        return False

    def _setup_scene(self):
        """
        로봇, 밸브, 지형(plane) 등을 씬에 추가한 뒤,
        부모 클래스의 _setup_scene()을 호출하고 카메라를 초기화합니다.
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

        # 조명 추가
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        super()._setup_scene()
        self.camera = Camera(self.cfg.camera)

    def _pre_physics_step(self, actions: torch.Tensor):
        """
        매 프레임 물리 시뮬레이션 전에 호출:
         - 행동(actions)을 받아 로봇 조인트 DOF 목표 업데이트
         - 불필요한 Z축 이동 제한(예시)
         - 타임스텝 증가
        """
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = (
            self.robot_dof_targets
            + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        )
        self.robot_dof_targets[:] = torch.clamp(
            targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits
        )
        # 예시로 베이스의 Z축은 0.0 고정
        self.robot_dof_targets[:, 2] = 0.0
        self.timestep += 1

    def _apply_action(self):
        """
        _pre_physics_step에서 계산한 self.robot_dof_targets를
        실제 물리 시뮬레이터(Articulation)에 적용합니다.
        """
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        에피소드 종료(done) 및 트렁케이트(truncated) 조건을 판별합니다.
        """
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        # 그립과 밸브가 가까운 경우
        d = torch.norm(self.robot_grasp_pos - self.valve_grasp_pos, p=2, dim=-1)
        close_condition = d <= 0.025

        # 손가락 간격
        finger_distance = torch.norm(
            self._robot.data.body_pos_w[:, self.left_finger_link_idx]
            - self._robot.data.body_pos_w[:, self.right_finger_link_idx],
            p=2,
            dim=-1,
        )
        finger_close_condition = finger_distance < 0.01
        success_condition = close_condition & finger_close_condition

        # 모바일 베이스가 밸브와 멀어지면 실패
        mobile_xy = self._robot.data.body_pos_w[:, self.base_link_idx][:, :2]
        valve_xy = self._valve.data.body_pos_w[:, self.valve_link_idx][:, :2]
        mobile_dist = torch.norm(mobile_xy - valve_xy, p=2, dim=-1)
        mobile_far_condition = mobile_dist > 5.0

        done = success_condition | mobile_far_condition
        return done, truncated

    def _get_rewards(self) -> torch.Tensor:
        """
        보상 함수를 계산하여 반환합니다.
        _compute_intermediate_values()로부터 필요한 중간 정보를 업데이트받은 후,
        _compute_rewards()를 호출합니다.
        """
        self._compute_intermediate_values()

        robot_left_finger_pos = self._robot.data.body_pos_w[:, self.left_finger_link_idx]
        robot_right_finger_pos = self._robot.data.body_pos_w[:, self.right_finger_link_idx]
        robot_base_pos = self._robot.data.body_pos_w[:, self.base_link_idx]
        valve_base_pos = self._valve.data.body_pos_w[:, self.valve_link_idx]

        return self._compute_rewards(
            self.actions,
            self._valve.data.joint_pos,
            self.robot_grasp_pos,
            self.valve_grasp_pos,
            self.robot_grasp_rot,
            self.valve_grasp_rot,
            robot_left_finger_pos,
            robot_right_finger_pos,
            self.gripper_forward_axis,
            self.valve_inward_axis,
            self.gripper_up_axis,
            self.valve_up_axis,
            robot_base_pos,
            valve_base_pos,
            self.num_envs,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.finger_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.base_reward_scale,
            self.cfg.base_penalty_scale,
            self.cfg.base_reward_scale,  # 호출부에서 예시로 마지막 인자를 넘기고 있음
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """
        주어진 env_ids에 대해 로봇, 밸브 등의 상태를 재설정하여
        다음 에피소드를 시작할 수 있게 합니다.
        """
        super()._reset_idx(env_ids)

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

        # 밸브 내부 조인트 초기화
        zeros = torch.zeros((len(env_ids), self._valve.num_joints), device=self.device)
        self._valve.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)

        # 밸브 루트 상태
        pos = self._valve.data.root_pos_w.clone()
        quat = self._valve.data.root_quat_w.clone()
        lin_vel = self._valve.data.root_lin_vel_w.clone()
        ang_vel = self._valve.data.root_ang_vel_w.clone()
        root_state = torch.cat([pos, quat, lin_vel, ang_vel], dim=-1)

        # 밸브를 랜덤 위치(예: x=2~4, y=-2~2)에 스폰
        random_x = sample_uniform(2.0, 4.0, (len(env_ids), 1), device=self.device)
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

        for env_id in env_ids:
            self.valve_detected_results[env_id] = False

        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict:
        """
        RL 정책에 입력될 관측값을 구성하여 반환합니다.
        관절 위치/속도, 로봇/밸브 거리 등 다양한 정보를 합칩니다.
        """
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        to_target = self.valve_grasp_pos - self.robot_grasp_pos

        mobile_xy = self._robot.data.body_pos_w[:, self.base_link_idx][:, :2]
        valve_xy = self._valve.data.body_pos_w[:, self.valve_link_idx][:, :2]
        mobile_dist = torch.norm(mobile_xy - valve_xy, p=2, dim=-1).unsqueeze(-1)

        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                to_target,
                self._valve.data.joint_pos[:, 0].unsqueeze(-1),
                self._valve.data.joint_vel[:, 0].unsqueeze(-1),
                mobile_xy,
                mobile_dist,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        """
        보상 계산과 종료 조건에 사용될 중간 정보를 업데이트합니다.
        (로봇/밸브 포즈, 축 정렬, 모바일 베이스-밸브 거리 등)
        """
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

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

        valve_grasp_pos = self.valve_grasp_pos.index_select(0, env_ids)
        look_at_vector_xy = valve_pos[:, :2] - valve_grasp_pos[:, :2]
        z_values = torch.zeros((look_at_vector_xy.shape[0], 1), device=self.device)
        look_at_vector = torch.cat((look_at_vector_xy, z_values), dim=-1)
        look_at_vector = look_at_vector / torch.norm(look_at_vector, p=2, dim=-1, keepdim=True)
        self.axis4 = look_at_vector

        # 모바일 베이스와 밸브 거리 기반으로 Franka DOF를 간단히 제어(예시)
        robot_base_pos = self._robot.data.body_pos_w[:, self.base_link_idx]
        valve_base_pos = self._valve.data.body_pos_w[:, self.valve_link_idx]
        mobile_dist = torch.norm(robot_base_pos[:, :2] - valve_base_pos[:, :2], p=2, dim=-1)
        mask = (mobile_dist < 0.75) | (mobile_dist > 0.85)

        self.robot_dof_targets[mask, 3:10] = 0.0
        self.robot_dof_targets[mask, 4:5] = 70.0
        self.robot_dof_targets[mask, 5:6] = 1.5
        self.robot_dof_targets[mask, 6:7] = -160.0
        self.robot_dof_targets[mask, 8:9] = 100.0

        z_diff = torch.abs(self.robot_grasp_pos[:, 2] - self.valve_grasp_pos[:, 2])
        finger_mask_open = z_diff > 0.01
        finger_mask_close = z_diff <= 0.01
        self.robot_dof_targets[finger_mask_open, 10:12] = 0.04
        self.robot_dof_targets[finger_mask_close, 10:12] = 0.0

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
        로봇 손과 밸브 핸들의 로컬 그립 포즈를 월드 좌표로 변환해 반환합니다.
        """
        robot_global_grasp_rot, robot_global_grasp_pos = tf_combine(
            hand_rot, hand_pos, robot_local_grasp_rot, robot_local_grasp_pos
        )
        valve_global_grasp_rot, valve_global_grasp_pos = tf_combine(
            valve_rot, valve_pos, valve_local_grasp_rot, valve_local_grasp_pos
        )
        return (
            robot_global_grasp_rot,
            robot_global_grasp_pos,
            valve_global_grasp_rot,
            valve_global_grasp_pos,
        )

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
        open_reward_scale,
        action_penalty_scale,
        finger_reward_scale,
        base_reward_scale,
        base_penalty_scale,
    ):
        """
        실제 보상 계산 메서드. 
        - YOLO 탐지 여부
        - 모바일 베이스-밸브 거리 보상
        - 프랑카 팔 정렬 및 축 회전 보상
        - 행동 패널티
        등을 종합하여 보상을 반환합니다.
        """
        DETECTION_INTERVAL = 12
        if self.timestep % DETECTION_INTERVAL == 0:
            undetected_envs = [i for i, detected in enumerate(self.valve_detected_results) if not detected]
            if len(undetected_envs) > 0:
                self.camera.update(self.dt * DETECTION_INTERVAL)
                rgb_all = self.camera.data.output["rgb"]
                if rgb_all is not None:
                    rgb_all_np = rgb_all[..., :3].cpu().numpy()
                    batch_imgs = []
                    for env_id in undetected_envs:
                        rgb_img = rgb_all_np[env_id]
                        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                        batch_imgs.append(bgr_img)
                    results = self.yolo_model.predict(batch_imgs, conf=0.8, verbose=False)
                    for i, result in enumerate(results):
                        env_id = undetected_envs[i]
                        if len(result.boxes) > 0:
                            for box in result.boxes.data:
                                class_id = int(box[-1])
                                class_name = self.yolo_model.names[class_id]
                                confidence = box[-2]
                                if class_name == "valve" and confidence >= 0.8:
                                    self.valve_detected_results[env_id] = True
                                    break

        valve_detected_tensor = torch.tensor(self.valve_detected_results, device=self.device)

        # ------------------ 모바일 베이스 보상 ------------------
        mobile_xy = robot_base_pos[:, :2]
        valve_xy = valve_base_pos[:, :2]
        mobile_dist = torch.norm(mobile_xy - valve_xy, p=2, dim=-1)
        is_mobile_phase = (mobile_dist < 0.75) | (mobile_dist > 0.85)

        target_dist = 0.8
        max_reward = 5.0
        reward_slope = max_reward / 0.05
        penalty_slope = -2.0

        mobile_distance_reward = torch.where(
            (mobile_dist >= 0.75) & (mobile_dist <= 0.85),
            max_reward - reward_slope * torch.abs(mobile_dist - target_dist),
            penalty_slope * torch.abs(mobile_dist - target_dist),
        )

        # YOLO 탐지 보상 (단순 +1 / -1)
        yolo_reward = []
        for env_id in range(num_envs):
            if valve_detected_tensor[env_id] or (not is_mobile_phase[env_id]):
                yolo_reward.append(1.0)
            else:
                yolo_reward.append(-1.0)
        yolo_reward = torch.tensor(yolo_reward, device=self.device)

        # ------------------ 프랑카 보상 ------------------
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

        # ------------------ 행동 패널티 ------------------
        mobile_action_penalty = torch.sum(actions[:, :2] ** 2, dim=-1) / actions[:, :2].shape[-1]
        franka_action_penalty = torch.sum(actions[:, 3:10] ** 2, dim=-1) / actions[:, 3:10].shape[-1]
        action_penalty = mobile_action_penalty + franka_action_penalty

        # ------------------ 최종 보상 계산 ------------------
        rewards = (
            yolo_reward
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
            "yolo_reward": yolo_reward.mean().item(),
            "mobile_distance_reward": mobile_distance_reward.mean().item(),
            "xy_alignment_reward": xy_alignment_reward.mean().item(),
            "z_reward": z_reward.mean().item(),
            "final_reward": final_reward.mean().item(),
            "rot_reward": rot_reward.mean().item(),
            "action_penalty": -action_penalty.mean().item(),
        }
        return rewards

