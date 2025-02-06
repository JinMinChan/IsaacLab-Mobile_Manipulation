# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom ,Gf
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
from omni.isaac.lab.sensors import Camera, CameraCfg
import omni.isaac.core.utils.prims as prim_utils
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
import cv2
@configclass
class CameraFrankaEnvCfg(DirectRLEnvCfg):
    # 환경 설정
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    action_space = 12  # Ridgeback(3) + Franka(9)
    observation_space = 32  # 예시로 추가된 관측 범위
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

    # 로봇 설정
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/vision/Downloads/floating_franka_original.usd",  # FLOATING_FRANKA USD 경로
            activate_contact_sensors=False,  # FLOATING_FRANKA_CFG의 설정 반영
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
                # Franka Arm
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.747,  # 허용 범위 내 값
                "panda_joint5": 0.0,
                "panda_joint6": 2.0,
                "panda_joint7": 0.741,
                # 그립퍼
                "panda_finger_joint1": 0.02,
                "panda_finger_joint2": 0.02,
            },
            pos=(0.0, 0.0, 0.0),  # 모바일 플랫폼 초기 위치
            rot=(0.0, 0.0, 0.0, 0.0),  # 초기 회전 (Quaternion)
        ),

        actuators={
            # 모바일 베이스
            "base": ImplicitActuatorCfg(
                joint_names_expr=["base_joint_.*"],  # FLOATING_FRANKA_CFG에 맞춤
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

    camera = CameraCfg(
        data_types=["rgb"],
        prim_path="/World/envs/env_.*/Robot/panda_hand/Camera",  # 동적 경로 설정
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
            pos=(0.05, 0.0, 0.0),
            rot=(0.0, 0.7071, 0.0, 0.7071),
            convention="world",
        ),
    )

    # valve 설정
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



    action_scale = 2.0              # 행동 크기를 줄여 더 섬세한 동작을 유도
    dof_velocity_scale = 0.1       # 속도 스케일도 감소시켜 안정적인 움직임 촉진

    # reward scales
    dist_reward_scale = 3
    rot_reward_scale = 5
    action_penalty_scale = 0.1
    finger_reward_scale = 2.0
    # 보상 및 패널티 스케일 추가
    base_reward_scale = 1.0  # 모바일 베이스가 목표에 가까워질 때 보상 크기
    base_penalty_scale = 0.1  # 모바일 베이스의 행동에 대한 패널티 크기


class CameraFrankaEnv(DirectRLEnv):
    cfg: CameraFrankaEnvCfg

    def __init__(self, cfg: CameraFrankaEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
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

        # 로봇 초기화
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        
        # 클래스 변수로 탐지 결과 저장 리스트 추가
        self.valve_detected_results = [False] * self.cfg.scene.num_envs
        # 전역 변수로 이전 좌표 초기화 (최초 값은 0)
        self.prev_robot_dof_targets = torch.zeros((self.cfg.scene.num_envs, 2), device=self.device)

        # 타임스텝 초기화
        self.timestep = 0  # 타임스텝 변수 추가
        # 이미지 저장 경로 설정
        self.yolo_model = YOLO('/home/vision/Downloads/minchan_yolo_320/train_franka2/weights/best.pt', verbose=False)
        
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

        self.base_link_idx = self._robot.find_bodies("robot_base_link")[0][0]

        # 로봇 및 손가락 위치 가져오기
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

        # 손가락의 그립 위치 계산
        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        robot_local_pose_pos += torch.tensor([0, 0.0, 0.0225], device=self.device)  # 오프셋 추가
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        # 밸브의 기준 그립 위치
        valve_local_grasp_pose = torch.tensor(
            [0.1150, 0.15, 0.0751, 0, 0, -0.707, -0.707], device=self.device
        )  # 기본 값, 실제 환경에 맞게 수정 필요
        self.valve_local_grasp_pos = valve_local_grasp_pose[0:3].repeat((self.num_envs, 1))
        self.valve_local_grasp_rot = valve_local_grasp_pose[3:7].repeat((self.num_envs, 1))

        # 축 설정
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

        # 링크 ID 초기화
        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        self.valve_link_idx = self._valve.find_bodies("valve_handle")[0][0]

        # 초기값 설정
        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.valve_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.valve_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)


    def save_camera_image(self):
        """카메라로부터 최신 RGB 이미지 저장."""
        # 최신 데이터 요청
        self.camera.update(self.dt)  # 카메라 데이터 업데이트

        # 최신 데이터 저장
        if self.camera.data.output["rgb"] is not None:
            rgb_image = self.camera.data.output["rgb"][0, ..., :3]
            filename = os.path.join(self.output_dir, "rgb", f"{self.timestep:04d}.jpg")
            img = rgb_image.detach().cpu().numpy()
            plt.imshow(img)
            plt.axis("off")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename)
            plt.close()

    def detect_valve_with_yolo(self, env_id: int):
        """환경별 YOLO로 밸브 탐지."""
        # 카메라 이미지 업데이트
        self.camera.update(self.dt * 12)
        if self.camera.data.output["rgb"] is not None:
            # 환경별 이미지 가져오기
            rgb_image = self.camera.data.output["rgb"][env_id, ..., :3].cpu().numpy()
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            # YOLO 예측 (conf > 0.8 설정)
            results = self.yolo_model.predict(bgr_image, conf=0.8, verbose=False)
            for box in results[0].boxes.data:
                class_id = int(box[-1])
                class_name = self.yolo_model.names[class_id]
                confidence = box[-2]  # 신뢰도 값 추출

                if class_name == "valve" and confidence >= 0.8:  # 신뢰도 조건 추가
                    # 바운딩 박스 그리기 및 저장
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(bgr_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    save_path = f"/home/vision/Downloads/isaaclab140/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/camera_franka/output/detect/env_{env_id}.jpg"
                    #cv2.imwrite(save_path, bgr_image)
                    return True
        return False






    # 씬 설정
    def _setup_scene(self):
        #self.cfg.valve.init_state.pos = (0.05, 0, 0.0)  # 위치 고정
        #self.cfg.valve.init_state.rot = (0.0, 0, 0.0, 0)  # x축을 기준으로 90도 회전

        # 로봇과 Valve 아티큘레이션 설정
        self._robot = Articulation(self.cfg.robot)
        self._valve = Articulation(self.cfg.valve)

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["valve"] = self._valve

        # 지형 설정
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # 환경 복제 및 물리 속성 필터링
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # 조명 추가
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # 씬 설정 후 카메라 초기화
        super()._setup_scene()
        self.camera = Camera(self.cfg.camera)  # 카메라 초기화

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        
        # DOF 타겟 계산 (속도 스케일 적용)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        
        # DOF 제한 적용
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        #self.robot_dof_targets[:, 0] = 0.0 
        #self.robot_dof_targets[:, 1] = 0.0 
        self.robot_dof_targets[:, 2] = 0.0 


        self.timestep += 1  # 타임스텝 증가
        # 이미지 저장 주기 제어 (1 step마다 저장)
        #if self.timestep % self.cfg.decimation == 0:  # decimation 값에 따라 실제 스텝마다 저장
        #    self.save_camera_image()

        # YOLO 판단 추가 (1 step마다 판단)
        #if self.timestep % self.cfg.decimation == 0:
        #    valve_detected = self.detect_valve_with_yolo()
        #    print("Valve Detected:" if valve_detected else "Valve Not Detected")

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 최대 시뮬레이션 시간 조건 (truncated)
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        # 성공 조건: 목표 도달
        d = torch.norm(self.robot_grasp_pos - self.valve_grasp_pos, p=2, dim=-1)
        close_condition = d <= 0.025

        # 손가락 닫힘 조건: 손가락 간 거리가 충분히 작은 경우
        finger_distance = torch.norm(
            self._robot.data.body_pos_w[:, self.left_finger_link_idx] -
            self._robot.data.body_pos_w[:, self.right_finger_link_idx], p=2, dim=-1
        )
        finger_close_condition = finger_distance < 0.01

        # 성공 조건
        success_condition = close_condition & finger_close_condition

        # Mobile 종료 조건: Mobile과 Valve가 너무 멀어진 경우
        mobile_xy = self._robot.data.body_pos_w[:, self.base_link_idx][:, :2]  # Mobile의 x, y 좌표
        valve_xy = self._valve.data.body_pos_w[:, self.valve_link_idx][:, :2]  # Valve의 x, y 좌표
        mobile_dist = torch.norm(mobile_xy - valve_xy, p=2, dim=-1)
        mobile_far_condition = mobile_dist > 5.0

        # 종료 조건 설정 (truncated를 포함하지 않음)
        done = success_condition | mobile_far_condition

        return done, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        # 왼손가락과 오른손가락 위치 가져오기
        robot_left_finger_pos = self._robot.data.body_pos_w[:, self.left_finger_link_idx]
        robot_right_finger_pos = self._robot.data.body_pos_w[:, self.right_finger_link_idx]

        # 모바일 베이스와 밸브 위치 가져오기
        robot_base_pos = self._robot.data.body_pos_w[:, self.base_link_idx]
        valve_base_pos = self._valve.data.body_pos_w[:, self.valve_link_idx]

        # valve의 현재 회전값 가져오기 (예시)
        valve_rotation = self._valve.data.joint_pos[:, 0]  # Valve의 첫 번째 조인트 회전 값을 가져옴

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
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.finger_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.base_reward_scale,             # 모바일 베이스 보상 스케일
            self.cfg.base_penalty_scale,             # 모바일 베이스 패널티 스케일
            self.cfg.base_reward_scale,  # 여기에 base_penalty_scale도 추가 필요        
        )


    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        # ---------------------------
        # 1) 로봇 (기존 초기화 로직)
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
        # 3) 밸브 루트 상태 (N,13) 텐서
        # ---------------------------
        pos = self._valve.data.root_pos_w.clone()       # (N,3)
        quat = self._valve.data.root_quat_w.clone()     # (N,4)
        lin_vel = self._valve.data.root_lin_vel_w.clone()  # (N,3)
        ang_vel = self._valve.data.root_ang_vel_w.clone()  # (N,3)
        root_state = torch.cat([pos, quat, lin_vel, ang_vel], dim=-1)  # (N,13)

        # ---------------------------
        # 4) 각 환경별 로컬 랜덤 위치
        #    x= 2~4,   y= -2~2
        # ---------------------------
        random_x = sample_uniform(2.0, 4.0, (len(env_ids), 1), device=self.device)
        random_y = sample_uniform(-2.0, 2.0, (len(env_ids), 1), device=self.device)

        # shape (len(env_ids), 3)
        new_valve_pos_local = torch.zeros((len(env_ids), 3), device=self.device)
        new_valve_pos_local[:, 0] = random_x[:, 0]
        new_valve_pos_local[:, 1] = random_y[:, 0]
        new_valve_pos_local[:, 2] = 0.5

        # ---------------------------
        # 5) 환경별 오프셋(env_origins)
        #    shape: (num_envs,3)
        # ---------------------------
        # env_origins[env_id] => 해당 환경의 (x0, y0, z0) 오프셋
        # env_ids에 해당하는 오프셋만 추출
        origins = self.scene.env_origins[env_ids]  # (len(env_ids),3)

        # 월드 좌표 = 로컬 랜덤 + 오프셋
        new_valve_pos_world = new_valve_pos_local + origins

        # ---------------------------
        # 6) 회전 / 속도 = 0
        # ---------------------------
        new_valve_quat = torch.zeros((len(env_ids), 4), device=self.device)
        new_valve_quat[:, 0] = 1.0  # qw=1, qx=qy=qz=0

        # root_state: [px,py,pz, qw,qx,qy,qz, vx,vy,vz, wx,wy,wz]
        root_state[env_ids, 0:3] = new_valve_pos_world
        root_state[env_ids, 3:7] = new_valve_quat
        root_state[env_ids, 7:10] = 0.0
        root_state[env_ids, 10:13] = 0.0

        # ---------------------------
        # 7) 텔레포트
        # ---------------------------
        self._valve.write_root_state_to_sim(root_state)

        # 나머지 내부 상태 정리
        for env_id in env_ids:
            self.valve_detected_results[env_id] = False

        self._compute_intermediate_values(env_ids)
        #print(new_valve_pos_world)





    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        to_target = self.valve_grasp_pos - self.robot_grasp_pos

        # Mobile의 x, y 좌표
        mobile_xy = self._robot.data.body_pos_w[:, self.base_link_idx][:, :2]

        # Mobile과 Valve 간 거리
        valve_xy = self._valve.data.body_pos_w[:, self.valve_link_idx][:, :2]
        mobile_dist = torch.norm(mobile_xy - valve_xy, p=2, dim=-1).unsqueeze(-1)

        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                to_target,
                self._valve.data.joint_pos[:, 0].unsqueeze(-1),
                self._valve.data.joint_vel[:, 0].unsqueeze(-1),
                mobile_xy,  # Mobile의 위치
                mobile_dist,  # Mobile과 Valve 간 거리
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}


    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        # 손 위치 및 회전 업데이트
        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]

        # 밸브의 위치 및 회전 업데이트
        valve_pos = self._valve.data.body_pos_w[env_ids, self.valve_link_idx]
        valve_rot = self._valve.data.body_quat_w[env_ids, self.valve_link_idx]

        # Grasp 위치와 회전 계산
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

        # 벡터 정규화
        look_at_vector = look_at_vector / torch.norm(look_at_vector, p=2, dim=-1, keepdim=True)

        # axis4 업데이트
        self.axis4 = look_at_vector

        # Mobile과 Valve의 거리 계산
        robot_base_pos = self._robot.data.body_pos_w[:, self.base_link_idx]
        valve_base_pos = self._valve.data.body_pos_w[:, self.valve_link_idx]
        mobile_dist = torch.norm(robot_base_pos[:, :2] - valve_base_pos[:, :2], p=2, dim=-1)

        # 조건에 따른 Franka DOF 업데이트
        mask = (mobile_dist < 0.75) | (mobile_dist > 0.85)
        self.robot_dof_targets[mask, 3:7] = 0.0  # Franka의 DOF를 0으로 설정

        # Z 축 차이 계산
        z_diff = torch.abs(self.robot_grasp_pos[:, 2] - self.valve_grasp_pos[:, 2])

        # 조건에 따른 핑거 DOF 업데이트
        finger_mask_open = z_diff > 0.01
        finger_mask_close = z_diff <= 0.01

        self.robot_dof_targets[finger_mask_open, 10:12] = 0.04  # 핑거 벌림
        self.robot_dof_targets[finger_mask_close, 10:12] = 0.0  # 핑거 닫힘
        
        #print(f"Valve Detected: {self.valve_detected_results}")

        # 조건문 추가 - 환경별 valve_detected_results에 따른 업데이트
        detected = torch.tensor(self.valve_detected_results, device=self.device)
        not_detected = ~detected  # 탐지 실패 여부

        # 탐지되지 않은 환경은 이전 값을 유지
        self.robot_dof_targets[not_detected, 0] = self.prev_robot_dof_targets[not_detected, 0]
        self.robot_dof_targets[not_detected, 1] = self.prev_robot_dof_targets[not_detected, 1]

        # 탐지된 환경은 값 업데이트
        self.prev_robot_dof_targets[detected, 0] = self.robot_dof_targets[detected, 0]
        self.prev_robot_dof_targets[detected, 1] = self.robot_dof_targets[detected, 1]

        #print(self.robot_dof_targets)
        # 저장된 값 프린트
        #print(f"X: {self.prev_robot_dof_targets[:, 0].cpu().numpy()}, Y: {self.prev_robot_dof_targets[:, 1].cpu().numpy()}")





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
 
        # YOLO 탐지 주기 설정 (12 타임스텝마다 탐지)
        DETECTION_INTERVAL = 12

        # YOLO를 통해 환경별 밸브 탐지 여부 확인
        if self.timestep % DETECTION_INTERVAL == 0:
            for env_id in range(num_envs):
                detected = self.detect_valve_with_yolo(env_id)
                self.valve_detected_results[env_id] = detected  # 이전 결과 유지

        # 탐지 결과 가져오기
        valve_detected = torch.tensor(self.valve_detected_results, device=self.device)

        # ** Mobile 보상 계산 **
        mobile_xy = robot_base_pos[:, :2]
        valve_xy = valve_base_pos[:, :2]
        mobile_dist = torch.norm(mobile_xy - valve_xy, p=2, dim=-1)
        is_mobile_phase = (mobile_dist < 0.75) | (mobile_dist > 0.85)

        target_dist = 0.8  # 목표 거리
        max_reward = 5.0  # 최대 보상
        reward_slope = max_reward / 0.05  # 양수 보상 감소율
        penalty_slope = -2.0  # 구간 밖에서의 음수 비례 보상 감소율

        mobile_distance_reward = torch.where(
            (mobile_dist >= 0.75) & (mobile_dist <= 0.85),
            max_reward - reward_slope * torch.abs(mobile_dist - target_dist),  # 양수 보상
            penalty_slope * torch.abs(mobile_dist - target_dist),  # 음수 비례 보상
        )

        # YOLO 탐지 보상 계산
        yolo_reward = torch.tensor(
            [1.0 if detected or not is_mobile else -1.0 for detected, is_mobile in zip(valve_detected, is_mobile_phase)], device=self.device
        )

        # ** Franka 보상 계산 **
        xy_distance = torch.norm(
            robot_grasp_pos[:, :2] - valve_grasp_pos[:, :2], p=2, dim=-1
        )
        xy_alignment_reward = 10.0 * torch.exp(-3.0 * xy_distance) * (~is_mobile_phase).float()

        xy_alignment_penalty = torch.where(
            (xy_distance > 0.08) & (~is_mobile_phase),
            -5.0,
            torch.tensor(0.0, device=self.device),
        )

        is_xy_aligned = (xy_distance < 0.05) & (~is_mobile_phase)
        z_diff = torch.abs(robot_grasp_pos[:, 2] - valve_grasp_pos[:, 2])

        z_reward = torch.where(
            is_xy_aligned,
            10.0 * torch.exp(-3.0 * z_diff),
            torch.tensor(0.0, device=self.device),
        )


        z_penalty = torch.where(
            is_xy_aligned & (z_diff > 0.025),
            -10.0, 
            torch.tensor(0.0, device=self.device),
        )

        axis1 = tf_vector(robot_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(valve_grasp_rot, valve_inward_axis)
        axis3 = tf_vector(robot_grasp_rot, gripper_up_axis)
        axis4 = self.axis4

        dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)

        rot_reward = 5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2) * (~is_mobile_phase).float()


        is_fully_aligned = is_xy_aligned & (z_diff < 0.025)
        final_reward = torch.where(
            is_fully_aligned,
            100.0*rot_reward/10,
            torch.tensor(0.0, device=self.device),
        )


        # 행동 패널티 계산
        mobile_action_penalty = torch.sum(actions[:, :2]**2, dim=-1) / actions[:, :2].shape[-1]
        franka_action_penalty = torch.sum(actions[:, 3:10]**2, dim=-1) / actions[:, 3:10].shape[-1]

        action_penalty = mobile_action_penalty + franka_action_penalty

        # ** 총 보상 계산 **
        rewards = (
            + yolo_reward  # YOLO 탐지 보상
            + mobile_distance_reward  # Mobile 보상
            + xy_alignment_reward  # XY 정렬 보상
            + xy_alignment_penalty  # XY 정렬 패널티
            + z_reward  # Z 축 보상
            #+ z_penalty  # Z 축 패널티
            + final_reward  # 최종 도달 보상
            + rot_reward  # 축 정렬 보상
            - action_penalty  # 행동 패널티
        )

        # 로그 정보 업데이트
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
        #print("yolo_reward:", yolo_reward.cpu().numpy())
        #print("mobile_dist:", mobile_dist.cpu().numpy())
        #print(torch.stack([valve_detected, is_mobile_phase, is_xy_aligned], dim=1).cpu().numpy())

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
        robot_global_grasp_rot, robot_global_grasp_pos = tf_combine(
            hand_rot, hand_pos, robot_local_grasp_rot, robot_local_grasp_pos
        )
        valve_global_grasp_rot, valve_global_grasp_pos = tf_combine(
            valve_rot, valve_pos, valve_local_grasp_rot, valve_local_grasp_pos
        )
        #print("valve_rot:", valve_rot[0].cpu().numpy())
        #print("valve_pos:", valve_pos[0].cpu().numpy())

        #print("valve_rot:", valve_rot[0].cpu().numpy())
        #print("valve_pos:", valve_pos[0].cpu().numpy())
        return robot_global_grasp_rot, robot_global_grasp_pos, valve_global_grasp_rot, valve_global_grasp_pos
        