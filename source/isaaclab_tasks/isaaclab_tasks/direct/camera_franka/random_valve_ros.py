# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import os
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge
from custom_message.msg import MultiImage

from pxr import UsdGeom
import matplotlib.pyplot as plt
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
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector


@configclass
class CameraFrankaEnvCfg(DirectRLEnvCfg):
    """
    카메라 + 모바일 베이스(Ridgeback) + 프랑카 로봇 암 환경 설정.
    기본 시뮬레이션 파라미터, 로봇/밸브/카메라 속성 등을 정의합니다.
    """

    # 환경
    episode_length_s = 8.3333
    decimation = 2
    action_space = 12
    observation_space = 32
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

    # 카메라
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

    # 밸브(Valve)
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

    # 바닥(plane)
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


class ROSInterface(Node):
    """
    ROS2 인터페이스 노드:
      - 카메라 이미지(MultiImage) 퍼블리시
      - 외부 노드가 보낸 밸브 탐지 결과(Int32MultiArray) 구독
    """
    def __init__(self, num_envs: int):
        super().__init__('simulation_ros_interface')
        self.num_envs = num_envs
        self.multi_image_pub = self.create_publisher(MultiImage, 'camera/multi_image', 10)
        self.detection_sub = self.create_subscription(
            Int32MultiArray,
            'valve/detections',
            self.detection_callback,
            10
        )
        self.latest_detections = [False] * self.num_envs
        self.bridge = CvBridge()

    def detection_callback(self, msg: Int32MultiArray):
        """
        valve/detections 토픽 구독 콜백.
        각 환경 env_id별로 1(탐지 성공)/0(미탐지) 정보를 받아서
        self.latest_detections 업데이트.
        """
        self.latest_detections = [bool(x) for x in msg.data]

    def publish_images(self, images: list[Image]):
        """
        다수의 sensor_msgs/Image를 하나의 MultiImage 메시지로 묶어 퍼블리시.
        """
        multi_image_msg = MultiImage()
        multi_image_msg.header.stamp = self.get_clock().now().to_msg()
        multi_image_msg.images = images
        self.multi_image_pub.publish(multi_image_msg)


class CameraFrankaEnv(DirectRLEnv):
    """
    모바일 베이스 + 프랑카 로봇 + 카메라를 사용해
    밸브를 찾아 제어하는 Isaac Sim 환경.
    ROS2를 통해 외부 YOLO 탐지 결과를 수신하여 밸브 위치를 인식합니다.
    """

    cfg: CameraFrankaEnvCfg

    def __init__(self, cfg: CameraFrankaEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """
            주어진 xformable(USD Prim)의 월드 좌표를,
            현재 환경(env_pos) 기준 로컬 좌표로 변환하여 반환.
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

        # 로봇 DOF 초기화
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        # 밸브 탐지 결과(환경별) 플래그
        self.valve_detected_results = [False] * self.cfg.scene.num_envs

        # 모바일 베이스 xy 기록 텐서
        self.prev_robot_dof_targets = torch.zeros((self.cfg.scene.num_envs, 2), device=self.device)

        # 타임스텝
        self.timestep = 0

        # ROS2 인터페이스 초기화 (노드, 퍼블리셔/서브스크라이버)
        if not rclpy.ok():
            rclpy.init()
        self.ros_interface = ROSInterface(self.cfg.scene.num_envs)

        # 모바일 베이스 / 그립퍼 조인트 속도 스케일 설정
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

        # USD Stage에서 실제 손/손가락 위치를 가져와 그립(Grasp) 포즈 계산
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

        # 손가락 중심 -> 그립 포즈
        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])
        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        robot_local_pose_pos += torch.tensor([0, 0.0, 0.0225], device=self.device)

        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        # 밸브(Valve) 그립 포즈 (오프셋)
        valve_local_grasp_pose = torch.tensor([0.1150, 0.15, 0.0751, 0, 0, -0.707, -0.707], device=self.device)
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

        # 링크 인덱스
        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        self.valve_link_idx = self._valve.find_bodies("valve_handle")[0][0]

        # 로봇/밸브 그립 포즈 저장용 텐서
        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.valve_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.valve_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

    def publish_ros_camera_images(self):
        """
        Isaac Sim의 카메라 데이터를 ROS2로 퍼블리시(MultiImage).
        외부 YOLO 노드가 이를 구독해 밸브 탐지 후 결과를 다시 보내줌.
        """
        if self.camera.data.output["rgb"] is not None:
            rgb_all = self.camera.data.output["rgb"]  # (num_envs, H, W, 4)
            images = []
            for env in range(self.num_envs):
                img_tensor = rgb_all[env, ..., :3]
                img_np = img_tensor.detach().cpu().numpy()
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype('uint8')
                else:
                    img_np = img_np.astype('uint8')
                img_msg = self.ros_interface.bridge.cv2_to_imgmsg(img_np, encoding="rgb8")
                images.append(img_msg)
            self.ros_interface.publish_images(images)

    def _setup_scene(self):
        """
        로봇, 밸브, 지형(plane) 등을 씬에 추가하고,
        복제(clone) 후 카메라를 초기화합니다.
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

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        super()._setup_scene()
        self.camera = Camera(self.cfg.camera)

    def _pre_physics_step(self, actions: torch.Tensor):
        """
        매 타임스텝 물리 시뮬레이션 전:
          - 액션(로봇 DOF 타겟) 적용
          - ROS2 spin으로 밸브 탐지 결과 구독
          - 일정 주기로 카메라 이미지 퍼블리시
        """
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(
            targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits
        )
        self.robot_dof_targets[:, 2] = 0.0  # 모바일 베이스 z축 고정
        self.timestep += 1

        # ROS2 콜백 spin
        rclpy.spin_once(self.ros_interface, timeout_sec=0)
        self.valve_detected_results = self.ros_interface.latest_detections

        # 주기적으로 카메라 이미지를 ROS2 퍼블리시 -> 외부 YOLO 검출
        DETECTION_INTERVAL = 12
        if self.timestep % DETECTION_INTERVAL == 0:
            self.camera.update(self.dt * 12)
            self.publish_ros_camera_images()

    def _apply_action(self):
        """
        _pre_physics_step에서 계산된 self.robot_dof_targets를 시뮬레이터에 반영.
        """
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        에피소드 종료 / 트렁케이트 판정.
        """
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        d = torch.norm(self.robot_grasp_pos - self.valve_grasp_pos, p=2, dim=-1)
        close_condition = d <= 0.025

        finger_distance = torch.norm(
            self._robot.data.body_pos_w[:, self.left_finger_link_idx]
            - self._robot.data.body_pos_w[:, self.right_finger_link_idx],
            p=2,
            dim=-1,
        )
        finger_close_condition = finger_distance < 0.01
        success_condition = close_condition & finger_close_condition

        robot_xy = self._robot.data.body_pos_w[:, self.base_link_idx][:, :2]
        valve_xy = self._valve.data.body_pos_w[:, self.valve_link_idx][:, :2]
        mobile_dist = torch.norm(robot_xy - valve_xy, p=2, dim=-1)
        mobile_far_condition = mobile_dist > 5.0

        done = success_condition | mobile_far_condition
        return done, truncated

    def _get_rewards(self) -> torch.Tensor:
        """
        보상 계산: intermediate values -> _compute_rewards().
        """
        self._compute_intermediate_values()
        left_finger_pos = self._robot.data.body_pos_w[:, self.left_finger_link_idx]
        right_finger_pos = self._robot.data.body_pos_w[:, self.right_finger_link_idx]
        base_pos = self._robot.data.body_pos_w[:, self.base_link_idx]
        valve_pos = self._valve.data.body_pos_w[:, self.valve_link_idx]

        return self._compute_rewards(
            self.actions,
            self._valve.data.joint_pos,
            self.robot_grasp_pos,
            self.valve_grasp_pos,
            self.robot_grasp_rot,
            self.valve_grasp_rot,
            left_finger_pos,
            right_finger_pos,
            self.gripper_forward_axis,
            self.valve_inward_axis,
            self.gripper_up_axis,
            self.valve_up_axis,
            base_pos,
            valve_pos,
            self.num_envs,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.finger_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.base_reward_scale,
            self.cfg.base_penalty_scale,
            # 호출부에서 인자를 하나 더 넘기고 있지만, 필요 시 정리 또는
            # _compute_rewards 시그니처 변경이 필요
            self.cfg.base_reward_scale,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """
        지정된 env_ids의 로봇/밸브 초기화.
        """
        super()._reset_idx(env_ids)

        base_joint_indices = self._robot.find_joints("base_joint.*")[0]
        base_joint_indices = torch.tensor(base_joint_indices, dtype=torch.long, device=self.device)
        init_positions = torch.zeros((len(env_ids), 3), device=self.device)
        self.robot_dof_targets[env_ids, :3] = init_positions

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

        zeros = torch.zeros((len(env_ids), self._valve.num_joints), device=self.device)
        self._valve.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)

        pos = self._valve.data.root_pos_w.clone()
        quat = self._valve.data.root_quat_w.clone()
        lin_vel = self._valve.data.root_lin_vel_w.clone()
        ang_vel = self._valve.data.root_ang_vel_w.clone()
        root_state = torch.cat([pos, quat, lin_vel, ang_vel], dim=-1)

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
        관측(obs) 값 구성: 로봇 관절, 속도, 목표(밸브) 위치, 모바일 베이스 등.
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
        보상/종료 계산용 중간값(로봇/밸브 포즈, 축 정렬 등) 업데이트.
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
        로봇 손(Grasp)과 밸브(Handle) 각각의 로컬 포즈를
        월드 좌표계로 변환하여 반환.
        """
        robot_global_grasp_rot, robot_global_grasp_pos = tf_combine(
            hand_rot, hand_pos, robot_local_grasp_rot, robot_local_grasp_pos
        )
        valve_global_grasp_rot, valve_global_grasp_pos = tf_combine(
            valve_rot, valve_pos, valve_local_grasp_rot, valve_local_grasp_pos
        )
        return robot_global_grasp_rot, robot_global_grasp_pos, valve_global_grasp_rot, valve_global_grasp_pos

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
        (외부 YOLO 결과 + 로봇/밸브 포즈) 기반 보상 계산.
        - ROS2로부터 self.valve_detected_results 업데이트
        - 모바일 베이스 접근
        - Franka 암 정렬
        - 행동(액션) 패널티
        """
        # valve_detected_results를 텐서로 변환
        valve_detected_tensor = torch.tensor(self.valve_detected_results, device=self.device, dtype=torch.float32)

        # 모바일 베이스 보상
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

        # YOLO (ROS2) 탐지 보상
        yolo_reward = []
        for env_id in range(num_envs):
            if valve_detected_tensor[env_id] or (not is_mobile_phase[env_id]):
                yolo_reward.append(1.0)
            else:
                yolo_reward.append(-1.0)
        yolo_reward = torch.tensor(yolo_reward, device=self.device)

        # Franka 보상
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

        # 행동 패널티
        mobile_action_penalty = torch.sum(actions[:, :2] ** 2, dim=-1) / actions[:, :2].shape[-1]
        franka_action_penalty = torch.sum(actions[:, 3:10] ** 2, dim=-1) / actions[:, 3:10].shape[-1]
        action_penalty = mobile_action_penalty + franka_action_penalty

        # 종합 보상
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

