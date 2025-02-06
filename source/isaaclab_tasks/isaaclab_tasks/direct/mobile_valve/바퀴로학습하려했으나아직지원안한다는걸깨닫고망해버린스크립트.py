# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom ,Gf, UsdPhysics
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
    # 환경 설정
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    action_space = 12  # Ridgeback(3) + Franka(9)
    observation_space = 36  # 예시로 추가된 관측 범위
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # 로봇 설정
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/vision/Downloads/floating_franka.usd",  # FLOATING_FRANKA USD 경로
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
                # 바퀴
                "robot_back_left_wheel_joint": 0.0,
                "robot_back_right_wheel_joint": 0.0,
                "robot_front_left_wheel_joint": 0.0,
                "robot_front_right_wheel_joint": 0.0,
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
            pos=(1.6, 0.0, 0.1),  # 모바일 플랫폼 초기 위치
            rot=(0.0, 0.0, 0.0, 1.0),  # 초기 회전 (Quaternion)
        ),
        actuators={
            # 바퀴 기반 구동
            "mobile_wheels": ImplicitActuatorCfg(
                joint_names_expr=[
                    "robot_back_left_wheel_joint", 
                    "robot_back_right_wheel_joint",
                    "robot_front_left_wheel_joint", 
                    "robot_front_right_wheel_joint"
                ],
                velocity_limit=100.0,  # 속도 제한
                effort_limit=100.0,  # 힘 제한
                stiffness=0.0,
                damping=100.0,
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


    # valve 설정
    valve = ArticulationCfg(
        prim_path="/World/envs/env_.*/Valve",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/vision/Downloads/valves/round_valve/round_valve_main.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-2.0, 0, 0.8),
            rot=(0.0, 0.0, 0.0, 1.0),
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

    # 바닥 평면 설정
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


class MobileValveEnv(DirectRLEnv):
    cfg: MobileValveEnvCfg

    def __init__(self, cfg: MobileValveEnvCfg, render_mode: str | None = None, **kwargs):
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
        print("soft_joint_pos_limits shape:", self._robot.data.soft_joint_pos_limits.shape)
        print("soft_joint_pos_limits values:", self._robot.data.soft_joint_pos_limits)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        print(type(self._robot))  # self._robot 객체의 타입을 출력해 확인

        # 바퀴 링크 인덱스 초기화
        self.wheel_link_idxs = [
            self._robot.find_bodies("robot_back_left_wheel_link")[0][0],
            self._robot.find_bodies("robot_back_right_wheel_link")[0][0],
            self._robot.find_bodies("robot_front_left_wheel_link")[0][0],
            self._robot.find_bodies("robot_front_right_wheel_link")[0][0]
        ]
        print(f"Wheel Links: {self.wheel_link_idxs}")  # 바퀴 링크들 출력

        # 바퀴 조인트 인덱스 초기화
        self.wheel_joint_ids = [
            self._robot.find_joints("robot_back_left_wheel_joint")[0][0],
            self._robot.find_joints("robot_back_right_wheel_joint")[0][0],
            self._robot.find_joints("robot_front_left_wheel_joint")[0][0],
            self._robot.find_joints("robot_front_right_wheel_joint")[0][0]
        ]
        print(f"Wheel Joints: {self.wheel_joint_ids}")  # 바퀴 조인트들 출력

        if isinstance(self.wheel_joint_ids, list):
            for joint in self.wheel_joint_ids:
                self.robot_dof_speed_scales[joint] = 1.0
        else:
            self.robot_dof_speed_scales[self.wheel_joint_ids] = 1.0

        # Gripper joints 설정
        gripper_joints = self._robot.find_joints("panda_finger_joint.*")[0]
        print(f"Gripper Joints: {gripper_joints}")
        if isinstance(gripper_joints, list):
            for joint in gripper_joints:
                self.robot_dof_speed_scales[joint] = 0.1
        else:
            self.robot_dof_speed_scales[gripper_joints] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        # 로봇의 바디 링크 인덱스 초기화
        self.base_link_idx = self._robot.find_bodies("robot_base_link")[0][0]

        # 로봇 및 손가락 위치 가져오기
        stage = get_current_stage()
        physics_scene = stage.GetPrimAtPath("/World/physicsScene")
        if not physics_scene:
            print("Physics scene not found. Please ensure it exists in the USD stage.")
        else:
            contact_report = UsdPhysics.ContactReportAPI(physics_scene)
            contact_report.CreateEnableContactReportsAttr(True)
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/floating_franka/panda_link7")),
            self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/floating_franka/panda_leftfinger")),
            self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/floating_franka/panda_rightfinger")),
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
        robot_local_pose_pos += torch.tensor([0, 0.0, 0.0425], device=self.device)  # 오프셋 추가
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

        print(self._robot.root_physx_view)  # PhysX 연결 상태 확인



    # 씬 설정
    def _setup_scene(self):
        # 로봇과 Valve 아티큘레이션 설정
        self._robot = Articulation(self.cfg.robot)
        self._valve = Articulation(self.cfg.valve)

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


        # physics step 전 호출
    def _pre_physics_step(self, actions: torch.Tensor):
        # 액션 클램핑
        self.actions = actions.clone().clamp(-1.0, 1.0)

        # 모바일 베이스 (바퀴) 제어
        mobile_actions = self.actions[:, :4]  # 바퀴의 속도 제어

        # 속도 목표값을 직접 설정 (누적 제거)
        self.robot_dof_targets[:, :4] = torch.clamp(
            mobile_actions * self.cfg.action_scale, min=-100.0, max=100.0
        )
        # print 문 제거: 바퀴 속도 출력 부분 삭제

        # 바퀴 속도 타겟 설정 (속도 제어만)
        self._robot.set_joint_velocity_target(self.robot_dof_targets[:, :4], joint_ids=self.wheel_joint_ids)

        # 매니퓰레이터 제어 (위치 제어)
        manipulator_actions = self.actions[:, 4:12]  # Franka Arm + Gripper (총 8)
        self.robot_dof_targets[:, 4:12] = manipulator_actions
        self._robot.set_joint_position_target(self.robot_dof_targets[:, 4:12], joint_ids=list(range(4, 12)))


        # 액션 출력
        #print(f"Actions: {actions}")

        # 시뮬레이션에 데이터 적용 및 단계 진행
        self._robot.write_data_to_sim()
        self.sim.step()
        #print("Physics step completed")  # 시뮬레이션 스텝 완료 확인

        # 현재 로봇의 중점 속도 출력 (GPU -> CPU 변환)
        robot_velocity = self._robot.data.body_vel_w[:, self.base_link_idx]
        #print(f"Robot center velocity: {robot_velocity}")  # 디버깅: 로봇 중점 속도 출력


    def _apply_action(self):
        pass  # 이미 _pre_physics_step에서 액션을 적용하므로 비워둡니다.

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 최대 시뮬레이션 시간 조건
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        # 거리 조건: 0.025 이하일 때 종료
        success_condition = torch.norm(self.robot_grasp_pos - self.valve_grasp_pos, p=2, dim=-1) <= 0.025

        # 종료 조건 설정
        done = truncated | success_condition

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

        # 바퀴 관련 위치 가져오기 (if needed)
        # 모바일 베이스 (바퀴)와 관련된 위치를 사용하려면 wheel_link_idxs를 이용하여 바퀴 위치를 가져올 수 있습니다.
        # robot_back_left_wheel_link, robot_back_right_wheel_link, robot_front_left_wheel_link, robot_front_right_wheel_link

        # 예시로 바퀴 링크의 위치를 가져오는 방법:
        wheel_positions = [self._robot.data.body_pos_w[:, idx] for idx in self.wheel_link_idxs]

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

        # 모바일 베이스 (바퀴) 초기화
        wheel_joint_ids = self._robot.find_joints("robot_.*_wheel_joint")[0]  # 바퀴 조인트 찾기
        wheel_joint_ids = torch.tensor(wheel_joint_ids, dtype=torch.long, device=self.device)
        mobile_base_pos = torch.zeros((len(env_ids), len(wheel_joint_ids)), device=self.device)

        # ensure env_ids is a tensor
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        # 바퀴 조인트 속도 초기화
        wheel_velocities = torch.zeros((len(env_ids), len(wheel_joint_ids)), device=self.device)  # 바퀴 속도를 0으로 초기화
        self._robot.set_joint_velocity_target(
            wheel_velocities,
            joint_ids=wheel_joint_ids.tolist(),
            env_ids=env_ids,
        )

        # 로봇의 초기 위치와 회전 값 설정
        # init_state에서 제공된 초기 위치와 회전 값
        init_pos = torch.tensor([[1.6, 0.0, 1.3]], device=self.device)  # 초기 위치
        init_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device)  # 초기 회전 (쿼터니언)

        # root pose (위치 + 회전) 설정
        self._robot.write_root_pose_to_sim(torch.cat([init_pos, init_rot], dim=1), env_ids)

        # 로봇의 다른 조인트 초기화
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # 밸브 초기화
        zeros = torch.zeros((len(env_ids), self._valve.num_joints), device=self.device)
        self._valve.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)

        # 중간 값 업데이트
        self._compute_intermediate_values(env_ids)



    def _get_observations(self) -> dict:
        # 로봇의 조인트 위치 (스케일링)
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        # 바퀴 조인트에 대해 NaN 방지 (무제한 범위이므로 NaN 발생 가능)
        for i, name in enumerate(self._robot.data.joint_names):
            if "wheel" in name:  # 바퀴 조인트 식별
                dof_pos_scaled[:, i] = 0.0  # 바퀴 조인트는 학습에 포함하지 않음

        # 로봇의 중점 선형 속도
        robot_velocity = self.robot_center_velocity  # 업데이트된 속도 사용
        robot_speed = torch.norm(robot_velocity, dim=-1, keepdim=True)  # 전체 평면 속도

        # 로봇의 중심 위치
        robot_position = self.robot_center_position  # 업데이트된 위치 사용

        # 목표 위치와의 차이
        to_target = self.valve_grasp_pos - self.robot_grasp_pos

        # 관찰값 생성
        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                to_target,
                self._valve.data.joint_pos[:, 0].unsqueeze(-1),
                self._valve.data.joint_vel[:, 0].unsqueeze(-1),
                robot_speed,          # 로봇 중점 속도 추가
                robot_position,       # 로봇 중심 위치 추가
                torch.zeros(self.num_envs, 1, device=self.device),
            ),
            dim=-1,
        )

        # NaN 또는 Inf 검사
        if torch.isnan(obs).any():
            print("Warning: obs contains NaN values")
        if torch.isinf(obs).any():
            print("Warning: obs contains Inf values")
        # 정책 입력값
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

        # 로봇의 중심 위치 업데이트
        robot_position = self._robot.data.body_pos_w[env_ids, self.base_link_idx]
        self.robot_center_position = robot_position  # 로봇의 중심 위치 저장

        # 디버깅: 로봇의 중심 위치 출력
        print(f"Robot center position: {robot_position}")

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

        # 로봇의 중점 선형 속도 업데이트
        self.robot_center_velocity = self._robot.data.body_vel_w[env_ids, self.base_link_idx, :3]

        # 디버깅: 업데이트된 중점 속도 출력
        print(f"Intermediate Robot center linear velocity: {self.robot_center_velocity}")



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
        # 로봇의 중심 위치 가져오기
        robot_position = self._robot.data.body_pos_w[:, self.base_link_idx]  # [x, y, z]

        # 디버깅: 로봇의 중심 위치 출력
        print(f"Robot center position: {robot_position}")

        # 기존의 보상 계산 코드 유지
        # 로봇의 중점 선형 속도 계산
        robot_velocity = self._robot.data.body_vel_w[:, self.base_link_idx, :3]
        robot_speed = torch.norm(robot_velocity, dim=-1)

        # 기존의 보상 계산 로직 유지
        robot_speed_reward = robot_speed * base_reward_scale
        rewards = (
            robot_speed_reward
            - action_penalty_scale * torch.sum(torch.abs(actions), dim=-1)
        )

        # 로깅 정보
        self.extras["log"] = {
            "robot_speed_reward": robot_speed_reward.mean()
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
