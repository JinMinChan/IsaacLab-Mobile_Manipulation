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
from omni.isaac.core.utils.stage import get_current_stage

@configclass
class FrankaValveEnvCfg(DirectRLEnvCfg):
    # 환경 설정
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    action_space = 9
    observation_space = 23
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
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
            activate_contact_sensors=True,
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
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.035,
            },
            pos=(1.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
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

    # valve 설정
    valve = ArticulationCfg(
        prim_path="/World/envs/env_.*/Valve",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/vision/Downloads/valves/round_valve/round_valve_main.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.5, 0, 0.0),
            rot=(-0.707, -0.707, 0.0, 0.0),
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
            static_friction=5.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    action_scale = 5.0              # 행동 크기를 줄여 더 섬세한 동작을 유도
    dof_velocity_scale = 0.05       # 속도 스케일도 감소시켜 안정적인 움직임 촉진

    # 보상 스케일
    dist_reward_scale = 8.0         # 거리 기반 보상: 현재보다 약간 감소
    rot_reward_scale = 5.0          # 축 정렬 보상 유지
    grip_reward_scale = 6.0         # 파지 시도 보상 강화 (파지를 더 강하게 유도)
    action_penalty_scale = 0.05     # 행동 페널티: 과도한 행동을 억제하기 위해 증가
    turn_reward_scale = 15.0        # 밸브 회전 보상 강화 (밸브 회전의 중요성 강조)
    stability_reward_scale = 0.7    # 안정성 보상 유지
    grip_threshold = 0.05           # 그립 임계값을 약간 늘려 파지 성공률 향상
    proximity_reward_scale=1.0


class FrankaValveEnv(DirectRLEnv):
    cfg: FrankaValveEnvCfg

    def __init__(self, cfg: FrankaValveEnvCfg, render_mode: str | None = None, **kwargs):
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
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1
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

        # 손가락의 그립 위치
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
        # 0.35 ,0 ,0.655
        valve_local_grasp_pose = torch.tensor([0.14, 0.18, 0.00, 1.0, 0.0, 0.0, 0.0], device=self.device)  # 중심 위치
        self.valve_local_grasp_pos = valve_local_grasp_pose[0:3].repeat((self.num_envs, 1))
        self.valve_local_grasp_rot = valve_local_grasp_pose[3:7].repeat((self.num_envs, 1))

        #EE가 Valve를 파지할 때 전진하는 축을 나타냅니다.
        self.gripper_forward_axis = torch.tensor([0, 0, -1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.valve_inward_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.valve_up_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )

        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        self.valve_link_idx = self._valve.find_bodies("valve_handle")[0][0]

        # 초기화
        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.valve_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.valve_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

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

    # physics step 전 호출
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return truncated, truncated


    def _get_rewards(self) -> torch.Tensor:
        # 중간 계산 수행
        self._compute_intermediate_values()

        # 왼손가락과 오른손가락 위치 가져오기
        robot_left_finger_pos = self._robot.data.body_pos_w[:, self.left_finger_link_idx]
        robot_right_finger_pos = self._robot.data.body_pos_w[:, self.right_finger_link_idx]
        
        # 왼손가락과 오른손가락의 중간 지점을 franka_grasp_pos로 설정
        franka_grasp_pos = (robot_left_finger_pos + robot_right_finger_pos) / 2

        # 보상 계산 호출
        return self._compute_rewards(
            actions=self.actions,
            franka_lfinger_pos=robot_left_finger_pos,
            franka_rfinger_pos=robot_right_finger_pos,
            valve_grasp_pos=self.valve_grasp_pos,
            franka_grasp_pos=franka_grasp_pos,  # 수정된 franka_grasp_pos 추가
            franka_grasp_rot=self.robot_grasp_rot,
            valve_grasp_rot=self.valve_grasp_rot,
            gripper_forward_axis=self.gripper_forward_axis,
            valve_inward_axis=self.valve_inward_axis,
            num_envs=self.num_envs,
            rot_reward_scale=self.cfg.rot_reward_scale,
            action_penalty_scale=self.cfg.action_penalty_scale,
            proximity_reward_scale=self.cfg.proximity_reward_scale
        )





    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
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

        zeros = torch.zeros((len(env_ids), self._valve.num_joints), device=self.device)
        self._valve.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)
        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        to_target = self.valve_grasp_pos - self.robot_grasp_pos
        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                to_target,
                self._valve.data.joint_pos[:, 0].unsqueeze(-1),
                self._valve.data.joint_vel[:, 0].unsqueeze(-1),
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
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

    def _compute_rewards(
        self,
        actions,
        franka_grasp_pos,
        franka_lfinger_pos,
        franka_rfinger_pos,
        valve_grasp_pos,
        franka_grasp_rot,
        valve_grasp_rot,
        gripper_forward_axis,
        valve_inward_axis,
        num_envs,
        rot_reward_scale,
        action_penalty_scale,
        proximity_reward_scale,
    ) -> torch.Tensor:

        # 1. 두 손가락 사이의 중간점과 Valve의 거리 계산
        mid_point = (franka_lfinger_pos + franka_rfinger_pos) / 2.0
        dist_to_valve = torch.norm(valve_grasp_pos - mid_point, p=2, dim=-1)

        # 2. 정렬에 비례하는 grip position 보상
        axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(valve_grasp_rot, valve_inward_axis)
        dot_product = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        
        # dot_product에 비례한 grip position 보상 설정
        grip_position_reward = torch.exp(-dist_to_valve * 10) * 5 * torch.clamp(dot_product, 0.0, 1.0)
        grip_position_reward = torch.where(
            dist_to_valve < 0.01,
            grip_position_reward * 2,  # 0.01 이하일 때 보상 2배
            grip_position_reward
        )

        # 3. 손가락 간격 패널티
        finger_distance = torch.norm(franka_lfinger_pos - franka_rfinger_pos, p=2, dim=-1)
        finger_distance_penalty = torch.where(
            dist_to_valve > 0.01,
            (1.0 / (1.0 + finger_distance**2)) * -1,
            torch.tensor(0.0, device=self.device)
        )

        # 최종 보상 합산
        rewards = (
            grip_position_reward  # 정렬 상태에 비례한 grip position 보상
            - finger_distance_penalty  # 손가락 간격 패널티
        )

        # 보상 로그 (디버깅용)
        self.extras["log"] = {
            "grip_position_reward": grip_position_reward.mean(),
            "finger_distance_penalty": finger_distance_penalty.mean(),
        }

        #print("dist_to_valve:", dist_to_valve[0].cpu().numpy())
        self.visualize_valve_grasp_pos()
        self.visualize_axes()
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
    

    def visualize_valve_grasp_pos(self):
        stage = get_current_stage()
        
        # Valve grasp position 시각화 (빨간색 구)
        for i, pos in enumerate(self.valve_grasp_pos):
            pos_np = pos.cpu().numpy().astype(float)
            sphere_path = f"/World/Visuals/valve_grasp_pos_sphere_{i}"
            
            if not stage.GetPrimAtPath(sphere_path):
                sphere = UsdGeom.Sphere.Define(stage, sphere_path)
                sphere.GetRadiusAttr().Set(0.01)
                sphere.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])  # 빨간색

            sphere_xform = UsdGeom.Xformable(stage.GetPrimAtPath(sphere_path))
            translate_ops = sphere_xform.GetOrderedXformOps()
            if translate_ops and translate_ops[0].GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_ops[0].Set(Gf.Vec3d(*pos_np))
            else:
                sphere_xform.AddTranslateOp().Set(Gf.Vec3d(*pos_np))

        # Franka grasp position 시각화 (파란색 구)
        # franka_grasp_pos를 왼손가락과 오른손가락의 중간 지점으로 계산
        for i in range(self.num_envs):
            robot_left_finger_pos = self._robot.data.body_pos_w[i, self.left_finger_link_idx]
            robot_right_finger_pos = self._robot.data.body_pos_w[i, self.right_finger_link_idx]
            franka_grasp_pos = (robot_left_finger_pos + robot_right_finger_pos) / 2

            pos_np = franka_grasp_pos.cpu().numpy().astype(float)
            sphere_path = f"/World/Visuals/franka_grasp_pos_sphere_{i}"
            
            if not stage.GetPrimAtPath(sphere_path):
                sphere = UsdGeom.Sphere.Define(stage, sphere_path)
                sphere.GetRadiusAttr().Set(0.01)
                sphere.GetDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.0, 1.0)])  # 파란색

            sphere_xform = UsdGeom.Xformable(stage.GetPrimAtPath(sphere_path))
            translate_ops = sphere_xform.GetOrderedXformOps()
            if translate_ops and translate_ops[0].GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_ops[0].Set(Gf.Vec3d(*pos_np))
            else:
                sphere_xform.AddTranslateOp().Set(Gf.Vec3d(*pos_np))



    def visualize_axes(self):
        stage = get_current_stage()

        # 축 시각화를 위한 설정 함수
        def create_axis_line(start_pos, direction, length, path, color):
            end_pos = start_pos + direction * length
            line_prim = stage.GetPrimAtPath(path)
            
            if not line_prim:
                line = UsdGeom.BasisCurves.Define(stage, path)
                line.CreateWidthsAttr([0.01, 0.01])  # 폭 설정: 두 개의 점마다 width 설정
                line.CreateTypeAttr().Set("linear")  # 선형 곡선으로 설정
                line.CreateDisplayColorAttr([color])  # 색상 설정
            else:
                line = UsdGeom.BasisCurves(line_prim)

            # 업데이트
            points = [Gf.Vec3f(*start_pos.cpu().numpy().tolist()), Gf.Vec3f(*end_pos.cpu().numpy().tolist())]
            line.GetPointsAttr().Set(points)
            line.GetCurveVertexCountsAttr().Set([2])  # 두 개의 포인트로 구성된 선형 곡선

        # Gripper Forward Axis 시각화 (파란색)
        gripper_start = self.robot_grasp_pos[0]
        create_axis_line(gripper_start, self.gripper_forward_axis[0], 0.1, "/World/Visuals/gripper_forward_axis_line", (0, 0, 1))

        # Valve Inward Axis 시각화 (빨간색)
        valve_start = self.valve_grasp_pos[0]
        create_axis_line(valve_start, self.valve_inward_axis[0], 0.1, "/World/Visuals/valve_inward_axis_line", (1, 0, 0))

        # Gripper Up Axis 시각화 (초록색)
        create_axis_line(gripper_start, self.gripper_up_axis[0], 0.1, "/World/Visuals/gripper_up_axis_line", (0, 1, 0))

        # Valve Up Axis 시각화 (노란색)
        create_axis_line(valve_start, self.valve_up_axis[0], 0.1, "/World/Visuals/valve_up_axis_line", (1, 1, 0))



    # 각 시뮬레이션 스텝마다 visualize_valve_grasp_pos 호출
    def _post_physics_step(self):
        self.visualize_valve_grasp_pos()
        self.visualize_axes()  # 매 스텝마다 축을 시각적으로 업데이트
