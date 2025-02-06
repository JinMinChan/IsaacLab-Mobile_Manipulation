# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates the different camera sensors that can be attached to a robot.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates the different camera sensor implementations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--disable_fabric", action="store_true", help="Disable Fabric API and use USD instead.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import cv2
from ultralytics import YOLO

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sensors import CameraCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

##
# Pre-defined configs
##
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort:skip
from omni.isaac.lab_assets.floating_franka import FLOATING_FRANKA_CFG  # isort:skip
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg

@configclass
class SensorsSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot and include valve configuration."""

    # ground plane
    ground = TerrainImporterCfg(
        prim_path="/World/ground",
        max_init_terrain_level=None,
        terrain_type="plane",
        terrain_generator=ROUGH_TERRAINS_CFG.replace(color_scheme="none"),
        visual_material=None,
        debug_vis=False,
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot: ArticulationCfg = FLOATING_FRANKA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/front_cam",  # 카메라 위치
        update_period=0.1,
        height=640,  # 해상도
        width=640,
        data_types=["rgb"],  # RGB만 받도록 설정
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.0), rot=(0.0, 0.7071, 0.0, 0.7071), convention="world"),
    )

    # valve 설정 추가
    valve = ArticulationCfg(
        prim_path="/World/envs/env_.*/Valve",  # 밸브의 prim_path 설정
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/vision/Downloads/valves/round_valve/round_valve_main.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(3.5, 0, 0.0),  # 초기 위치 설정
            rot=(0.0, 0.0, 0.0, 0.0),  # 초기 회전 설정
            joint_pos={
                "valve_handle_joint": 0.0,  # 초기 조인트 상태
            },
        ),
        actuators={
            "valve": ImplicitActuatorCfg(
                joint_names_expr=["valve_handle_joint"],  # 밸브 조인트 이름
                effort_limit=100.0,
                velocity_limit=10.0,
                stiffness=20.0,
                damping=10.0,
            ),
        },
    )

def save_images_grid(
    images: list[torch.Tensor],
    cmap: str | None = None,
    nrow: int = 1,
    subtitles: list[str] | None = None,
    title: str | None = None,
    filename: str | None = None,
):
    """Save images in a grid with optional subtitles and title."""
    n_images = len(images)
    ncol = int(np.ceil(n_images / nrow))

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))

    if n_images == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    for idx, (img, ax) in enumerate(zip(images, axes)):
        img = img.detach().cpu().numpy()
        ax.imshow(img, cmap=cmap)
        ax.axis("off")
        if subtitles:
            ax.set_title(subtitles[idx])

    for ax in axes[n_images:]:
        fig.delaxes(ax)

    if title:
        plt.suptitle(title)

    plt.tight_layout()

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
    plt.close()

def generate_random_joint_positions(robot, device="cuda:0"):
    """Generate random joint positions for Franka within its joint limits."""
    joint_names = robot.data.joint_names
    # 'panda'라는 문자열을 포함하는 조인트 이름만 선택 (panda0, panda1, ... panda7)
    franka_joint_indices = [
        i for i, name in enumerate(joint_names) if "panda" in name.lower() and "panda_link" not in name.lower()
    ]

    # Franka 로봇의 joint limits (12개의 조인트)
    joint_limits = robot.data.joint_limits.squeeze(0)  # (12, 2)
    joint_lower_limits = joint_limits[franka_joint_indices, 0]  # 하한선
    joint_upper_limits = joint_limits[franka_joint_indices, 1]  # 상한선

    # 선택된 조인트 인덱스에 대해 랜덤 포지션 생성
    random_positions = torch.rand_like(joint_lower_limits) * (joint_upper_limits - joint_lower_limits) + joint_lower_limits
    return random_positions.to(device), franka_joint_indices

def capture_image_from_camera(camera):
    """Capture an image from the Isaac Sim camera sensor and save it for debugging."""
    # GPU에서 데이터를 CPU로 가져오고 numpy로 변환
    rgb_image = camera.data.output["rgb"][0, ..., :3].cpu().numpy()
    rgb_image = np.array(rgb_image, dtype=np.uint8)

    # RGB에서 BGR로 변환 (OpenCV를 위한 설정)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # 디버깅용 이미지 저장
    debug_image_path = "/home/vision/Downloads/minchan_yolo/debug_image.jpg"
    cv2.imwrite(debug_image_path, bgr_image)
    print(f"[DEBUG] Image saved for debugging at {debug_image_path}")

    return bgr_image




def detect_valve_with_yolo(image, model, valve_class_name="valve"):
    """
    Use YOLO to detect objects in the image and check if 'valve' is detected.
    
    Args:
        image: Captured image from the camera.
        model: YOLO model instance.
        valve_class_name: The class name of the valve in your dataset.
    
    Returns:
        bool: True if valve is detected, False otherwise.
    """
    results = model.predict(image)
    for box in results[0].boxes.data:
        class_id = int(box[-1])
        class_name = model.names[class_id]
        if class_name == valve_class_name:
            print(f"[INFO] Valve detected: {box[:4].cpu().numpy()}")
            return True
    return False

def run_simulator_with_random_motion_and_detection(sim, scene, camera, yolo_model):
    """Run the simulator and use YOLO to detect valves in camera images."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Debug available scene entities
    available_entities = [key for key in dir(scene) if not key.startswith("_")]
    print(f"[DEBUG]: Available entities in the scene: {available_entities}")

    while simulation_app.is_running():
        if count % 100 == 0:
            # 개별적으로 값을 설정하려면, 각 joint마다 값을 지정
            fixed_positions = torch.tensor([
                0.0,  # panda_joint0
                0.0,   # panda_joint1
                0.0,   # panda_joint2
                0.0,   # panda_joint3
                0.0,   # panda_joint4
                1.5,   # panda_joint5
                0.8,   # panda_joint6
                0.0, # panda_joint7
            ]).to("cuda:0")  # CUDA로 이동

            # 각 조인트에 대한 인덱스를 정확하게 설정
            joint_ids = [3, 4, 5, 6, 7, 8, 9, 10]  # panda_joint0부터 panda_joint7에 해당하는 인덱스

            # 각 조인트의 값을 하나씩 설정
            scene["robot"].set_joint_position_target(fixed_positions.unsqueeze(0), joint_ids=joint_ids)

            # Capture image from the camera
            captured_image = capture_image_from_camera(camera)
            valve_detected = detect_valve_with_yolo(captured_image, yolo_model)

            if valve_detected:
                print("[INFO] Valve detected!")
            else:
                print("[INFO] Valve not detected.")

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device, use_fabric=not args_cli.disable_fabric)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

    scene_cfg = SensorsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    camera = scene["camera"]

    # Load YOLO model
    yolo_model = YOLO('/home/vision/Downloads/minchan_yolo/train/weights/best.pt')

    sim.reset()

    print("[INFO]: Setup complete...")
    run_simulator_with_random_motion_and_detection(sim, scene, camera, yolo_model)

if __name__ == "__main__":
    main()
    simulation_app.close()
