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


def setup_environment():
    """로봇, 밸브, 평면을 추가하고 초기 상태를 설정"""
    sim = SimulationContext(render=True)  # 렌더링 활성화

    # 로봇 추가
    robot_usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd"
    robot = Articulation(
        prim_path="/World/Robot",
        usd_path=robot_usd_path,
        translation=(0.0, 0.0, 0.5),  # 로봇 위치
        rotation=(0.0, 0.0, 0.0, 1.0),  # 로봇 방향
    )

    # 밸브 추가
    valve_usd_path = "/home/vision/Downloads/valves/round_valve/round_valve_main.usd"
    valve = Articulation(
        prim_path="/World/Valve",
        usd_path=valve_usd_path,
        translation=(0.5, 0.0, 0.0),  # 밸브 위치
        rotation=(0.0, 0.0, 0.0, 1.0),  # 밸브 방향
    )

    # 평면 추가
    sim.add_articulation(robot)
    sim.add_articulation(valve)

    return sim


def main():
    sim = setup_environment()
    sim.reset()
    print("[INFO]: Simulation setup complete. Press Play to start simulation.")

    while True:
        if sim.is_playing():
            sim.step()
        elif not sim.is_running():
            break


if __name__ == "__main__":
    main()
