# Valve Manipulation Environments

Envirionment
- Ubuntu 22.04
- IsaacSim 4.5.0
- IsaacLab 2.0.0
- GeForce RTX 4070Ti
- Nvidia Driver 550.120
- CUDA 12.4

아래에는 총 세 가지 환경 예제가 포함되어 있습니다:
- **360_random_valve**
- **random_valve**
- **random_valve_ros** (ROS2로 카메라 처리를 분리한 버전)

---

## 360_random_valve

<video src="./360valve.mp4" controls style="max-width: 100%; height: auto;">
  동영상을 재생할 수 없는 환경이라면, `360valve.mp4` 파일을 직접 다운로드하여 확인하세요.
</video>

### 소개
- 밸브(Valve)가 360도 무작위 각도로 배치되어, 로봇(모바일 베이스+프랑카)이 다양한 방향에서 접근하여 밸브를 찾고 회전시킬 수 있도록 설계된 환경입니다.
- 에피소드 시작 시 밸브 회전 각도와 로봇 초기 자세 등을 무작위화하여, 보다 **일반화된 학습**을 유도합니다.

### 학습 명령어
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py
--task Isaac-camera-franka-Direct-v0
--num_envs 1024
--headless

- `camera_franka.py` 파일은 `360_random_valve.py`와 **동일**해야 합니다.  
- `num_envs`가 1024를 초과하면 GPU/CPU 리소스 부족 문제가 발생할 수 있습니다.

### 실행 명령어
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py
--task Isaac-camera-franka-Direct-v0
--num_envs 1
--checkpoint logs/rl_games/camera_franka_direct/360_random_valve/nn/camera_franka_direct.pth
- 학습 완료된 모델(`.pth`)을 원하는 버전으로 바꿔서 지정할 수 있습니다.

---

## random_valve

<video src="./random_valve.mp4" controls style="max-width: 100%; height: auto;">
  동영상을 재생할 수 없는 환경이라면, `random_valve.mp4` 파일을 직접 다운로드하여 확인하세요.
</video>

### 소개
- 밸브의 위치와 각도가 무작위로 배치되는 환경입니다.
- 로봇(모바일 베이스+프랑카)이 에피소드마다 다른 위치/자세의 밸브를 탐색하고 조작해야 하므로, **탐색(approach) 능력**과 **조작(manipulation) 능력**을 고루 학습할 수 있습니다.

### 학습 명령어
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py
--task Isaac-camera-franka-Direct-v0
--num_envs 64
--enable_cameras
--headless
- `camera_franka.py` 파일은 `random_valve.py`와 **동일**해야 합니다.  
- `num_envs`가 64를 초과하면 리소스 부족 문제가 발생할 수 있습니다.

### 실행 명령어
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py
--task Isaac-camera-franka-Direct-v0
--num_envs 1
--enable_cameras
--checkpoint logs/rl_games/camera_franka_direct/random_valve/nn/camera_franka_direct.pth
- 학습된 모델(`.pth`)은 필요에 따라 다른 체크포인트로 변경 가능합니다.

---

## random_valve_ros

### 소개
- **random_valve** 환경과 동일한 무작위 밸브 배치를 사용하지만, **카메라 이미지를 ROS2 노드로 전달**하여 별도의 `yolo.py`에서 YOLO 추론을 수행하도록 구성한 버전입니다.
- 시뮬레이션(Isaac Sim) 측과 **ROS2** 측(외부 YOLO 노드)이 **통신**하며 밸브 검출 결과를 주고받아 로봇의 행동에 반영하는 구조입니다.
- 학습/실행 명령어는 `random_valve`와 동일하지만, **별도의 ROS2 노드를 통해 `yolo.py`** 등을 실행해야 합니다.
