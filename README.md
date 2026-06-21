# Unitree G1 Humanoid Locomotion RL (Isaac Lab)

## 목표

NVIDIA Isaac Lab 환경에서 RSL-RL(PPO) 기반으로 Unitree G1 로봇의
velocity tracking 보행 정책을 학습시킨다.

---

## 강화학습 이론 배경

### 강화학습이란

강화학습(RL)은 에이전트가 환경과 상호작용하며 시행착오를 통해 보상을 최대화하는
행동을 학습하는 머신러닝 방법이다. 에이전트는 환경의 상태(state)를 관찰하고,
행동(action)을 선택하며, 그 결과로 보상(reward)을 받는다. 이 상태-행동 매핑을
정책(policy)이라 하며, 학습의 목표는 누적 보상을 최대화하는 정책을 찾는 것이다.

### 강화학습의 기본 유형 3가지

1. **동적 프로그래밍 (Dynamic Programming)** — 환경의 모델을 알고 있다고 가정하고 계산
2. **몬테카를로 (Monte Carlo)** — 모델 없이, 에피소드가 끝날 때까지 기다려서 학습
3. **시간차 학습 (Temporal Difference, TD)** — 모델 없이, 매 스텝마다 즉시 업데이트
   (위 두 방식의 절충안)

### 알고리즘 구조에 따른 분류

- **Value-based**: 가치 함수를 기반으로 행동 선택
- **Policy-based**: 가치 함수 없이 정책을 직접 학습 (고차원 환경에 유리)
- **Actor-Critic**: 위 둘을 결합. Actor가 행동을 결정하고, Critic이 그 행동을 평가

### 이 프로젝트에서 사용한 알고리즘: PPO

PPO(Proximal Policy Optimization)는 TD 학습 계열의 on-policy Actor-Critic
알고리즘이다.

- 정책을 한 번에 급격히 바꾸지 않도록 변화량을 clip(제한)하여 학습 안정성 확보
- GAE(Generalized Advantage Estimation)로 advantage를 추정해 분산을 줄이고
  학습 효율을 높임
- 본 프로젝트에서는 **RSL-RL** 라이브러리(ETH Zurich 개발, 로보틱스 locomotion
  특화)로 구현된 PPO를 사용
- **Isaac Lab** 환경에서 Unitree G1 로봇의 velocity tracking 태스크 학습에 적용

---

## 기술 스택

```
RSL-RL (PPO 알고리즘)
    └─ Isaac Lab + unitree_rl_lab (RL 환경 프레임워크)
         └─ Isaac Sim / PhysX (물리 시뮬레이션 + 로봇 모델)
              └─ GPU, RTX 5060 Ti (병렬 시뮬레이션 연산)
```

| 레이어 | 도구 | 역할 |
|---|---|---|
| 알고리즘 | RSL-RL (PPO) | PyTorch 기반 정책/가치 신경망 학습 |
| 프레임워크 | Isaac Lab, unitree_rl_lab | G1 전용 RL 환경(observation/action/reward) 정의 |
| 시뮬레이션 | Isaac Sim (PhysX) | 물리 연산, 로봇 모델(URDF/USD) 적용 |
| 하드웨어 | GPU (RTX 5060 Ti) | 수백 개 환경 병렬 시뮬레이션 |

보조 도구: Conda(환경 관리), Git/GitHub(버전 관리), Ubuntu, Hugging Face(로봇 모델 파일)

---

## 진행 과정

- [x] Isaac Lab / unitree_rl_lab 환경 세팅 (Isaac Sim 6.0 호환을 위해 IsaacLab `develop` 브랜치로 전환, 누락 의존성 설치)
- [x] G1 로봇 모델(USD/URDF) 연결 (`unitree_model` 데이터셋, `UNITREE_MODEL_DIR` 환경변수)
- [x] `Unitree-G1-29dof-Velocity` 태스크 학습 (headless, `num_envs=2048`, `max_iterations=3000`)
- [x] 학습 곡선 모니터링 (tensorboard)
- [x] `play` 모드로 학습된 정책 시각화
- [ ] (가능하면) reward 파라미터 변경 실험

코드 구조를 직접 분석한 내용은 [`docs/code_walkthrough.md`](docs/code_walkthrough.md) 참고.

## 결과

- 학습 설정: `num_envs=2048`, `max_iterations=3000`, 소요 시간 1시간 19분 32초 (RTX 5060 Ti)
- Mean reward: 약 -4.8 (초반) → 양수로 개선
- `Metrics/base_velocity/error_vel_xy`: 0.55 → 0.21 (속도 추종 오차 감소)
- `Metrics/base_velocity/error_vel_yaw`: 3.1 → 0.90 (회전 추종 오차 감소)
- `Episode_Reward/feet_clearance`: 0.04 → 0.74, `Episode_Reward/gait`: 0.01 → 0.35 (걸음걸이 패턴 형성)
- 데모 영상: [`results/g1_walking_demo.mp4`](results/g1_walking_demo.mp4) (`play` 모드, 32 envs)
- 학습 진행 영상: [`results/g1_training_progression.mp4`](results/g1_training_progression.mp4) — 체크포인트(iteration 0, 100, 300, 600, 1000, 1500, 2200, 2999)별로 같은 정책을 `play` 모드로 재생해 이어붙인 것. 초반엔 제자리에서 넘어지기만 하던 로봇이 점점 명령된 속도로 걷는 정책으로 바뀌는 과정을 확인할 수 있음.

## 배운 점

> (이 부분은 본인이 `docs/code_walkthrough.md`를 직접 읽고, observation/action/reward/termination 설계와 PPO 하이퍼파라미터를 이해한 내용을 본인 언어로 정리할 것. 환경 호환성 디버깅 과정에서 Isaac Sim/IsaacLab 버전 관리에 대해 배운 점도 포함하면 좋음.)

## 향후 계획

학습된 정책을 MuJoCo에서 sim-to-sim 검증 후, unitree_sdk2를 통해 실물 로봇에
적용(sim-to-real)하는 것까지 확장 가능. 로보티즈의 다이나믹셀-Q 액추에이터에
내장된 AI SIM 기능도 이와 유사한 시뮬레이션→실물 적용 컨셉이라 향후 연결 지점으로
고려 중.

---

## Reference

- [Unitree RL Lab (GitHub, 원본)](https://github.com/unitreerobotics/unitree_rl_lab)
- [Unitree RL Lab (본인 fork, Isaac Sim 6.0 호환 패치)](https://github.com/dbswnstj0413/unitree_rl_lab)
- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab)

## 도구 사용 관련

환경 세팅(IsaacLab/Isaac Sim 버전 호환성 디버깅, 의존성 설치, 스크립트 실행)은 Claude Code의 도움을 받아 진행함.
RL 알고리즘 설계, reward/observation 구성, 하이퍼파라미터의 의미는 `docs/code_walkthrough.md`를 통해 직접 분석하고 이해함.


# Unitree RL Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-6.0.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-3.0.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?style=flat&logo=Discord&logoColor=white)](https://discord.gg/ZwcVwxv5rq)


## Overview

This project provides a set of reinforcement learning environments for Unitree robots, built on top of [IsaacLab](https://github.com/isaac-sim/IsaacLab).

Currently supports Unitree **Go2**, **H1** and **G1-29dof** robots.

<div align="center">

| <div align="center"> Isaac Lab </div> | <div align="center">  Mujoco </div> |  <div align="center"> Physical </div> |
|--- | --- | --- |
| [<img src="https://oss-global-cdn.unitree.com/static/d879adac250648c587d3681e90658b49_480x397.gif" width="240px">](g1_sim.gif) | [<img src="https://oss-global-cdn.unitree.com/static/3c88e045ab124c3ab9c761a99cb5e71f_480x397.gif" width="240px">](g1_mujoco.gif) | [<img src="https://oss-global-cdn.unitree.com/static/6c17c6cf52ec4e26bbfab1fbf591adb2_480x270.gif" width="240px">](g1_real.gif) |

</div>

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
- Install the Unitree RL IsaacLab standalone environments.

  - Clone or copy this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

    ```bash
    git clone https://github.com/unitreerobotics/unitree_rl_lab.git
    ```
  - Use a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    conda activate env_isaaclab
    ./unitree_rl_lab.sh -i
    # restart your shell to activate the environment changes.
    ```
- Download unitree robot description files

  *Method 1: Using USD Files*
  - Download unitree usd files from [unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model/tree/main), keeping folder structure
    ```bash
    git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
    ```
  - Config `UNITREE_MODEL_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`.

    ```bash
    UNITREE_MODEL_DIR = "</home/user/projects/unitree_usd>"
    ```

  *Method 2: Using URDF Files [Recommended]* Only for Isaacsim >= 5.0
  -  Download unitree robot urdf files from [unitree_ros](https://github.com/unitreerobotics/unitree_ros)
      ```
      git clone https://github.com/unitreerobotics/unitree_ros.git
      ```
  - Config `UNITREE_ROS_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`.
    ```bash
    UNITREE_ROS_DIR = "</home/user/projects/unitree_ros/unitree_ros>"
    ```
  - [Optional]: change *robot_cfg.spawn* if you want to use urdf files



- Verify that the environments are correctly installed by:

  - Listing the available tasks:

    ```bash
    ./unitree_rl_lab.sh -l # This is a faster version than isaaclab
    ```
  - Running a task:

    ```bash
    ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity # support for autocomplete task-name
    # same as
    python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity
    ```
  - Inference with a trained agent:

    ```bash
    ./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity # support for autocomplete task-name
    # same as
    python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity
    ```

## Deploy

After the model training is completed, we need to perform sim2sim on the trained strategy in Mujoco to test the performance of the model.
Then deploy sim2real.

### Setup

```bash
# Install dependencies
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
# Install unitree_sdk2
git clone git@github.com:unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF # Install on the /usr/local directory
sudo make install
# Compile the robot_controller
cd unitree_rl_lab/deploy/robots/g1_29dof # or other robots
mkdir build && cd build
cmake .. && make
```

### Sim2Sim

Installing the [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco?tab=readme-ov-file#installation).

- Set the `robot` at `/simulate/config.yaml` to g1
- Set `domain_id` to 0
- Set `enable_elastic_hand` to 1
- Set `use_joystck` to 1.

```bash
# start simulation
cd unitree_mujoco/simulate/build
./unitree_mujoco
# ./unitree_mujoco -i 0 -n eth0 -r g1 -s scene_29dof.xml # alternative
```

```bash
cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl
# 1. press [L2 + Up] to set the robot to stand up
# 2. Click the mujoco window, and then press 8 to make the robot feet touch the ground.
# 3. Press [R1 + X] to run the policy.
# 4. Click the mujoco window, and then press 9 to disable the elastic band.
```

### Sim2Real

You can use this program to control the robot directly, but make sure the on-borad control program has been closed.

```bash
./g1_ctrl --network eth0 # eth0 is the network interface name.
```

## Acknowledgements

This repository is built upon the support and contributions of the following open-source projects. Special thanks to:

- [IsaacLab](https://github.com/isaac-sim/IsaacLab): The foundation for training and running codes.
- [mujoco](https://github.com/google-deepmind/mujoco.git): Providing powerful simulation functionalities.
- [robot_lab](https://github.com/fan-ziqi/robot_lab): Referenced for project structure and parts of the implementation.
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking): Versatile humanoid control framework for motion tracking.
