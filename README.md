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
# G1-29dof-Velocity 코드 분석 (강화학습 파이프라인 한 줄씩 이해하기)

> 대상 레포: [unitree_rl_lab](https://github.com/dbswnstj0413/unitree_rl_lab) (`isaaclab3.0` 브랜치, unitreerobotics 원본을 Isaac Sim 6.0 / IsaacLab `develop`에 맞게 패치)
> 목적: "환경 세팅은 AI 도움을 받았지만, RL 알고리즘과 코드 내용은 본인이 이해하고 있다"는 걸 보이기 위한 정리.

---

## 학습 진입점 — `scripts/rsl_rl/train.py`

학습 한 번 실행할 때 일어나는 일의 순서:

1. `gym.make(args_cli.task, cfg=env_cfg, ...)` — Isaac Lab의 시뮬레이션 환경을 생성. 이 한 줄이 실제로는 GPU에 `num_envs`개(우리는 2048개)의 G1 로봇을 병렬로 스폰하고, 물리 시뮬레이션을 초기화하는 무거운 작업.
2. `RslRlVecEnvWrapper(env, ...)` — IsaacLab 환경을 RSL-RL 라이브러리가 이해하는 벡터화된 인터페이스로 감싸는 어댑터.
3. `OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=...)` — RSL-RL의 PPO 러너 생성. 정책망(actor)/가치망(critic) 초기화는 여기서 일어남.
4. `runner.learn(num_learning_iterations=agent_cfg.max_iterations, ...)` — 실제 학습 루프 시작. 내부적으로 매 iteration마다:
   - **Rollout 수집**: 2048개 환경에서 `num_steps_per_env`(=24) 스텝만큼 행동을 실행하고 (state, action, reward) 기록
   - **Advantage 계산**: GAE(Generalized Advantage Estimation)로 각 스텝이 "얼마나 좋았는지" 추정
   - **정책 업데이트**: PPO clip loss로 actor/critic 네트워크를 `num_learning_epochs`(=5)번 반복 학습

→ "AI가 학습을 돌렸다"가 아니라 이 4단계가 PPO 알고리즘의 표준 흐름이라는 걸 알고 있어야 함.

---

## 환경 정의 — `velocity_env_cfg.py`

강화학습에서 가장 중요한 파일. MDP(Markov Decision Process)의 구성요소가 전부 여기 정의됨.

### Observation (관측값) — `ObservationsCfg`

정책(policy)이 보는 입력:

| 항목 | 의미 | 노이즈 |
|---|---|---|
| `base_ang_vel` | 몸통 각속도 | ±0.2 |
| `projected_gravity` | 중력 방향 (기울어짐 감지용) | ±0.05 |
| `velocity_commands` | "이 방향/속도로 가라"는 명령 | - |
| `joint_pos_rel` | 각 관절의 (기준 자세 대비) 위치 | ±0.01 |
| `joint_vel_rel` | 각 관절 속도 | ±1.5 |
| `last_action` | 이전 스텝에 취한 행동 | - |

노이즈를 일부러 넣는 이유: 실제 센서는 항상 노이즈가 있으므로, 시뮬레이션에서도 노이즈를 넣어야 sim-to-real 전이 시 정책이 깨지지 않음.

`CriticCfg`는 정책망과 별개로 가치망(critic)이 보는 입력인데, `base_lin_vel`(실제 선속도)처럼 실제 로봇에서는 직접 측정하기 어려운 "특권 정보(privileged information)"를 추가로 포함. 학습할 때만 critic이 더 정확한 정보를 보고, 실제 배포되는 건 policy 망뿐이라 문제 없음.

### Action (행동) — `ActionsCfg`

```python
JointPositionAction = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)
```

행동 = 모든 관절의 "목표 위치(joint position target)". `scale=0.25`는 신경망 출력(-1~1 근처)을 실제 관절각 변화량으로 줄여서 매핑하는 계수 — 신경망이 한 스텝에 너무 큰 관절각 변화를 명령하지 못하게 제한.

### Reward (보상) — `RewardsCfg`

각 항목과 가중치(weight)가 왜 그런지가 핵심:

- **`track_lin_vel_xy` (+1.0), `track_ang_vel_z` (+0.5)**: 명령받은 속도를 얼마나 잘 추종하는지. 이게 태스크의 본질적 목표라 가중치가 가장 큼.
- **`alive` (+0.15)**: 안 넘어지고 살아있으면 매 스텝 보상.
- **`base_linear_velocity`/`angular_velocity` (-2.0/-0.05)**: 불필요한 수직/회전 흔들림에 패널티.
- **`joint_vel`, `joint_acc`, `action_rate`, `energy`**: 전부 음수 가중치 — 관절을 거칠게/비효율적으로 쓰면 패널티. 실제 모터 마모, 전력 소모를 고려한 항목.
- **`joint_deviation_arms/waists/legs`**: 팔/허리/다리가 기본 자세에서 너무 벗어나면 패널티 — 걷는 동작이 부자연스러워지는 걸 방지.
- **`flat_orientation_l2` (-5.0), `base_height` (-10, target 0.78m)**: 몸통을 수평으로, 목표 높이(0.78m)로 유지.
- **`gait` (+0.5)**: 양발이 `period=0.8초` 주기로 서로 0.5만큼 위상차를 두고 번갈아 땅을 딛도록 유도 — "자연스러운 걸음걸이" 강제.
- **`feet_slide` (-0.2)**: 발이 땅에 닿은 상태에서 미끄러지면 패널티 (마찰/불안정 보행 방지).
- **`feet_clearance` (+1.0, target 0.1m)**: 발을 들 때 10cm 정도 띄우도록 유도 (발끝 끌림 방지).
- **`undesired_contacts` (-1)**: 발(ankle) 아닌 부위(무릎, 팔 등)가 땅에 닿으면 패널티.

→ "이 reward들의 합을 최대화하도록 PPO가 정책을 업데이트한다"가 강화학습 관점에서의 본질. 가중치 하나만 바꿔도 걸음걸이 특성이 달라짐 (예: `feet_clearance` weight를 낮추면 발을 덜 들고 걷는 정책이 나올 수 있음).

### Termination (종료 조건) — `TerminationsCfg`

- `time_out`: 에피소드 길이(`episode_length_s=20.0`초) 초과
- `base_height`: 몸통이 0.2m 이하로 내려감 (쓰러짐)
- `bad_orientation`: 기울어짐이 0.8 rad 한계 초과

### Curriculum (난이도 커리큘럼) — `CurriculumCfg`

- `terrain_levels_vel`: 잘 걸으면 지형 난이도를 점점 올림 (커리큘럼 러닝)
- `lin_vel_cmd_levels`: 잘 추종하면 속도 명령 범위를 점점 넓힘

처음부터 어려운 지형/빠른 속도를 요구하면 학습이 거의 항상 실패하기 때문에, 쉬운 것부터 시작해서 점점 어렵게 만드는 전략.

---

## PPO 하이퍼파라미터 — `rsl_rl_ppo_cfg.py`

| 파라미터 | 값 | 의미 |
|---|---|---|
| `num_steps_per_env` | 24 | 업데이트 1번 전에 환경마다 모으는 스텝 수. `2048 envs × 24 steps = 49,152` 샘플/iteration |
| `max_iterations` | 50000 (우리는 `--max_iterations 3000`으로 override) | 총 업데이트 횟수 |
| `clip_param` | 0.2 | PPO의 핵심 — 정책이 한 번에 너무 크게 바뀌지 않도록 확률비를 [0.8, 1.2]로 clip |
| `gamma` | 0.99 | 미래 보상 할인율 (1에 가까울수록 장기적 보상을 중시) |
| `lam` | 0.95 | GAE의 λ — advantage 추정의 bias-variance 트레이드오프 |
| `num_learning_epochs` | 5 | 모은 데이터를 몇 번 재사용해서 학습할지 |
| `num_mini_batches` | 4 | 한 epoch을 몇 개 미니배치로 나눌지 |
| `learning_rate` | 1e-3 (adaptive) | KL divergence(`desired_kl=0.01`)를 보면서 자동 조절 |
| `actor/critic_hidden_dims` | [512, 256, 128] | MLP 구조 (3-layer) |

`num_envs=2048`로 직접 줄인 이유: 기본값 4096은 VRAM 16GB(RTX 5060 Ti)에서 다른 프로세스와 겹치면 OOM 위험이 있어서, 안전 마진을 두고 절반으로 줄임.

---

## 실제 학습 결과 (이번 실행)

- 설정: `num_envs=2048`, `max_iterations=3000`, 소요 시간 1시간 19분
- Mean reward: 초반 약 -4.8 → 종료 시점 양수로 개선
- `Metrics/base_velocity/error_vel_xy`: 0.55 → 0.21 (속도 추종 오차 감소)
- `Metrics/base_velocity/error_vel_yaw`: 3.1 → 0.90 (회전 추종 오차 감소)
- `Episode_Reward/feet_clearance`: 0.04 → 0.74 (발 들어올리는 동작 형성)
- `Episode_Reward/gait`: 0.01 → 0.35 (걸음걸이 패턴 형성)

→ 숫자만으로도 "넘어지지 않고 명령된 속도로 걷는 정책"이 형성되었음을 알 수 있음. `play.py`로 실제 시각화하여 확인.

---

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
