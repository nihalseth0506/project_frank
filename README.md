# FRANK — Franka Reinforcement and Autonomy via Neural Kinematics

A reinforcement learning framework for training a Franka FR3 robot arm to autonomously reach target positions in 3D space using MuJoCo physics simulation and PPO (Proximal Policy Optimization).

---

## What is FRANK

FRANK is a from-scratch implementation of a robot learning pipeline built on top of MuJoCo and Stable-Baselines3. The goal is to train a neural network policy that replaces classical Inverse Kinematics — instead of computing joint angles mathematically, the policy learns through trial and error in simulation to move the end effector to any target position within the robot's workspace.

This project covers the full RL pipeline: custom Gymnasium environment, curriculum learning, observation normalization, reward shaping, policy evaluation, and inference — all built modularly so each component can be extended independently.

---

## Current Capabilities

- Franka FR3 7-DOF arm simulated in MuJoCo with accurate joint limits and physics
- PPO-trained reach policy achieving **85%+ live success rate** at 0.08m goal threshold
- Curriculum learning — target difficulty increases from 0.1m radius to 0.4m workspace over training
- Progressive reward shaping — stronger signal as end effector approaches target
- Observation normalization via VecNormalize — stable training across differently scaled inputs
- Checkpoint saving synchronized with normalization statistics — policy always deployable
- Specific target testing — probe policy capability at arbitrary 3D positions

---

## Architecture

```
Observation (20 values)
    7 joint positions
    7 joint velocities
    3 end effector xyz
    3 target xyz
         ↓
Neural Network Policy
    Input layer:   20 neurons
    Hidden layer:  256 neurons (ReLU)
    Hidden layer:  256 neurons (ReLU)
    Output layer:  7 neurons
         ↓
Action (7 joint deltas)
    max ±0.1 rad per joint per step
    clipped to joint limits
         ↓
MuJoCo Physics
    0.002s timestep
    joint angles update
    end effector position computed
         ↓
Reward
    -distance to target
    +0.5  if distance < 0.20m
    +2.0  if distance < 0.10m
    +5.0  if distance < 0.05m
    +10.0 on success (distance < 0.08m)
```

---

## Project Structure

```
project_frank/
├── config/
│   └── robot_config.py       # all constants — model path, joint limits,
│                             # curriculum settings, reward parameters
├── environment/
│   └── reach_env.py          # custom Gymnasium environment
│                             # reset, step, reward, curriculum, observation
├── robot/
│   ├── controller.py         # manual robot controller (pre-RL foundation)
│   └── observer.py           # state reading — joints, end effector position
├── scripts/
│   ├── train.py              # PPO training with callbacks and normalization
│   ├── run_policy.py         # load and run trained policy with visualization
│   └── run_robot_manual.py   # manual joint control for testing
├── tests/
│   ├── test_env.py           # environment verification
│   ├── test_mujoco.py        # basic MuJoCo setup
│   ├── test_mujoco_2.py      # cube drop physics test
│   ├── test.py               # general scratch testing
│   └── robot_controller_v0_monolithic.py  # original single-file reference
├── models/
│   ├── trained/              # current active training run
│   │   └── best/             # best policy checkpoint
│   │       ├── best_model.zip
│   │       └── vec_normalize.pkl
│   └── trained_v*/           # archived training runs
├── logs/                     # TensorBoard training logs
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Installation

**Requirements:**
- Python 3.10 or 3.11
- Windows 10/11 or Ubuntu 22.04
- NVIDIA GPU recommended (CPU works for this scale)

**Install dependencies:**

```bash
git clone https://github.com/YOUR_USERNAME/project_frank.git
cd project_frank
pip install -r requirements.txt
```

**Download robot models:**

Download MuJoCo Menagerie from:
```
https://github.com/google-deepmind/mujoco_menagerie/archive/refs/heads/main.zip
```
Extract to `models/mujoco_menagerie-main/`

---

## Usage

**Train a policy:**
```bash
python scripts/train.py
```

**Run trained policy with visualization:**
```bash
python scripts/run_policy.py
```

**Test specific target positions:**

In `scripts/run_policy.py`, modify the `__main__` block:
```python
test_specific_target(model_path, norm_path, [0.35, 0.0,  0.45])  # center
test_specific_target(model_path, norm_path, [0.45, 0.15, 0.50])  # right and up
test_specific_target(model_path, norm_path, [0.25, -0.1, 0.35])  # left and low
```

**Visualize training curves:**
```bash
tensorboard --logdir logs/
```

---

## Training Configuration

Key parameters in `config/robot_config.py`:

```python
MAX_EPISODE_STEPS        = 1000    # steps per episode
GOAL_THRESHOLD           = 0.08    # success distance in meters
CURRICULUM_START_RADIUS  = 0.1     # initial target spawn radius
CURRICULUM_END_RADIUS    = 0.4     # final target spawn radius
CURRICULUM_STEPS         = 500_000 # steps to reach full difficulty
```

Key parameters in `scripts/train.py`:

```python
learning_rate   = 1e-4       # conservative — prevents divergence
n_steps         = 2048       # rollout buffer size
batch_size      = 64         # gradient update batch size
n_epochs        = 10         # passes through each rollout
ent_coef        = 0.05       # entropy coefficient — maintains exploration
net_arch        = [256, 256] # two hidden layers of 256 neurons
total_timesteps = 1_000_000
```

---

## Training Progression

Seven training runs were conducted to reach the current result. Each run identified and addressed a specific limitation:

| Version | Key Change | Result | Limitation |
|---------|------------|--------|------------|
| v1 | Absolute joint angle actions | 0% success | Action space too large to learn |
| v2 | Delta actions (±0.05 rad) | 0% success | Reward scale inflated numbers |
| v3 | Lower learning rate, curriculum | 10% eval, diverged | Diverged after peak — learning rate too high |
| v4 | VecNormalize observation normalization | 30% eval success | First real learning — eval success dropped after peak |
| v5 | Threshold 0.15m | 100% success | Reward hacking — agent parked at boundary, never truly reached target |
| v6 | Threshold 0.08m, progressive reward | 85% live success, positive ep_rew_mean | Agent still parks at threshold, left-low workspace uncovered |
| v7 | Instant reset, ent_coef 0.05 | 85% live success, 40% at 0.15m threshold | Left-low workspace region outside training distribution |
| v8 | Expanded workspace, 0.05m threshold, 2M steps | 65% live success, all test targets passed | Diverged after 1.06M peak — early stopping needed |

**Key lessons learned:**

- **Entropy collapse** — agent stops exploring and locks onto a mediocre strategy. Fixed by increasing `ent_coef` from 0.01 to 0.05
- **Reward hacking** — agent learns to stop exactly at the success threshold rather than reaching the target. Fixed by progressive reward bonuses that pull the agent past the threshold
- **Observation normalization** — without VecNormalize the policy completely failed at inference. The pkl file must always be saved and loaded alongside the model weights
- **Delta vs absolute actions** — absolute joint angle actions require the agent to learn the full robot configuration space. Delta actions only require learning small corrective movements — much easier to learn
- **pkl synchronization** — model weights and normalization statistics must always be saved from the same checkpoint. A mismatch causes complete inference failure

---

## Results

| Metric | Value |
| Best eval success rate       | 90% at step 1,065,000    |
| Live success rate (0.05m)    | 65%                       |
| Average episode length at peak | 403 steps               |
| All specific targets passed  | 3/3                       |

**Specific target test results:**

| Target | Location | Result | Final Distance |
|--------|----------|--------|----------------|
| [0.35, 0.0, 0.45] | Center of workspace | SUCCESS | 0.0798m |
| [0.45, 0.15, 0.50] | Right and up | SUCCESS | 0.0787m |
| [0.25, -0.1, 0.35] | Left and low | FAILED | 0.3333m |

The failed case identifies the boundary of the current training distribution. The left-low region of the workspace was underrepresented during training and is the primary target for v8.

---

## Roadmap

```
Phase 1 — Reach (current)
    ✅ FR3 model in MuJoCo
    ✅ Custom Gymnasium environment
    ✅ PPO training pipeline
    ✅ Curriculum learning
    ✅ 85% success rate at 0.08m threshold
    🔲 v8 — expanded workspace coverage including left-low region
    🔲 Full workspace generalization

Phase 2 — Pick and Place
    🔲 Object spawning in scene
    🔲 Grasp reward shaping
    🔲 Place target with contact detection

Phase 3 — Vision Integration
    🔲 Camera sensor on FR3 end effector
    🔲 CNN visual encoder
    🔲 Vision-based observation combining image and joint state
    🔲 YOLO object detection integration

Phase 4 — ROS2 Bridge
    🔲 Joint state publisher from MuJoCo to ROS2
    🔲 Policy action subscriber
    🔲 Real robot deployment interface
    🔲 UR10e and Diana 7 compatibility
```

---

## Skills Demonstrated

- Custom Gymnasium environment implementation from scratch
- PPO training with Stable-Baselines3 — hyperparameter tuning and callback design
- Debugging RL-specific failures — entropy collapse, reward hacking, curriculum mismatch
- MuJoCo physics engine — direct API usage (MjModel, MjData, mj_step, mj_forward)
- Observation normalization — VecNormalize with synchronized pkl checkpoint saving
- Modular software architecture — separation of config, environment, robot, and scripts
- Experiment tracking with git — meaningful commit history documenting research progression
- Franka FR3 robot — joint limits, workspace geometry, MuJoCo Menagerie model

---

## Dependencies

```
mujoco>=3.0.0
stable-baselines3>=2.0.0
gymnasium>=0.29.0
numpy>=1.24.0
matplotlib>=3.7.0
torch>=2.0.0
```

---

## References

- [MuJoCo Documentation](https://mujoco.readthedocs.io)
- [MuJoCo Menagerie — Franka FR3](https://github.com/google-deepmind/mujoco_menagerie)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io)
- [Proximal Policy Optimization — Schulman et al. 2017](https://arxiv.org/abs/1707.06347)
- [Gymnasium Documentation](https://gymnasium.farama.org)