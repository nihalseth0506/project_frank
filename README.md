# FRANK — Franka Reinforcement and Autonomy via Neural Kinematics

A reinforcement learning framework for training a Franka Panda robot arm to autonomously perform pick and place tasks in MuJoCo physics simulation using PPO (Proximal Policy Optimization).

---

## What is FRANK

FRANK is a from-scratch implementation of a robot learning pipeline built on top of MuJoCo and Stable-Baselines3. A neural network policy replaces classical Inverse Kinematics — instead of computing joint angles mathematically, the policy learns through trial and error in simulation to move the end effector to any target position within the robot's workspace.

The reach policy is then used as the motion backbone for a full pick and place system, extended in Phase 2 with vision-based object detection — the arm no longer reads the cube position from simulation state but detects it from an overhead camera using a YOLO model trained entirely on synthetic data.

---

## Phase 1 — RL Pick and Place

### What it does

- Franka Panda 7-DOF arm simulated in MuJoCo with accurate joint limits and physics
- PPO-trained reach policy driving all arm motion — approach, descent, lift, and transport
- Curriculum learning — target difficulty increases from 8cm radius to 30cm workspace over training
- Observation normalization via VecNormalize — stable training and inference
- Full pick and place pipeline: approach → descend → grasp → lift → transport → place
- **Up to 100% success rate** on pick and place with constrained spawn and target zones

### Architecture

The reach policy takes a 20-value observation and outputs 7 joint deltas:

```
Observation (20 values)
    7  joint positions
    7  joint velocities
    3  pinch site xyz (fingertip position in world frame)
    3  target xyz
         ↓
PPO Policy  [256 → 256 → 7]
         ↓
Action: 7 joint deltas (±0.05 rad, clipped to joint limits)
         ↓
MuJoCo Physics (5 substeps per action)
         ↓
Reward
    -2.0 × distance
    +0.5  if distance < 0.20m
    +2.0  if distance < 0.10m
    +5.0  if distance < 0.05m
    +10.0 on success (distance < 0.04m)
```

### Pick and Place Pipeline

```
Stage 0  APPROACH    Reach policy → above cube (8cm)
Stage 1  DESCEND     Reach policy → cube position, incremental step-down
Stage 2  GRASP       Scripted gripper close, arm frozen
Stage 3  LIFT        Reach policy → above cube at lift height
Stage 4  TRANSPORT   Reach policy → above tray target
Stage 5  RELEASE     Scripted gripper open, cube settles into tray
```

The policy drives all arm motion. Only the gripper open/close commands are scripted.

### Training History

Seven policy versions were trained before reaching the final result:

| Version | Key Change | Outcome |
|---------|------------|---------|
| v1–v3 | Delta actions, reward scaling, curriculum | First real learning, 10–30% eval |
| v4 | VecNormalize observation normalization | Stable training, 30% success |
| v5 | Progressive reward, lower threshold | **70% eval** — best reach policy, used in pick and place |
| v6 | Higher curriculum center z=0.60 | Policy hovers above table, doesn't descend |
| v7 | Joint6 constraint during training | Arm stuck sideways, worse than v5 |

Key lessons:
- **Observation normalization is mandatory** — without VecNormalize, inference fails completely
- **Curriculum center must match deployment height** — v6 failed because training z=0.60 was above the table
- **Delta actions beat absolute** — small corrective joint deltas are far easier to learn
- **Training distribution must match deployment** — the single most important lesson across all versions

### Phase 1 Results

The v5 reach policy achieves approximately 70% success on unconstrained reach targets across the full workspace.

The pick and place system achieves **up to 100% success rate** (20/20 episodes) with:
- Cube spawned randomly in x=[0.50, 0.60], y=[-0.02, 0.10]
- Target tray placed randomly in x=[0.45, 0.55], y=[-0.25, -0.15]
- Success defined as cube landing inside the tray (within 9cm of target center)

### Phase 1 Demo

https://github.com/nihalseth0506/project_frank/assets/frank_pick_place_demo.mp4

---

## Phase 2 — Vision-Based Object Detection

### What it adds

Phase 2 removes the dependency on ground-truth simulation state. Instead of reading the cube's position directly from the physics engine, the arm detects it from an overhead camera using a YOLO model trained entirely on synthetic MuJoCo frames.

The pick and place pipeline is unchanged — only the source of the cube coordinates changes.

### Vision Pipeline

```
Episode start
    │
    ├─ Arm moves to retracted photo pose (out of camera frame)
    ├─ Overhead camera renders frame (640×480)
    ├─ YOLOv8n detects cube → bounding box center pixel (u, v)
    ├─ Pixel-to-world transform → world coordinates (x, y)
    │       scale = tan(fovy/2) × camera_height_above_table
    │       wx = cam_x + (u_norm × scale × aspect)
    │       wy = cam_y - (v_norm × scale)
    ├─ object_pos = (x, y, TABLE_Z + 0.025) ← locked for episode
    └─ Arm restores to home pose, pick and place executes normally
```

### Synthetic Data Generation

800 training + 200 validation frames generated automatically:
- Cube spawned at random positions and random z-rotations
- Arm retracted to photo pose so cube is fully visible
- YOLO labels auto-generated from known world position — no manual labelling
- `pixel_to_world` transform inverted to get ground-truth bounding box center

### YOLO Training Results

| Metric | Value |
|--------|-------|
| Model | YOLOv8n (fine-tuned) |
| Training data | 800 synthetic MuJoCo frames |
| Epochs to convergence | 16 of 50 |
| mAP50 | 0.995 |
| mAP50-95 | 0.843 |
| Precision | 1.0 |
| Recall | 1.0 |
| Localization error | 0.2mm |

### Phase 2 Demo

https://github.com/nihalseth0506/project_frank/assets/frank_phase2_vision_pick_place.mp4

---

## Getting Started

**Requirements:**
- Python 3.10 or 3.11
- Windows 10/11 or Ubuntu 22.04

**Install:**
```bash
git clone https://github.com/nihalseth0506/project_frank.git
cd project_frank
pip install -r requirements.txt
```

**Download Franka Panda model from MuJoCo Menagerie:**
```
https://github.com/google-deepmind/mujoco_menagerie
```
Extract to `models/mujoco_menagerie-main/`

**Train a reach policy:**
```bash
python scripts/train_reach_panda.py
```

**Generate YOLO training data:**
```bash
python scripts/generate_training_data.py
```

**Train YOLO detector:**
```bash
python scripts/train_yolo.py
```

**Run Phase 1 pick and place (ground truth):**
```bash
python scripts/run_pick_place_scripted.py --episodes 10
python scripts/run_pick_place_scripted.py --episodes 10 --no-render
```

**Run Phase 2 pick and place (vision):**
```bash
python scripts/run_pick_place_scripted.py --episodes 10
```
Phase 2 is the default — YOLO detection runs automatically if the model exists.

---

## Skills Demonstrated

- Custom Gymnasium environment from scratch with curriculum learning
- PPO training with Stable-Baselines3 — hyperparameter tuning, reward shaping, callback design
- Debugging RL-specific failures — entropy collapse, reward hacking, curriculum distribution mismatch
- MuJoCo physics engine — direct API usage, contact dynamics, actuator tuning
- Multi-stage scripted + policy hybrid control pipeline
- Synthetic dataset generation — 1000 auto-labelled frames from simulation
- YOLOv8 fine-tuning on synthetic data — mAP50 0.995, 0.2mm localization accuracy
- Pixel-to-world coordinate transform via perspective projection
- Observation normalization with synchronized checkpoint saving

---

## What's Next — Phase 3

Phase 3 will introduce multi-object scenes and language-directed picking:
- Multiple cubes of different colours on the table
- Language instruction: "pick the red cube" / "place the blue cube in the tray"
- YOLO trained with one class per colour
- Language model parses instruction → selects target detection by class
- Same reach policy executes the motion unchanged

This is the Vision-Language-Action (VLA) pattern — vision detects, language selects, policy acts.

---

## Dependencies

```
mujoco>=3.0.0
stable-baselines3>=2.0.0
gymnasium>=0.29.0
ultralytics>=8.0.0
numpy>=1.24.0
torch>=2.0.0
opencv-python>=4.8.0
```

---

## References

- [MuJoCo Documentation](https://mujoco.readthedocs.io)
- [MuJoCo Menagerie — Franka Panda](https://github.com/google-deepmind/mujoco_menagerie)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io)
- [Ultralytics YOLOv8](https://docs.ultralytics.com)
- [Proximal Policy Optimization — Schulman et al. 2017](https://arxiv.org/abs/1707.06347)