# FRANK — Franka Reinforcement and Autonomy via Neural Kinematics

A reinforcement learning framework for training a Franka Panda robot arm to autonomously perform pick and place tasks in MuJoCo physics simulation using PPO (Proximal Policy Optimization).

---

## What is FRANK

FRANK is a from-scratch implementation of a robot learning pipeline built on top of MuJoCo and Stable-Baselines3. A neural network policy replaces classical Inverse Kinematics — instead of computing joint angles mathematically, the policy learns through trial and error in simulation to move the end effector to any target position within the robot's workspace.

The reach policy is then used as the motion backbone for a full pick and place system, extended in Phase 2 with vision-based object detection, and in Phase 3 with language-directed multi-object picking — the full Vision-Language-Action (VLA) pattern.

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

Training happened in two phases — first on a bare FR3 arm to develop the RL pipeline, then retrained on the full Panda arm with gripper for deployment.

**Stage 1 — FR3 reach experiments** (`models/reach/trained_v*/`) — pipeline development:

| Version | Key Change | Outcome |
|---------|------------|---------|
| v1 | Absolute joint angle actions | 0% success — action space too large |
| v2 | Delta actions (±0.05 rad) | 0% success — reward scale wrong |
| v3 | Curriculum learning added | 10% eval — first real learning |
| v4 | VecNormalize observation normalization | 30% success — stable training |
| v5_005 | Progressive reward, 0.05m threshold | 65% live success |
| v6 | Expanded workspace, 2M steps | Diverged after peak |
| v7_008 | Entropy coef 0.05 | 85% live success |
| v8_005 | Further tuning | Best FR3 result |

**Stage 2 — Panda reach policy** (`models/reach/panda/trained_v*/`) — with gripper and pinch site:

| Version | Key Change | Outcome |
|---------|------------|---------|
| v1–v4 | Porting FR3 lessons to Panda with gripper | Incremental improvement |
| v5 | Progressive reward, curriculum center at table height | **Active policy — used in all phases** |
| v6 | Higher curriculum center z=0.60 | Hovers above table, doesn't descend |
| v7 | Joint6 constraint during training | Arm stuck sideways |
| v8 | Orientation reward, no joint constraints | std explosion at 1M steps |
| v9 | Conditional orientation bonuses, soft constraints | 50% eval at 170k, std explosion at 1.1M |

The Panda v5 policy is used in all three phases of FRANK. v8 and v9 were attempts to learn top-down vertical approach via reward shaping — both failed due to PPO instability with mixed reward signals at scale.

Key lessons:
- **Observation normalization is mandatory** — without VecNormalize, inference fails completely
- **Curriculum center must match deployment height** — v6 failed because training z=0.60 was above the table
- **Delta actions beat absolute** — small corrective joint deltas are far easier to learn
- **Orientation reward destabilises PPO at scale** — additive orientation reward causes std explosion after ~600k steps regardless of weight
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

## Phase 3 — Vision-Language-Action (VLA)

### What it adds

Phase 3 introduces three cubes of different colours (red, blue, yellow) and a language interface. The user types a natural language instruction — "pick the red cube" — and the system identifies, localises, and picks only that cube, leaving the others untouched. All three cubes can be picked in sequence within a single session.

This is the full VLA pattern: **Vision** detects all cubes, **Language** selects the target, **Action** (reach policy) executes the pick.

### Architecture

```
Session start
    │
    ├─ Three cubes spawned in non-overlapping zones
    ├─ Fixed tray placed at target position
    │
    For each pick:
    │
    ├─ User types instruction in OpenCV window ("pick the blue cube")
    ├─ Language parser extracts target colour
    ├─ Arm retracts to photo pose
    ├─ YOLOv8n detects all cubes (single class: 'cube')
    ├─ HSV classifier identifies colour of each detected box
    ├─ Target cube position locked from matching detection
    ├─ Arm returns to home, reach policy executes pick and place
    ├─ Arm returns to home, waits for next instruction
    └─ Session ends when all three cubes are placed
```

### Why single-class YOLO + HSV classifier

YOLO is trained on cube shape only (nc=1). Colour classification is handled separately by an OpenCV HSV classifier on the bounding box crop. This means adding a new colour requires no YOLO retraining — only a new HSV hue range.

### VLA YOLO Training Results

| Metric | Value |
|--------|-------|
| Model | YOLOv8n (fine-tuned) |
| Training data | 800 frames, 3 cubes per frame (2400 instances) |
| Epochs to convergence | 19 of 50 |
| mAP50 | 0.995 |
| mAP50-95 | 0.858 |
| Precision | 1.0 |
| Recall | 1.0 |

### Modular Code Structure

```
environment/
└── pick_place_scripted_env_vla.py   ← orchestrator (~200 lines)

scripts/vla/
├── run_vla.py                       ← session entry point
├── generate_training_data_vla.py    ← synthetic data generation
├── train_yolo_vla.py                ← YOLO training
└── modules/
    ├── vision.py                    ← YOLO detection, HSV classifier, pixel-to-world
    ├── spawner.py                   ← grid-based cube spawning
    └── stages.py                    ← 6-stage pick and place loop
```

### Phase 3 Demo

https://github.com/nihalseth0506/project_frank/assets/frank_phase3_vla_pick_place.mp4

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

**Generate Phase 2 YOLO training data:**
```bash
python scripts/generate_training_data.py
```

**Train Phase 2 YOLO detector:**
```bash
python scripts/train_yolo.py
```

**Generate Phase 3 VLA YOLO training data:**
```bash
python scripts/vla/generate_training_data_vla.py
```

**Train Phase 3 VLA YOLO detector:**
```bash
python scripts/vla/train_yolo_vla.py
```

**Run Phase 1 pick and place:**
```bash
python scripts/run_pick_place_scripted.py --episodes 10
python scripts/run_pick_place_scripted.py --episodes 10 --no-render
```

**Run Phase 3 VLA session:**
```bash
python scripts/vla/run_vla.py
```
The OpenCV window opens — adjust the MuJoCo viewer angle, press any key, then type colour instructions to pick each cube.

---

## Skills Demonstrated

- Custom Gymnasium environment from scratch with curriculum learning
- PPO training with Stable-Baselines3 — hyperparameter tuning, reward shaping, callback design
- Debugging RL-specific failures — entropy collapse, reward hacking, curriculum distribution mismatch, orientation reward instability
- MuJoCo physics engine — direct API usage, contact dynamics, actuator tuning
- Multi-stage scripted + policy hybrid control pipeline
- Synthetic dataset generation — auto-labelled frames from simulation, no manual annotation
- YOLOv8 fine-tuning on synthetic data — mAP50 0.995, 0.2mm localization accuracy
- HSV colour classification — separates colour identification from shape detection
- Pixel-to-world coordinate transform via perspective projection
- Vision-Language-Action architecture — modular perception, language parsing, and policy execution
- Observation normalization with synchronized checkpoint saving
- Modular software design — vision, spawning, and stage logic separated into independent modules

---

## Dependencies

```
mujoco>=3.9.0
stable-baselines3>=2.8.0
gymnasium>=1.2.3
ultralytics>=8.4.41
numpy>=2.2.6
torch>=2.5.1
opencv-python>=4.13.0

# PyTorch — install CUDA version manually for GPU support:
# pip install torch>=2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

---

## References

- [MuJoCo Documentation](https://mujoco.readthedocs.io)
- [MuJoCo Menagerie — Franka Panda](https://github.com/google-deepmind/mujoco_menagerie)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io)
- [Ultralytics YOLOv8](https://docs.ultralytics.com)
- [Proximal Policy Optimization — Schulman et al. 2017](https://arxiv.org/abs/1707.06347)
