# Highway-Env Training Guide for DreamerV3-Torch

This guide explains how to train autonomous driving agents using [highway-env](https://highway-env.farama.org/) environments with DreamerV3.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Available Environments](#available-environments)
- [Training](#training)
- [Configuration Options](#configuration-options)
- [Monitoring Training](#monitoring-training)
- [Troubleshooting](#troubleshooting)

## Overview

Highway-env is a collection of environments for autonomous driving and tactical decision-making. This integration allows you to train DreamerV3 world models on various driving scenarios using image-based observations.

### Supported Environments

| Environment | Description | Action Type | Vehicle Config |
|-------------|-------------|-------------|----------------|
| `highway` | Highway driving with lane changes | Discrete | `vehicles_count`, `vehicles_density` |
| `intersection` | Navigate through an intersection | Discrete | `initial_vehicle_count` (max 20) |
| `parking` | Park in a designated spot | Continuous | No other vehicles |
| `merge` | Merge onto a highway | Discrete | Fixed (vehicles from ramp) |
| `roundabout` | Navigate a roundabout | Discrete | Dynamic spawning |
| `racetrack` | Race on a curved track | Continuous | `other_vehicles` (max 10) |

### Action Types

#### Discrete Actions (`DiscreteMetaAction`)
High-level, predefined actions. The agent chooses from **5 discrete options**:

| Action | ID | Description |
|--------|-----|-------------|
| `LANE_LEFT` | 0 | Change to left lane |
| `IDLE` | 1 | Stay in current lane, maintain speed |
| `LANE_RIGHT` | 2 | Change to right lane |
| `FASTER` | 3 | Accelerate |
| `SLOWER` | 4 | Decelerate |

**Best for:** Highway, intersection, merge, roundabout (lane-based decisions)

#### Continuous Actions (`ContinuousAction`)
Low-level, direct control. The agent outputs **2 continuous values**:

| Parameter | Range | Description |
|-----------|-------|-------------|
| `acceleration` | [-1, 1] | Throttle/brake (-1 = full brake, +1 = full throttle) |
| `steering` | [-1, 1] | Steering angle (-1 = full left, +1 = full right) |

**Best for:** Parking, racetrack (precise maneuvering needed)

## Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)
- Windows, Linux, or macOS

### Setup

1. **Create and activate a virtual environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # Windows PowerShell
   # or
   source .venv/bin/activate      # Linux/macOS
   ```

2. **Install PyTorch with CUDA support:**
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   pip install gymnasium highway-env pygame
   ```

4. **Install compatible numpy (required):**
   ```powershell
   pip install "numpy>=1.23.5,<2.0"
   ```

## Available Environments

### Highway (`highway`)
Standard highway driving scenario. The agent must navigate through traffic, change lanes, and maintain speed.

```powershell
python dreamer.py --configs highway --logdir ./logdir/highway --device cuda:0
```

### Intersection (`intersection`)
The agent must cross an intersection while avoiding collisions with other vehicles.

```powershell
python dreamer.py --configs intersection --logdir ./logdir/intersection --device cuda:0
```

### Parking (`parking`)
Goal-conditioned parking task. The agent must park in a designated spot using continuous steering and acceleration.

```powershell
python dreamer.py --configs parking --logdir ./logdir/parking --device cuda:0
```

### Merge (`merge`)
The agent must merge onto a highway from an on-ramp while avoiding collisions.

```powershell
python dreamer.py --configs merge --logdir ./logdir/merge --device cuda:0
```

### Roundabout (`roundabout`)
Navigate through a roundabout with multiple entry and exit points.

```powershell
python dreamer.py --configs roundabout --logdir ./logdir/roundabout --device cuda:0
```

### Racetrack (`racetrack`)
Race on a curved track using continuous control for smooth steering.

```powershell
python dreamer.py --configs racetrack --logdir ./logdir/racetrack --device cuda:0
```

## Training

### Basic Training Command

```powershell
python dreamer.py --configs <environment> --logdir <path> --device cuda:0
```

### Examples

**Train on highway environment:**
```powershell
python dreamer.py --configs highway --logdir E:\MyWork\dreamerv3-torch\logdir\highway --device cuda:0
```

**Train with continuous actions:**
```powershell
python dreamer.py --configs highway_continuous --logdir ./logdir/highway_continuous --device cuda:0
```

**Resume training from checkpoint:**
```powershell
python dreamer.py --configs highway --logdir ./logdir/highway --device cuda:0
# (automatically resumes if checkpoint exists)
```

### Training Time Estimates

With default settings (RTX 4060 or similar):
- ~1-3 hours for 100k steps
- Evaluations every 5k steps
- Logs every 1k steps

## Configuration Options

### Pre-configured Variants

| Config Name | Observation | Action | Use Case |
|-------------|-------------|--------|----------|
| `highway` | Image (64x64 RGB) | Discrete | Standard training |
| `highway_kinematics` | Kinematics vectors | Discrete | Faster training, no images |
| `highway_continuous` | Image (64x64 RGB) | Continuous | Fine-grained control |

### Vehicle Density Configuration

Control the number of vehicles in environments:

```powershell
# Highway with 50 vehicles (default)
python dreamer.py --configs highway --highway_vehicles_count 50 --device cuda:0

# Fewer vehicles for easier learning
python dreamer.py --configs highway --highway_vehicles_count 10 --device cuda:0

# More dense traffic
python dreamer.py --configs highway --highway_vehicles_count 80 --highway_vehicles_density 2.0 --device cuda:0
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `highway_vehicles_count` | 50 | Number of vehicles |
| `highway_vehicles_density` | 1.5 | Density of vehicles on road |

**Note:** Different environments use these parameters differently:
- **Highway**: Direct `vehicles_count` and `vehicles_density`
- **Intersection**: Converted to `initial_vehicle_count` (capped at 20)
- **Racetrack**: Converted to `other_vehicles` (capped at 10)
- **Merge/Roundabout**: Fixed vehicle spawning (not configurable)
- **Parking**: No other vehicles

### Reward Shaping

The highway environments include comprehensive reward shaping for improved learning. This can be customized in `configs.yaml`:

```yaml
highway:
  # ... other configs ...
  highway_reward_shaping: True  # Enable/disable reward shaping
  highway_reward_config:
    # Speed rewards
    high_speed_reward: 0.4        # Reward for maintaining high speed
    reward_speed_range: [25, 30]  # Optimal speed range [min, max] m/s
    
    # Safety penalties
    collision_reward: -1.0        # Strong penalty for collision
    safe_distance_reward: 0.1     # Reward for keeping safe distance
    min_safe_distance: 15.0       # Minimum safe distance (meters)
    safe_distance_penalty: 0.3    # Penalty for being too close
    
    # Lane behavior - Smart lane changing
    right_lane_reward: 0.05       # Small reward for rightmost lane
    lane_change_reward: 0.0       # No penalty for lane changes
    smart_lane_change_reward: 0.3 # Reward for overtaking slow vehicles
    blocked_lane_penalty: 0.2     # Penalty for staying behind slow vehicle
    clear_lane_reward: 0.15       # Reward for being in a clear lane
    look_ahead_distance: 50.0     # Distance to look ahead for obstacles
    slow_vehicle_threshold: 0.7   # Vehicle is "slow" if < 70% of ego speed
    
    # Road adherence
    on_road_reward: 0.1           # Reward for staying on road
    off_road_penalty: -0.5        # Penalty for going off-road
    
    # Progress and alignment
    heading_reward: 0.1           # Reward for proper road alignment
    progress_reward_scale: 0.01   # Scale for forward progress reward
    
    # Survival
    survival_reward: 0.01         # Small constant reward for staying alive
    success_reward: 0.5           # Bonus for completing episode without crash
    
    # Blending factor (0-1)
    shaped_reward_weight: 0.8     # 0.8 = 80% shaped, 20% original reward
```

#### Reward Components Explained

1. **Speed Reward**: Encourages maintaining optimal driving speed within the specified range
2. **Collision Penalty**: Strong negative reward when crashing
3. **Safe Distance Reward**: Rewards maintaining safe following distance from other vehicles
4. **Smart Lane Change Reward**: Rewards changing lanes to overtake slower vehicles
5. **Blocked Lane Penalty**: Penalizes staying behind a slow vehicle when overtaking is possible
6. **Clear Lane Reward**: Rewards being in a lane with no obstacles ahead
7. **Right Lane Reward**: Small encouragement to stay in rightmost lane (reduced to allow overtaking)
8. **On-Road Reward**: Rewards staying on the road surface
9. **Heading Reward**: Rewards proper alignment with road direction
10. **Progress Reward**: Rewards forward movement
11. **Survival Reward**: Small constant reward for each step survived
12. **Success Reward**: Bonus for completing episode without crashing

#### Disabling Reward Shaping

To use the original highway-env rewards:

```yaml
highway:
  highway_reward_shaping: False
```

Or via command line:
```powershell
python dreamer.py --configs highway --highway_reward_shaping False --logdir ./logdir/highway_original
```

### Custom Configuration

You can override config values via command line:

```powershell
# Train for more steps
python dreamer.py --configs highway --steps 5e5 --logdir ./logdir/highway_long

# Change training ratio (fewer gradient steps per environment step)
python dreamer.py --configs highway --train_ratio 32 --logdir ./logdir/highway_fast

# Adjust evaluation frequency
python dreamer.py --configs highway --eval_every 1e4 --logdir ./logdir/highway
```

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `steps` | 1e5 | Total training steps |
| `train_ratio` | 64 | Gradient steps per env step |
| `eval_every` | 5e3 | Evaluation frequency |
| `log_every` | 1e3 | Logging frequency |
| `time_limit` | 200 | Max steps per episode |
| `size` | [64, 64] | Image observation size |
| `highway_vehicles_count` | 50 | Number of vehicles in environment |
| `highway_vehicles_density` | 1.5 | Vehicle density on road |
| `highway_reward_shaping` | True | Enable reward shaping |
| `highway_action_type` | discrete | Action type (discrete/continuous) |

## Monitoring Training

### TensorBoard

Training logs are saved to the logdir. View with TensorBoard:

```powershell
tensorboard --logdir ./logdir
```

Then open http://localhost:6006 in your browser.

### Key Metrics to Monitor

- **episode/score**: Total episode reward
- **episode/length**: Episode length
- **agent/policy_entropy**: Exploration metric
- **agent/model_loss**: World model loss

### Checkpoints

Checkpoints are automatically saved to the logdir:
- `latest.pt`: Most recent checkpoint
- Training automatically resumes from checkpoint if it exists

## Troubleshooting

### Common Issues

#### 1. "Unknown observation type" error
This has been fixed in the wrapper. Ensure you're using the latest `envs/highway.py`.

#### 2. Pygame window not appearing
The environment uses `render_mode="rgb_array"` and displays via pygame. Make sure pygame is installed:
```powershell
pip install pygame
```

#### 3. CUDA out of memory
Reduce batch size or train_ratio:
```powershell
python dreamer.py --configs highway --batch_size 8 --train_ratio 32 --logdir ./logdir/highway
```

#### 4. Slow training
- Use `train_ratio: 64` instead of `512` for faster iterations
- Ensure CUDA is being used (`--device cuda:0`)
- Check GPU utilization with `nvidia-smi`

#### 5. numpy compatibility issues
Install compatible numpy version:
```powershell
pip install "numpy>=1.23.5,<2.0"
```

### Windows-Specific Notes

- The codebase has been patched for Windows compatibility
- `torch.compile` is disabled on Windows (not fully supported)
- Process termination uses `terminate()` instead of `os.kill()`

## File Structure

```
dreamerv3-torch/
├── dreamer.py          # Main training script
├── configs.yaml        # All environment configurations
├── envs/
│   ├── highway.py      # Highway-env wrapper
│   └── wrappers.py     # Common environment wrappers
├── models.py           # DreamerV3 world model
├── networks.py         # Neural network architectures
└── tools.py            # Training utilities
```

## References

- [Highway-env Documentation](https://highway-env.farama.org/)
- [DreamerV3 Paper](https://arxiv.org/abs/2301.04104)
- [Original DreamerV3 Implementation](https://github.com/danijar/dreamerv3)
