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

| Environment | Description | Action Type |
|-------------|-------------|-------------|
| `highway` | Highway driving with lane changes | Discrete |
| `intersection` | Navigate through an intersection | Discrete |
| `parking` | Park in a designated spot | Continuous |
| `merge` | Merge onto a highway | Discrete |
| `roundabout` | Navigate a roundabout | Discrete |
| `racetrack` | Race on a curved track | Continuous |

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
