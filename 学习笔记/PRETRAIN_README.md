# Auxiliary Head Pretraining - Quick Start Guide

## Overview
This pretraining pipeline trains the ResNet+GRU vision encoder to predict drone state from FPV images **before** the main SAC training. This gives the network better initialization and faster convergence.

## Files Created
1. **`pretrain_aux.py`** - Main pretraining script
2. **`example_load_pretrained.py`** - Example showing how to use pretrained weights

## What Gets Pretrained
- **ResNet auxiliary head**: Predicts relative position (3D) + rotation (6D continuous representation)
- **GRU auxiliary head**: Predicts relative velocity (3D) + angular velocity (3D)

These are the **exact same** auxiliary heads used in your main SAC training (sac.py).

## Data Flow (100% Compatible with Main Pipeline)

```
AirSim Environment
    ↓
env.get_img_sequence() → collects 4 frames with 0.025s interval
    ↓
img_tensor: (1, 4, 3, 256, 144)          ← FPV RGB images
relative_next_target_pos: (4, 3) ÷ 10   ← position to next gate
attitude_9d: (4, 3, 3)                   ← rotation matrix
relative_next_target_vel: (4, 3) ÷ 10   ← velocity relative to target
fpv_angular_vel: (4, 3)                  ← angular velocity
    ↓
Buffer stores last frame's labels: pos[-1], rot[-1], vel[-1], ang_vel[-1]
    ↓
GRU(img_seq) → resnet_aux, gru_aux predictions
    ↓
MSE loss vs ground truth → backprop
```

## Quick Start

### Step 1: Run Pretraining
```powershell
cd path/to/SACfD
python pretrain_aux.py
```

**What happens:**
- Flies drone with random actions for 500 episodes
- Collects 10,000 samples in buffer
- Trains vision encoder with supervised learning
- Saves checkpoints to `pretrained_models/aux_pretrain_TIMESTAMP/`

**Monitor progress:**
- TensorBoard: `tensorboard --logdir=runs`
- Loss curves for position, rotation, velocity, angular velocity

### Step 2: Load Pretrained Weights in Main Training

In your `main.py`, after creating the SAC agent:

```python
from pretrain_aux import load_pretrained_weights_to_policy

# Create agent
agent = SAC(args)

# Load pretrained weights
agent.policy = load_pretrained_weights_to_policy(
    agent.policy,
    'pretrained_models/aux_pretrain_XXXXXX/best_aux_model.pt',
    device
)

# Continue with normal training
```

### Step 3: Train SAC as Usual
The vision encoder now has good initialization! The policy MLP will still be randomly initialized.

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PRETRAIN_EPISODES` | 500 | Number of random flight episodes |
| `PRETRAIN_BATCH_SIZE` | 32 | Batch size for training |
| `PRETRAIN_LEARNING_RATE` | 1e-3 | Learning rate (higher than main training) |
| `PRETRAIN_EPOCHS_PER_EPISODE` | 5 | Train multiple times on collected data |
| `PRETRAIN_BUFFER_SIZE` | 10000 | Replay buffer capacity |

Adjust these in `pretrain_aux.py` if needed.

## Key Features

✓ **Exact same data pipeline** as main training (env.get_img_sequence, env.get_drone_state)  
✓ **Same loss function** as main training (aux_loss from sac.py)  
✓ **Compatible architecture** - directly loads into GaussianPolicy.GRU  
✓ **Ground truth from AirSim** - perfect state for supervision  
✓ **TensorBoard logging** - monitor training progress  
✓ **Auto-checkpointing** - saves best model and periodic checkpoints  

## Expected Results

**Without pretraining:**
- Vision encoder learns from scratch during SAC
- Slow initial progress (exploration inefficient)
- Many episodes needed to converge

**With pretraining:**
- Vision encoder understands scene from episode 1
- Faster convergence (fewer episodes)
- More stable training (grounded features)
- Better final performance

## Troubleshooting

**Q: Error "Import torch could not be resolved"**  
A: This is just a VS Code linting issue. The code will run fine if PyTorch is installed.

**Q: Environment keeps crashing**  
A: Check AirSim is running. Try increasing `time.sleep()` in data collection.

**Q: Loss not decreasing**  
A: Check learning rate. Try reducing batch size. Ensure buffer has enough samples.

**Q: How do I know pretraining worked?**  
A: Check TensorBoard - losses should decrease. When loaded into SAC, early episodes should show better performance than random initialization.

## Advanced Usage

### Freeze Encoder During Early Training
```python
# Freeze vision encoder for first 1000 updates
for param in agent.policy.GRU.parameters():
    param.requires_grad = False

# ... train for 1000 updates ...

# Unfreeze
for param in agent.policy.GRU.parameters():
    param.requires_grad = True
```

### Resume Pretraining
```python
# In pretrain_aux.py, before training loop:
checkpoint = torch.load('pretrained_models/aux_pretrain_XXX/checkpoint_ep100.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_episode = checkpoint['episode'] + 1
```

### Use Different Environment
Just change `env_args` in `pretrain_vision_encoder()` to match your environment configuration.

## File Structure
```
SACfD/
├── pretrain_aux.py              # Main pretraining script
├── example_load_pretrained.py   # Usage example
├── pretrained_models/           # Saved checkpoints (created automatically)
│   └── aux_pretrain_TIMESTAMP/
│       ├── best_aux_model.pt
│       ├── final_aux_model.pt
│       └── checkpoint_epXX.pt
└── runs/                        # TensorBoard logs
    └── aux_pretrain_TIMESTAMP/
```

## Next Steps

1. Run pretraining until loss plateaus (or 500 episodes)
2. Note the path to `best_aux_model.pt`
3. Modify your main training script to load these weights
4. Compare training curves with/without pretraining
5. Enjoy faster convergence! 🚀
