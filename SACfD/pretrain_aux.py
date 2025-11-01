"""
Pretrain ResNet+GRU auxiliary heads for predicting drone state from images.

═══════════════════════════════════════════════════════════════════════════════
OVERVIEW
═══════════════════════════════════════════════════════════════════════════════
This script pretrains the vision encoder (ResNet+GRU) to predict physical state
from FPV camera images BEFORE running the main SAC training. This gives the 
network a head start in understanding the visual scene.

WHAT IT TRAINS:
  - ResNet auxiliary head: predicts relative position (3D) + rotation (6D→9D matrix)
  - GRU auxiliary head: predicts relative velocity (3D) + angular velocity (3D)
  
These are the SAME auxiliary heads used in the main SAC training (sac.py).

═══════════════════════════════════════════════════════════════════════════════
HOW IT WORKS
═══════════════════════════════════════════════════════════════════════════════
1. DATA COLLECTION:
   - Two modes available:
     a) RANDOM: Flies the drone with random actions (fast, diverse, noisy)
     b) EXPERT: Uses CEM-MPC controller (slower, higher quality, better trajectories)
   - Collects: image sequences + ground truth state from get_drone_state()
   - Ground truth is computed from AirSim's perfect state (position, velocity, etc.)
   - Set USE_EXPERT_DATA = True/False to choose mode
   
2. TRAINING:
   - Supervised learning with MSE loss
   - Same loss function as main training (aux_loss in sac.py)
   - Trains ONLY the vision encoder (ResNet+GRU), NOT the policy
   
3. SAVING:
   - Saves checkpoints every N episodes
   - Saves best model (lowest loss)
   - Final model saved at end

═══════════════════════════════════════════════════════════════════════════════
DATA FLOW (matches main pipeline exactly)
═══════════════════════════════════════════════════════════════════════════════
env.get_img_sequence() 
    ↓
img_tensor: (1, T=4, 3, 256, 144)  ← FPV images
relative_next_target_pos: (T, 3)  ← position to next gate/target (÷10)
attitude_9d: (T, 3, 3)             ← rotation matrix
relative_next_target_vel: (T, 3)  ← velocity relative to target (÷10)
fpv_angular_vel: (T, 3)            ← angular velocity (raw)
    ↓
PretrainBuffer stores: (img_seq, pos[-1], rot[-1], vel[-1], ang_vel[-1])
    ↓
GRU.forward(img_seq) → (features, resnet_aux_seq, gru_aux_seq, hidden)
    ↓
Take last timestep: resnet_aux_seq[:, -1, :], gru_aux_seq[:, -1, :]
    ↓
Compute MSE loss vs ground truth
    ↓
Backprop + update weights

═══════════════════════════════════════════════════════════════════════════════
USAGE
═══════════════════════════════════════════════════════════════════════════════
1. Run pretraining:
   python pretrain_aux.py
   
2. Wait for training to complete (or interrupt when loss plateaus)

3. In your main training script (main.py), load pretrained weights:
   
   from pretrain_aux import load_pretrained_weights_to_policy
   
   # After creating SAC agent
   agent = SAC(args)
   agent.policy = load_pretrained_weights_to_policy(
       agent.policy, 
       'pretrained_models/aux_pretrain_XXXXXX/best_aux_model.pt',
       device
   )
   
4. Start main SAC training - the vision encoder now has a good initialization!

═══════════════════════════════════════════════════════════════════════════════
HYPERPARAMETERS
═══════════════════════════════════════════════════════════════════════════════
Adjust these in the script if needed:
  - USE_EXPERT_DATA: True/False (use CEM-MPC or random actions)
  - PRETRAIN_EPISODES: 500 (number of flight episodes)
  - PRETRAIN_BATCH_SIZE: 32
  - PRETRAIN_LEARNING_RATE: 1e-3 (higher than main training)
  - PRETRAIN_EPOCHS_PER_EPISODE: 5 (train multiple times on same data)
  - PRETRAIN_BUFFER_SIZE: 10000 (replay buffer for collected data)

Expert vs Random Data Collection:
  - EXPERT (CEM-MPC): Better trajectories, slower collection, recommended
  - RANDOM: Faster collection, more diverse but noisy, good for exploration

═══════════════════════════════════════════════════════════════════════════════
BENEFITS
═══════════════════════════════════════════════════════════════════════════════
✓ Faster convergence in main training
✓ Better vision feature representations from the start
✓ Reduced sample complexity (fewer episodes needed)
✓ More stable training (vision grounded in physics)
✓ Can pretrain on different environments/scenarios

═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import os
from collections import deque
from datetime import datetime

from env import env
from model import GRU, init_weights
from config import *
from CEM_MPC import CEM_MPC

# Pretraining specific hyperparameters
PRETRAIN_EPISODES = 500  # Number of random flight episodes for data collection
PRETRAIN_BATCH_SIZE = 32
PRETRAIN_LEARNING_RATE = 1e-3
PRETRAIN_EPOCHS_PER_EPISODE = 5  # Train multiple epochs on collected data
PRETRAIN_BUFFER_SIZE = 10000
PRETRAIN_SAVE_INTERVAL = 50  # Save checkpoint every N episodes
USE_EXPERT_DATA = True  # Set to True to use CEM-MPC for data collection, False for random actions

# Loss weights (same as main training)
AUX_LOSS_WEIGHT_PRETRAIN = 1.0
POS_LOSS_WEIGHT_PRETRAIN = POS_LOSS_WEIGHT
ROT_LOSS_WEIGHT_PRETRAIN = ROT_LOSS_WEIGHT
VEL_LOSS_WEIGHT_PRETRAIN = VEL_LOSS_WEIGHT
ANG_VEL_LOSS_WEIGHT_PRETRAIN = ANG_VEL_LOSS_WEIGHT


class PretrainBuffer:
    """Buffer to store image sequences and corresponding ground truth labels"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, img_seq, pos, rot_mat, vel, ang_vel):
        """
        Store one timestep of data
        img_seq: (frames, C, H, W) - image sequence
        pos: (3,) - relative position to next target
        rot_mat: (3, 3) - rotation matrix
        vel: (3,) - relative velocity
        ang_vel: (3,) - angular velocity
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (img_seq, pos, rot_mat, vel, ang_vel)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample a batch of data"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        img_seqs = []
        positions = []
        rot_mats = []
        velocities = []
        ang_vels = []
        
        for idx in indices:
            img_seq, pos, rot_mat, vel, ang_vel = self.buffer[idx]
            img_seqs.append(img_seq)
            positions.append(pos)
            rot_mats.append(rot_mat)
            velocities.append(vel)
            ang_vels.append(ang_vel)
        
        # Stack into tensors
        img_seqs = torch.from_numpy(np.stack(img_seqs)).float()  # (B, T, C, H, W)
        positions = torch.from_numpy(np.stack(positions)).float()  # (B, 3)
        rot_mats = torch.from_numpy(np.stack(rot_mats)).float()  # (B, 3, 3)
        velocities = torch.from_numpy(np.stack(velocities)).float()  # (B, 3)
        ang_vels = torch.from_numpy(np.stack(ang_vels)).float()  # (B, 3)
        
        return img_seqs, positions, rot_mats, velocities, ang_vels
    
    def __len__(self):
        return len(self.buffer)


def six_d_to_rot_mat(pred_6d):
    """
    Convert 6D representation to 3x3 rotation matrix.
    Same as in sac.py to ensure compatibility.
    """
    a1 = pred_6d[..., 0:3]
    a2 = pred_6d[..., 3:6]
    
    b1 = F.normalize(a1, dim=-1)
    dot_product = torch.sum(b1 * a2, dim=-1, keepdim=True)
    a2_orthogonal = a2 - dot_product * b1
    b2 = F.normalize(a2_orthogonal, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    
    return torch.stack([b1, b2, b3], dim=-1)


def compute_aux_loss(resnet_output, gru_output, gt_pos, gt_rot_mat, gt_vel, gt_ang_vel):
    """
    Compute auxiliary loss for pretraining.
    Same logic as sac.py aux_loss to ensure compatibility.
    """
    # Extract predictions
    pred_pos = resnet_output[:, 0:3]  # (B, 3)
    pred_rot_6d = resnet_output[:, 3:9]  # (B, 6)
    pred_vel = gru_output[:, 0:3]  # (B, 3)
    pred_ang_vel = gru_output[:, 3:6]  # (B, 3)
    
    # Convert 6D rotation to 9D rotation matrix
    pred_rot_mat = six_d_to_rot_mat(pred_rot_6d)  # (B, 3, 3)
    
    # Compute weighted MSE for each component
    loss_pos = POS_LOSS_WEIGHT_PRETRAIN * F.mse_loss(pred_pos, gt_pos)
    loss_rot = ROT_LOSS_WEIGHT_PRETRAIN * F.mse_loss(pred_rot_mat, gt_rot_mat)
    loss_vel = VEL_LOSS_WEIGHT_PRETRAIN * F.mse_loss(pred_vel, gt_vel)
    loss_ang_vel = ANG_VEL_LOSS_WEIGHT_PRETRAIN * F.mse_loss(pred_ang_vel, gt_ang_vel)
    
    return loss_pos + loss_rot + loss_vel + loss_ang_vel, loss_pos, loss_rot, loss_vel, loss_ang_vel


def collect_random_flight_data(env_instance, buffer, num_steps=100):
    """
    Collect data by flying the drone with random actions.
    """
    step_count = 0
    collision_count = 0
    done = False
    
    while step_count < num_steps:
        # Random action (within control limits)
        random_action = np.random.uniform(
            low=[CONTROL_MIN] * ACTION_DIM,
            high=[CONTROL_MAX] * ACTION_DIM,
            size=(ACTION_DIM,)
        )
        
        # Step the environment
        try:
            # env.step returns: (current_drone_state, img_tensor, Q_state, reward, done, phase_idx, info, elapsed_time,
            #                     relative_next_target_pos, attitude_9d, relative_next_target_vel, fpv_angular_vel)
            _, img_tensor, _, reward, done, _, info, _, \
                relative_next_target_pos, attitude_9d, relative_next_target_vel, fpv_angular_vel = env_instance.step(random_action)
            
            # img_tensor shape: (1, T, C, H, W)
            # relative_next_target_pos shape: (T, 3) - already divided by 10 in get_drone_state
            # attitude_9d shape: (T, 3, 3) - rotation matrix
            # relative_next_target_vel shape: (T, 3) - already divided by 10
            # fpv_angular_vel shape: (T, 3)
            
            # Remove batch dimension from img_tensor
            img_seq_np = img_tensor.squeeze(0).cpu().numpy()  # (T, C, H, W)
            
            # For auxiliary head training, we use the LAST frame's ground truth
            # (since the network outputs predictions for the full sequence, we take the final one)
            pos = relative_next_target_pos[-1]  # (3,) - last frame's relative position
            rot_mat = attitude_9d[-1]  # (3, 3) - last frame's rotation matrix
            vel = relative_next_target_vel[-1]  # (3,) - last frame's relative velocity
            ang_vel = fpv_angular_vel[-1]  # (3,) - last frame's angular velocity
            
            # Store in buffer
            buffer.push(img_seq_np, pos, rot_mat, vel, ang_vel)
            step_count += 1
            
            if done:
                if info == 0:  # info=0 means collision
                    collision_count += 1
                # Reset and continue collecting
                env_instance.reset()
                done = False
                
        except Exception as e:
            print(f"Error during data collection: {e}")
            import traceback
            traceback.print_exc()
            try:
                env_instance.reset()
            except:
                pass
            continue
    
    return collision_count


def collect_expert_flight_data(env_instance, buffer, mpc_controller, num_steps=100):
    """
    Collect data by flying the drone with CEM-MPC controller (expert demonstrations).
    This provides higher quality data than random actions.
    
    Args:
        env_instance: Environment instance
        buffer: PretrainBuffer to store data
        mpc_controller: CEM_MPC controller instance
        num_steps: Number of steps to collect
    
    Returns:
        collision_count: Number of collisions during data collection
        success_count: Number of successful episodes (reached target)
    """
    step_count = 0
    collision_count = 0
    success_count = 0
    done = False
    
    print("  Collecting expert demonstrations using CEM-MPC...")
    
    while step_count < num_steps:
        try:
            # Get current drone state for MPC (13D state vector)
            drone_state, _, _, _, _, _ = env_instance.get_drone_state()
            
            # Get MPC action
            # CEM_MPC.step signature: step(current_true_state, current_idx, elapsed_time)
            # The MPC controller internally computes target sequences based on phase_idx
            mpc_action = mpc_controller.step(
                current_true_state=drone_state,
                current_idx=env_instance.phase_idx,
                elapsed_time=env_instance.elapsed_time
            )
            
            # Apply action to environment
            # env.step returns: (current_drone_state, img_tensor, Q_state, reward, done, phase_idx, info, elapsed_time,
            #                     relative_next_target_pos, attitude_9d, relative_next_target_vel, fpv_angular_vel)
            _, img_tensor, _, reward, done, _, info, _, \
                relative_next_target_pos, attitude_9d, relative_next_target_vel, fpv_angular_vel = env_instance.step(mpc_action)
            
            # Remove batch dimension from img_tensor
            img_seq_np = img_tensor.squeeze(0).cpu().numpy()  # (T, C, H, W)
            
            # Use the LAST frame's ground truth
            pos = relative_next_target_pos[-1]  # (3,)
            rot_mat = attitude_9d[-1]  # (3, 3)
            vel = relative_next_target_vel[-1]  # (3,)
            ang_vel = fpv_angular_vel[-1]  # (3,)
            
            # Store in buffer
            buffer.push(img_seq_np, pos, rot_mat, vel, ang_vel)
            step_count += 1
            
            if done:
                if info == 0:  # Collision
                    collision_count += 1
                elif info == 1:  # Success
                    success_count += 1
                
                # Reset and continue collecting
                env_instance.reset()
                done = False
                print(f"    Episode ended: {'Success' if info == 1 else 'Collision'}, collected {step_count}/{num_steps} steps")
                
        except Exception as e:
            print(f"  Error during expert data collection: {e}")
            import traceback
            traceback.print_exc()
            try:
                env_instance.reset()
            except:
                pass
            continue
    
    print(f"  Expert collection complete: {step_count} steps, {success_count} successes, {collision_count} collisions")
    return collision_count, success_count


def train_epoch(model, buffer, optimizer, device, batch_size):
    """
    Train for one epoch on the collected data.
    """
    if len(buffer) < batch_size:
        return None
    
    # Sample batch
    img_seqs, positions, rot_mats, velocities, ang_vels = buffer.sample(batch_size)
    
    # Move to device
    img_seqs = img_seqs.to(device)
    positions = positions.to(device)
    rot_mats = rot_mats.to(device)
    velocities = velocities.to(device)
    ang_vels = ang_vels.to(device)
    
    # Forward pass
    # GRU.forward returns: (final_feature_vector, resnet_aux_predictions, gru_aux_predictions, last_hidden_state)
    # resnet_aux_predictions shape: (B, T, 9) - we want last frame (B, 9)
    # gru_aux_predictions shape: (B, T, 6) - we want last frame (B, 6)
    _, resnet_output_seq, gru_output_seq, _ = model(img_seqs, hidden_state=None)
    
    # Take the last timestep predictions
    resnet_output = resnet_output_seq[:, -1, :]  # (B, 9)
    gru_output = gru_output_seq[:, -1, :]  # (B, 6)
    
    # Compute loss
    total_loss, loss_pos, loss_rot, loss_vel, loss_ang_vel = compute_aux_loss(
        resnet_output, gru_output, positions, rot_mats, velocities, ang_vels
    )
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return {
        'total_loss': total_loss.item(),
        'loss_pos': loss_pos.item(),
        'loss_rot': loss_rot.item(),
        'loss_vel': loss_vel.item(),
        'loss_ang_vel': loss_ang_vel.item()
    }


def pretrain_vision_encoder():
    """
    Main pretraining loop
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_type = "expert" if USE_EXPERT_DATA else "random"
    save_dir = f"pretrained_models/aux_pretrain_{data_type}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(f"runs/aux_pretrain_{data_type}_{timestamp}")
    
    # Initialize environment
    env_args = {
        'DT': DT,
        'img_time': img_time,
        'door_frames': door_frames_names,
        'door_param': door_param,
        'pass_threshold_y': WAYPOINT_PASS_THRESHOLD_Y,
        'control_max': CONTROL_MAX,
        'control_min': CONTROL_MIN,
        'reward_weight': REWARD_WEIGHT,
        'POS_TOLERANCE': POS_TOLERANCE,
        'frames': NUM_TRANSFORMER_FRAMES
    }
    env_instance = env(env_args)
    
    # Initialize MPC controller if using expert data
    mpc_controller = None
    if USE_EXPERT_DATA:
        print("Initializing CEM-MPC controller for expert demonstrations...")
        cem_hyperparams = {
            'prediction_horizon': PREDICTION_HORIZON,
            'n_samples': N_SAMPLES_CEM,
            'n_elites': N_ELITES_CEM,
            'n_iter': N_ITER_CEM,
            'initial_std': INITIAL_STD_CEM,
            'min_std': MIN_STD_CEM,
            'alpha': ALPHA_CEM
        }
        mpc_params = {
            'waypoint_pass_threshold_y': WAYPOINT_PASS_THRESHOLD_Y,
            'dt': DT,
            'control_max': CONTROL_MAX,
            'control_min': CONTROL_MIN,
            'q_state_matrix_gpu': Q_STATE_COST_MATRIX_GPU,
            'r_control_matrix_gpu': R_CONTROL_COST_MATRIX_GPU,
            'q_terminal_matrix_gpu': Q_TERMINAL_COST_MATRIX_GPU,
            'q_state_matrix_gpu_two': Q_STATE_COST_MATRIX_GPU_TWO,
            'r_control_matrix_gpu_two': R_CONTROL_COST_MATRIX_GPU,
            'q_terminal_matrix_gpu_two': Q_TERMINAL_COST_MATRIX_GPU_TWO,
            'static_q_state_matrix_gpu': STATIC_Q_STATE_COST_MATRIX_GPU,
            'static_r_control_matrix_gpu': STATIC_R_CONTROL_COST_MATRIX_GPU,
            'static_q_terminal_matrix_gpu': STATIC_Q_TERMINAL_COST_MATRIX_GPU,
            'action_dim': ACTION_DIM,
            'state_dim': MPC_STATE_DIM,
            'device': device,
            'pos_tolerence': POS_TOLERANCE,
            'align_cost': ALIGH_COST
        }
        mpc_controller = CEM_MPC(cem_hyperparams, mpc_params)
        print("CEM-MPC controller initialized successfully")
    
    # Initialize model (GRU contains ResNet)
    model = GRU(
        resnet_aux_outputs=RESNET_AUX_DIM,
        gru_hidden_dim=256,  # Embedding dimension, should match main training
        gru_aux_outputs=GRU_AUX_DIM,
        gru_layers=GRU_LAYER,
        dropout=DROP_OUT
    ).to(device)
    
    # Apply weight initialization
    model.apply(init_weights)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LEARNING_RATE, weight_decay=0.01)
    
    # Buffer
    buffer = PretrainBuffer(PRETRAIN_BUFFER_SIZE)
    
    # Training loop
    global_step = 0
    best_loss = float('inf')
    total_successes = 0
    total_collisions = 0
    
    print("Starting pretraining...")
    print(f"Data collection mode: {'EXPERT (CEM-MPC)' if USE_EXPERT_DATA else 'RANDOM'}")
    print(f"Total episodes: {PRETRAIN_EPISODES}")
    print(f"Save directory: {save_dir}")
    
    for episode in range(PRETRAIN_EPISODES):
        episode_start_time = time.time()
        
        # Reset environment
        try:
            env_instance.reset()
        except Exception as e:
            print(f"Error resetting environment: {e}")
            continue
        
        # Collect data (expert or random)
        print(f"\nEpisode {episode + 1}/{PRETRAIN_EPISODES}: Collecting data...")
        if USE_EXPERT_DATA:
            collision_count, success_count = collect_expert_flight_data(
                env_instance, buffer, mpc_controller, num_steps=50
            )
            total_successes += success_count
            total_collisions += collision_count
        else:
            collision_count = collect_random_flight_data(
                env_instance, buffer, num_steps=50
            )
            total_collisions += collision_count
        
        # Train on collected data
        if len(buffer) >= PRETRAIN_BATCH_SIZE:
            print(f"Training on {len(buffer)} samples...")
            epoch_losses = []
            
            for epoch in range(PRETRAIN_EPOCHS_PER_EPISODE):
                loss_dict = train_epoch(model, buffer, optimizer, device, PRETRAIN_BATCH_SIZE)
                if loss_dict is not None:
                    epoch_losses.append(loss_dict)
                    global_step += 1
                    
                    # Log to TensorBoard
                    writer.add_scalar('Loss/total', loss_dict['total_loss'], global_step)
                    writer.add_scalar('Loss/position', loss_dict['loss_pos'], global_step)
                    writer.add_scalar('Loss/rotation', loss_dict['loss_rot'], global_step)
                    writer.add_scalar('Loss/velocity', loss_dict['loss_vel'], global_step)
                    writer.add_scalar('Loss/angular_velocity', loss_dict['loss_ang_vel'], global_step)
            
            # Average losses for this episode
            if epoch_losses:
                avg_loss = np.mean([l['total_loss'] for l in epoch_losses])
                avg_loss_pos = np.mean([l['loss_pos'] for l in epoch_losses])
                avg_loss_rot = np.mean([l['loss_rot'] for l in epoch_losses])
                avg_loss_vel = np.mean([l['loss_vel'] for l in epoch_losses])
                avg_loss_ang = np.mean([l['loss_ang_vel'] for l in epoch_losses])
                
                episode_time = time.time() - episode_start_time
                
                print(f"Episode {episode + 1} completed in {episode_time:.2f}s")
                print(f"  Buffer size: {len(buffer)}")
                if USE_EXPERT_DATA:
                    print(f"  Total successes: {total_successes}, Total collisions: {total_collisions}")
                else:
                    print(f"  Total collisions: {total_collisions}")
                print(f"  Avg Total Loss: {avg_loss:.6f}")
                print(f"  Avg Pos Loss: {avg_loss_pos:.6f}")
                print(f"  Avg Rot Loss: {avg_loss_rot:.6f}")
                print(f"  Avg Vel Loss: {avg_loss_vel:.6f}")
                print(f"  Avg Ang Loss: {avg_loss_ang:.6f}")
                
                # Save best model
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_path = os.path.join(save_dir, "best_aux_model.pt")
                    torch.save({
                        'episode': episode,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                    }, save_path)
                    print(f"  *** New best model saved! Loss: {best_loss:.6f} ***")
        
        # Periodic checkpoint
        if (episode + 1) % PRETRAIN_SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_ep{episode + 1}.pt")
            torch.save({
                'episode': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'buffer_size': len(buffer),
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(save_dir, "final_aux_model.pt")
    torch.save({
        'episode': PRETRAIN_EPISODES,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': best_loss,
    }, final_path)
    print(f"\nFinal model saved: {final_path}")
    print(f"Best loss achieved: {best_loss:.6f}")
    
    writer.close()
    env_instance.client.reset()
    
    return model, save_dir


if __name__ == "__main__":
    print("="*60)
    print("ResNet+GRU Auxiliary Head Pretraining")
    print("="*60)
    
    try:
        pretrained_model, save_directory = pretrain_vision_encoder()
        print("\n" + "="*60)
        print("Pretraining completed successfully!")
        print(f"Models saved in: {save_directory}")
        print("="*60)
        print("\nTo use the pretrained weights in main training:")
        print(f"  1. Load the model from: {save_directory}/best_aux_model.pt")
        print("  2. In sac.py __init__, after creating self.policy, add:")
        print("     ```")
        print("     checkpoint = torch.load('path/to/best_aux_model.pt')")
        print("     self.policy.GRU.load_state_dict(checkpoint['model_state_dict'])")
        print("     print('Loaded pretrained auxiliary heads')")
        print("     ```")
        print("  3. This will initialize the ResNet+GRU encoder with pretrained weights")
        print("  4. The policy MLP heads will still be randomly initialized")
        print("="*60)
    except KeyboardInterrupt:
        print("\n\nPretraining interrupted by user.")
    except Exception as e:
        print(f"\n\nError during pretraining: {e}")
        import traceback
        traceback.print_exc()


def load_pretrained_weights_to_policy(policy_model, checkpoint_path, device):
    """
    Helper function to load pretrained GRU weights into a GaussianPolicy model.
    
    Args:
        policy_model: Instance of GaussianPolicy from model.py
        checkpoint_path: Path to the pretrained checkpoint (e.g., 'best_aux_model.pt')
        device: torch device
    
    Returns:
        policy_model with loaded pretrained weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy_model.GRU.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded pretrained auxiliary heads from {checkpoint_path}")
    print(f"  Pretrained episode: {checkpoint.get('episode', 'unknown')}")
    print(f"  Pretrained loss: {checkpoint.get('loss', 'unknown'):.6f}")
    return policy_model
