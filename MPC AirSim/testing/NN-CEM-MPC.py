import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time

# --- System Parameters & Constants (mostly unchanged) ---
DT = 0.1
MAX_SIM_TIME_PER_EPISODE = 20 # Max time for a single episode
# N_STEPS_PER_EPISODE calculated later

TARGET_POS = np.array([20.0, 10.0])
TARGET_VEL = np.array([0.0, 0.0])
TARGET_STATE = np.concatenate((TARGET_POS, TARGET_VEL))

START_POS = np.array([0.0, 0.0])
START_VEL = np.array([0.0, 0.0])
START_STATE = np.concatenate((START_POS, START_VEL))

ACCEL_MAX = 2.0
PREDICTION_HORIZON = 10
N_SAMPLES_CEM = 100
N_ELITES_CEM = 10
N_ITER_CEM = 5
INITIAL_STD_CEM = 0.5
MIN_STD_CEM = 0.05
ALPHA_CEM = 0.5

R_CONTROL_COST_MATRIX = 0.01 * np.eye(2)
Q_STATE_COST_MATRIX = np.diag([10.0, 10.0, 1.0, 1.0])
Q_TERMINAL_COST_MATRIX = np.diag([100.0, 100.0, 10.0, 10.0])

STATE_DIM = 4
ACTION_DIM = 2
NN_HIDDEN_SIZE = 64
LEARNING_RATE = 1e-3 # Might need adjustment for multi-episode learning
BUFFER_SIZE = 20000 # Increased buffer for more experience
BATCH_SIZE = 64
NN_TRAIN_EPOCHS_PER_STEP = 5 # How many training updates per actual system step
MIN_BUFFER_FOR_TRAINING = 64 # Start training NN once buffer has this many samples

# --- Multi-Episode Parameters ---
NUM_EPISODES = 20 # Number of episodes to run

# --- TrueBlackBoxDynamics (unchanged) ---
class TrueBlackBoxDynamics:
    def __init__(self, dt):
        self.dt = dt
        self.alpha_accel_effect = np.random.uniform(0.7, 1.3, size=2)
        print(f"黑箱动力学模型初始化，未知的控制有效性 alpha: {self.alpha_accel_effect}")

    def step(self, state, control):
        x, y, vx, vy = state
        ax, ay = control
        effective_ax = self.alpha_accel_effect[0] * ax
        effective_ay = self.alpha_accel_effect[1] * ay
        new_vx = vx + effective_ax * self.dt
        new_vy = vy + effective_ay * self.dt
        new_x = x + vx * self.dt
        new_y = y + vy * self.dt
        return np.array([new_x, new_y, new_vx, new_vy])

# --- DynamicsNN (unchanged) ---
class DynamicsNN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(DynamicsNN, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, state_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        next_state_pred = self.fc3(x)
        return next_state_pred

# --- ReplayBuffer (unchanged) ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, next_state):
        self.buffer.append((state, action, next_state))
    def sample(self, batch_size):
        state, action, next_state = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(next_state)
    def __len__(self):
        return len(self.buffer)

# --- cost_function (unchanged) ---
def cost_function(state_trajectory, control_sequence):
    total_cost = 0.0
    for t in range(PREDICTION_HORIZON):
        predicted_state_at_t_plus_1 = state_trajectory[t+1]
        control_input_at_t = control_sequence[t]
        state_error = predicted_state_at_t_plus_1 - TARGET_STATE
        total_cost += state_error.T @ Q_STATE_COST_MATRIX @ state_error
        total_cost += control_input_at_t.T @ R_CONTROL_COST_MATRIX @ control_input_at_t
    terminal_state = state_trajectory[-1]
    terminal_state_error = terminal_state - TARGET_STATE
    total_cost += terminal_state_error.T @ Q_TERMINAL_COST_MATRIX @ terminal_state_error
    return total_cost

# --- train_nn_model (unchanged) ---
def train_nn_model(nn_model, optimizer, replay_buffer, batch_size, epochs, min_buffer_size):
    if len(replay_buffer) < min_buffer_size: # Use passed min_buffer_size
        return 0.0
    total_loss_this_training_cycle = 0
    actual_epochs_run = 0
    for epoch in range(epochs):
        if len(replay_buffer) < batch_size: # Ensure enough samples for a batch
            break
        states, actions, next_states_true = replay_buffer.sample(batch_size)
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.FloatTensor(actions)
        next_states_true_tensor = torch.FloatTensor(next_states_true)
        next_states_pred_tensor = nn_model(states_tensor, actions_tensor)
        loss = nn.MSELoss()(next_states_pred_tensor, next_states_true_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss_this_training_cycle += loss.item()
        actual_epochs_run +=1
    return total_loss_this_training_cycle / actual_epochs_run if actual_epochs_run > 0 else 0.0


# --- Function to run a single episode ---
def run_single_episode_adaptive_cem_mpc(
    episode_num, true_system, nn_model, optimizer, replay_buffer,
    start_state, target_state, max_sim_time, dt,
    cem_params, nn_train_params
):
    n_steps_this_episode = int(max_sim_time / dt)
    current_true_state = start_state.copy()
    # Reset CEM warm-start for each episode to rely on learned model
    mean_control_sequence_for_warm_start = np.zeros((cem_params['prediction_horizon'], ACTION_DIM))

    actual_trajectory = [current_true_state.copy()]
    applied_controls = []
    time_points = [0.0]
    model_losses_this_episode = []
    reached_target = False
    steps_taken_in_episode = 0

    print(f"\n--- 开始第 {episode_num + 1} 次训练 ---")
    for step in range(n_steps_this_episode):
        steps_taken_in_episode = step + 1
        if step == n_steps_this_episode - 1:
            print(f"第 {episode_num + 1} 次训练: 仿真时间到，未严格到达目标。")

        min_cost_this_mpc_step = float('inf')

        # Decide whether to use CEM or random actions
        use_cem = len(replay_buffer) >= nn_train_params['min_buffer_for_training'] or episode_num > 0 # Be less conservative after 1st episode
        a=time.time()
        if not use_cem and step < 50 : # More initial random exploration in early episodes if buffer small
             actual_control_to_apply = np.random.uniform(-ACCEL_MAX, ACCEL_MAX, size=ACTION_DIM)
        else:
            cem_iter_mean = mean_control_sequence_for_warm_start.copy()
            cem_iter_std = np.full((cem_params['prediction_horizon'], ACTION_DIM), cem_params['initial_std'])
            a=time.time()
            for cem_iter in range(cem_params['n_iter']):
                perturbations = np.random.normal(loc=0.0, scale=1.0,
                                                 size=(cem_params['n_samples'], cem_params['prediction_horizon'], ACTION_DIM))
                sampled_control_sequences = cem_iter_mean[np.newaxis, :, :] + \
                                            perturbations * cem_iter_std[np.newaxis, :, :]
                sampled_control_sequences = np.clip(sampled_control_sequences, -ACCEL_MAX, ACCEL_MAX)

                costs_cem = np.zeros(cem_params['n_samples'])
                for k_sample in range(cem_params['n_samples']):
                    temp_state_trajectory_nn = [current_true_state.copy()]
                    current_sim_state_nn = torch.FloatTensor(current_true_state)
                    for t_horizon in range(cem_params['prediction_horizon']):
                        control_input_nn = torch.FloatTensor(sampled_control_sequences[k_sample, t_horizon])
                        with torch.no_grad():
                            next_state_pred_nn = nn_model(current_sim_state_nn.unsqueeze(0),
                                                          control_input_nn.unsqueeze(0)).squeeze(0)
                        temp_state_trajectory_nn.append(next_state_pred_nn.numpy().copy())
                        current_sim_state_nn = next_state_pred_nn
                    costs_cem[k_sample] = cost_function(np.array(temp_state_trajectory_nn), sampled_control_sequences[k_sample])
                
                min_cost_this_mpc_step = min(min_cost_this_mpc_step, np.min(costs_cem))
                elite_indices = np.argsort(costs_cem)[:cem_params['n_elites']]
                elite_sequences = sampled_control_sequences[elite_indices]
                
                new_mean = np.mean(elite_sequences, axis=0)
                new_std = np.std(elite_sequences, axis=0)
                cem_iter_mean = cem_params['alpha'] * new_mean + (1 - cem_params['alpha']) * cem_iter_mean
                cem_iter_std = cem_params['alpha'] * new_std + (1 - cem_params['alpha']) * cem_iter_std
                cem_iter_std = np.maximum(cem_iter_std, cem_params['min_std'])

            optimal_control_sequence_this_step = cem_iter_mean
            actual_control_to_apply = optimal_control_sequence_this_step[0, :].copy()
            
            # Update warm start for *next MPC step within this episode*
            mean_control_sequence_for_warm_start = np.roll(optimal_control_sequence_this_step, -1, axis=0)
            mean_control_sequence_for_warm_start[-1, :] = optimal_control_sequence_this_step[-2, :].copy()
        b=time.time()
        print(b-a)
        next_true_state = true_system.step(current_true_state, actual_control_to_apply)
        replay_buffer.push(current_true_state, actual_control_to_apply, next_true_state)
        
        avg_nn_loss = train_nn_model(nn_model, optimizer, replay_buffer,
                                     nn_train_params['batch_size'], nn_train_params['epochs_per_step'],
                                     nn_train_params['min_buffer_for_training'])
        model_losses_this_episode.append(avg_nn_loss if avg_nn_loss > 0 else np.nan)

        applied_controls.append(actual_control_to_apply.copy())
        current_true_state = next_true_state.copy()
        actual_trajectory.append(current_true_state.copy())
        time_points.append((step + 1) * dt)

        if step % 20 == 0: # Print less frequently within an episode
            print(f"  Ep {episode_num+1}, 步骤: {step}, 位置: {current_true_state[:2]}, "
                  f"MPC代价: {min_cost_this_mpc_step if min_cost_this_mpc_step != float('inf') else -1:.2f}, "
                  f"NN损失: {avg_nn_loss:.4f}, Buffer: {len(replay_buffer)}")
            
    avg_episode_loss = np.nanmean(model_losses_this_episode) if len(model_losses_this_episode) > 0 else 0
    print(f"--- 第 {episode_num + 1} 次训练结束 --- 步数: {steps_taken_in_episode}, "
          f"是否到达: {reached_target}, 平均损失: {avg_episode_loss:.4f}")

    return (np.array(actual_trajectory), np.array(applied_controls),
            np.array(time_points), np.array(model_losses_this_episode),
            reached_target, steps_taken_in_episode, avg_episode_loss)

# --- Main Script ---
if __name__ == "__main__":
    try:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("中文字体 WenQuanYi Micro Hei 未找到，将使用默认字体。")

    # Initialize persistent components ONCE
    true_system = TrueBlackBoxDynamics(DT)
    nn_dynamics_model = DynamicsNN(STATE_DIM, ACTION_DIM, NN_HIDDEN_SIZE)
    nn_optimizer = optim.Adam(nn_dynamics_model.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    cem_parameters = {
        'prediction_horizon': PREDICTION_HORIZON,
        'n_samples': N_SAMPLES_CEM,
        'n_elites': N_ELITES_CEM,
        'n_iter': N_ITER_CEM,
        'initial_std': INITIAL_STD_CEM,
        'min_std': MIN_STD_CEM,
        'alpha': ALPHA_CEM
    }
    nn_training_parameters = {
        'batch_size': BATCH_SIZE,
        'epochs_per_step': NN_TRAIN_EPOCHS_PER_STEP,
        'min_buffer_for_training': MIN_BUFFER_FOR_TRAINING
    }

    # Store metrics for each episode
    episode_steps_taken = []
    episode_reached_target_flags = []
    episode_avg_losses = []
    
    last_episode_data = None # To store data for plotting the last episode

    for episode in range(NUM_EPISODES):
        trajectory, controls, times, nn_losses_ep, \
        reached_target_ep, steps_taken_ep, avg_loss_ep = run_single_episode_adaptive_cem_mpc(
            episode_num=episode,
            true_system=true_system, # Same system
            nn_model=nn_dynamics_model, # Persistent model
            optimizer=nn_optimizer, # Persistent optimizer
            replay_buffer=replay_buffer, # Persistent buffer
            start_state=START_STATE,
            target_state=TARGET_STATE,
            max_sim_time=MAX_SIM_TIME_PER_EPISODE,
            dt=DT,
            cem_params=cem_parameters,
            nn_train_params=nn_training_parameters
        )
        
        episode_steps_taken.append(steps_taken_ep)
        episode_reached_target_flags.append(1 if reached_target_ep else 0) # Convert boolean to int for easy plotting/summing
        episode_avg_losses.append(avg_loss_ep)

        if episode == NUM_EPISODES - 1: # Save data of the last episode for detailed plotting
            last_episode_data = {
                "trajectory": trajectory, "controls": controls,
                "times": times, "nn_losses": nn_losses_ep
            }

    # --- Plotting ---
    # Plot 1: Detailed plots for the LAST episode
    if last_episode_data:
        fig_last_ep, axs_last_ep = plt.subplots(2, 3, figsize=(22, 12))
        fig_last_ep.suptitle(f'自适应CEM-MPC - 第 {NUM_EPISODES} 次训练结果', fontsize=16)
        
        # Trajectory
        axs_last_ep[0, 0].plot(last_episode_data["trajectory"][:, 0], last_episode_data["trajectory"][:, 1], marker='.', linestyle='-', label='实际轨迹')
        axs_last_ep[0, 0].plot(START_POS[0], START_POS[1], 'go', markersize=10, label='起点')
        axs_last_ep[0, 0].plot(TARGET_POS[0], TARGET_POS[1], 'rx', markersize=10, mew=2, label='目标点')
        axs_last_ep[0, 0].set_xlabel('X 位置'); axs_last_ep[0, 0].set_ylabel('Y 位置'); axs_last_ep[0, 0].set_title('运动轨迹')
        axs_last_ep[0, 0].legend(); axs_last_ep[0, 0].grid(True); axs_last_ep[0, 0].axis('equal')

        # Position vs. Time
        axs_last_ep[0, 1].plot(last_episode_data["times"], last_episode_data["trajectory"][:, 0], label='X 位置')
        axs_last_ep[0, 1].plot(last_episode_data["times"], last_episode_data["trajectory"][:, 1], label='Y 位置')
        axs_last_ep[0, 1].axhline(TARGET_POS[0], color='r', linestyle='--', label='目标 X'); axs_last_ep[0, 1].axhline(TARGET_POS[1], color='g', linestyle='--', label='目标 Y')
        axs_last_ep[0, 1].set_xlabel('时间 (s)'); axs_last_ep[0, 1].set_ylabel('位置'); axs_last_ep[0, 1].set_title('位置 vs. 时间')
        axs_last_ep[0, 1].legend(); axs_last_ep[0, 1].grid(True)

        # Velocity vs. Time
        axs_last_ep[1, 0].plot(last_episode_data["times"], last_episode_data["trajectory"][:, 2], label='X 速度')
        axs_last_ep[1, 0].plot(last_episode_data["times"], last_episode_data["trajectory"][:, 3], label='Y 速度')
        axs_last_ep[1, 0].axhline(TARGET_VEL[0], color='r', linestyle='--', label='目标 X Vel'); axs_last_ep[1, 0].axhline(TARGET_VEL[1], color='g', linestyle='--', label='目标 Y Vel')
        axs_last_ep[1, 0].set_xlabel('时间 (s)'); axs_last_ep[1, 0].set_ylabel('速度'); axs_last_ep[1, 0].set_title('速度 vs. 时间')
        axs_last_ep[1, 0].legend(); axs_last_ep[1, 0].grid(True)

        # Control Input vs. Time
        control_times_last_ep = last_episode_data["times"][:len(last_episode_data["controls"])]
        if len(last_episode_data["controls"]) > 0:
            axs_last_ep[1, 1].plot(control_times_last_ep, last_episode_data["controls"][:, 0], label='X 加速度')
            axs_last_ep[1, 1].plot(control_times_last_ep, last_episode_data["controls"][:, 1], label='Y 加速度')
        axs_last_ep[1, 1].set_xlabel('时间 (s)'); axs_last_ep[1, 1].set_ylabel('控制输入'); axs_last_ep[1, 1].set_title('控制输入 vs. 时间')
        axs_last_ep[1, 1].legend(); axs_last_ep[1, 1].grid(True); axs_last_ep[1, 1].set_ylim([-ACCEL_MAX*1.1, ACCEL_MAX*1.1])

        # NN Model Loss for the last episode
        nn_losses_plot = last_episode_data["nn_losses"]
        if len(nn_losses_plot) > 0:
            valid_loss_indices = ~np.isnan(nn_losses_plot)
            times_for_losses = last_episode_data["times"][1:len(nn_losses_plot)+1]
            if np.any(valid_loss_indices):
                 axs_last_ep[0, 2].plot(times_for_losses[valid_loss_indices], nn_losses_plot[valid_loss_indices], label='NN 模型损失 (MSE)')
            else:
                 axs_last_ep[0, 2].text(0.5,0.5, '无有效损失', transform=axs_last_ep[0,2].transAxes, ha='center', va='center')
        else:
            axs_last_ep[0, 2].text(0.5,0.5, '无损失数据', transform=axs_last_ep[0,2].transAxes, ha='center', va='center')
        axs_last_ep[0, 2].set_xlabel('时间 (s)'); axs_last_ep[0, 2].set_ylabel('平均 MSE 损失'); axs_last_ep[0, 2].set_title('学习的动力学模型损失（上次训练）')
        axs_last_ep[0, 2].legend(); axs_last_ep[0, 2].grid(True); axs_last_ep[0, 2].set_yscale('log')
        
        axs_last_ep[1, 2].axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Plot 2: Learning progress across episodes
    fig_progress, axs_progress = plt.subplots(1, 2, figsize=(15, 6))
    fig_progress.suptitle('跨训练周期的学习进度', fontsize=16)

    # Steps to reach target per episode
    axs_progress[0].plot(range(1, NUM_EPISODES + 1), episode_steps_taken, marker='o', linestyle='-')
    axs_progress[0].set_xlabel('训练周期编号')
    axs_progress[0].set_ylabel('到达目标所需步数')
    axs_progress[0].set_title('每轮训练到达目标的步数')
    axs_progress[0].grid(True)
    
    # Add a secondary y-axis for "Reached Target"
    ax2_steps = axs_progress[0].twinx()
    ax2_steps.plot(range(1, NUM_EPISODES + 1), episode_reached_target_flags, marker='x', linestyle='--', color='r', label='是否到达目标 (1=是)')
    ax2_steps.set_ylabel('是否到达目标 (1=是)', color='r')
    ax2_steps.tick_params(axis='y', labelcolor='r')
    ax2_steps.set_ylim([-0.1, 1.1]) # For boolean 0/1
    
    # Average NN loss per episode
    axs_progress[1].plot(range(1, NUM_EPISODES + 1), episode_avg_losses, marker='o', linestyle='-')
    axs_progress[1].set_xlabel('训练周期编号')
    axs_progress[1].set_ylabel('平均 NN 模型损失 (MSE)')
    axs_progress[1].set_title('每轮训练的平均模型损失')
    axs_progress[1].grid(True)
    axs_progress[1].set_yscale('log') # Loss often decreases over orders of magnitude

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()