import numpy as np
import matplotlib.pyplot as plt
import time

# --- 系统参数 ---
DT = 0.1  # 时间步长
MAX_SIM_TIME = 12.0 # 增加一点仿真时间，确保能到达
N_STEPS = int(MAX_SIM_TIME / DT)

# 目标点和目标状态
TARGET_POS = np.array([20.0, 10.0])
TARGET_VEL = np.array([0.0, 0.0])
TARGET_STATE = np.concatenate((TARGET_POS, TARGET_VEL)) # [target_x, target_y, target_vx, target_vy]

START_POS = np.array([0.0, 0.0])
START_VEL = np.array([0.0, 0.0])
START_STATE = np.concatenate((START_POS, START_VEL))


# 控制限制 (假设加速度限制)
ACCEL_MAX = 2.0

# --- MPPI 参数 ---
N_SAMPLES = 1000       # 采样数量
PREDICTION_HORIZON = 100 # 预测时域 (步数)
LAMBDA = 0.05           # 温度参数 (可能需要调整)
NOISE_SIGMA_CONTROL = np.array([0.5, 0.5]) # 控制噪声标准差 (ax, ay)

# --- 代价函数权重 (使用你提供的矩阵形式) ---
# 控制输入的代价权重矩阵 R
R_CONTROL_COST_MATRIX = 0.01 * np.eye(2) # 对 ax, ay 的惩罚

# 状态误差的代价权重矩阵 Q
Q_STATE_COST_MATRIX = np.diag([
    10.0,  # x 位置误差的权重
    10.0,  # y 位置误差的权重
    1.0,   # x 速度误差的权重
    1.0    # y 速度误差的权重
])
# 终端状态误差的代价权重矩阵 Q_terminal
Q_TERMINAL_COST_MATRIX = np.diag([
    100.0, # 终端 x 位置误差
    100.0, # 终端 y 位置误差
    10.0,  # 终端 x 速度误差
    10.0   # 终端 y 速度误差
])

# --- 系统动力学模型 (二维质点) ---
# state = [x, y, vx, vy]
# control = [ax, ay]
def dynamics(state, control):
    x, y, vx, vy = state
    ax, ay = control

    new_x = x + vx * DT
    new_y = y + vy * DT
    new_vx = vx + ax * DT
    new_vy = vy + ay * DT
    return np.array([new_x, new_y, new_vx, new_vy])

# --- 修改后的成本函数 ---
def cost_function(state_trajectory, control_sequence):
    """
    计算给定状态轨迹和控制序列的总代价。
    state_trajectory: 形状 (PREDICTION_HORIZON + 1, 4)，包含从当前状态开始的预测状态
    control_sequence: 形状 (PREDICTION_HORIZON, 2)，预测的控制输入
    """
    total_cost = 0.0
    
    # 运行代价 (Stage cost)
    # 遍历预测时域中的每一步控制和其导致的状态
    for t in range(PREDICTION_HORIZON):
        # state_trajectory[0] 是当前状态 (current_state)
        # state_trajectory[t+1] 是应用 control_sequence[t] 后的预测状态
        predicted_state_at_t_plus_1 = state_trajectory[t+1]
        control_input_at_t = control_sequence[t]

        # 状态误差 e = x - x_target
        state_error = predicted_state_at_t_plus_1 - TARGET_STATE
        # 状态代价 e.T @ Q @ e
        total_cost += state_error.T @ Q_STATE_COST_MATRIX @ state_error
        
        # 控制代价 u.T @ R @ u
        total_cost += control_input_at_t.T @ R_CONTROL_COST_MATRIX @ control_input_at_t
        
    # 终端代价 (Terminal cost)
    # state_trajectory[-1] 是预测时域末端的状态
    terminal_state = state_trajectory[-1] 
    terminal_state_error = terminal_state - TARGET_STATE
    # 终端状态代价 e_terminal.T @ Q_terminal @ e_terminal
    total_cost += terminal_state_error.T @ Q_TERMINAL_COST_MATRIX @ terminal_state_error
    
    return total_cost

# --- MPPI 主循环 ---
def run_mppi_simulation():
    current_state = START_STATE.copy() # x, y, vx, vy
    
    nominal_control_sequence = np.zeros((PREDICTION_HORIZON, 2)) 

    actual_trajectory = [current_state.copy()]
    applied_controls = []
    time_points = [0.0]

    print("Running MPPI Simulation with new cost function...")
    for step in range(N_STEPS):
        # 检查是否到达目标 (位置和速度都接近目标)
        pos_dist = np.linalg.norm(current_state[:2] - TARGET_POS)
        vel_dist = np.linalg.norm(current_state[2:] - TARGET_VEL) # 检查速度
        if pos_dist < 0.2 and vel_dist < 0.2: # 更严格的停止条件
            print(f"目标已在第 {step} 步到达 (位置和速度)!")
            break
        if step == N_STEPS -1:
            print("仿真时间到，未严格到达目标。")


        control_perturbations = np.random.normal(
            loc=0.0, 
            scale=NOISE_SIGMA_CONTROL, 
            size=(N_SAMPLES, PREDICTION_HORIZON, 2)
        )

        sampled_control_sequences = nominal_control_sequence[np.newaxis, :, :] + control_perturbations # 确保nominal_control_sequence被正确广播
        sampled_control_sequences = np.clip(sampled_control_sequences, -ACCEL_MAX, ACCEL_MAX)

        costs = np.zeros(N_SAMPLES)
        
        for k in range(N_SAMPLES):
            temp_state_trajectory = [current_state.copy()] # 轨迹从当前状态开始
            # temp_state = current_state.copy() # No, use the list directly
            for t in range(PREDICTION_HORIZON):
                next_state = dynamics(temp_state_trajectory[-1], sampled_control_sequences[k, t])
                temp_state_trajectory.append(next_state)
            costs[k] = cost_function(np.array(temp_state_trajectory), sampled_control_sequences[k])

        min_cost = np.min(costs) 
        weights = np.exp(-(1.0 / LAMBDA) * (costs - min_cost)) 
        weights_sum = np.sum(weights)
        if weights_sum < 1e-8: # 如果所有权重都接近于0，可能LAMBDA太小或成本太高
            weights_sum = 1e-8 # 防止除零
            print(f"Warning: Low sum of weights at step {step}. Check LAMBDA or cost function.")
        weights /= weights_sum

        # weighted_perturbations = np.sum(weights[:, np.newaxis, np.newaxis] * control_perturbations, axis=0)
        # 我们是对完整的采样序列进行加权平均，而不是仅对扰动
        # nominal_control_sequence = np.sum(weights[:, np.newaxis, np.newaxis] * sampled_control_sequences, axis=0)
        # 实际上，MPPI标准做法是更新名义序列： nominal_new = nominal_old + weighted_perturbations
        # 或者，直接将加权平均后的 sampled_control_sequences 作为新的名义序列
        # 让我们坚持原始MPPI的扰动更新方式：
        
        weighted_perturbations = np.zeros_like(nominal_control_sequence)
        for t_step in range(PREDICTION_HORIZON):
            for i_sample in range(N_SAMPLES):
                weighted_perturbations[t_step, :] += weights[i_sample] * control_perturbations[i_sample, t_step, :]

        nominal_control_sequence += weighted_perturbations 
        nominal_control_sequence = np.clip(nominal_control_sequence, -ACCEL_MAX, ACCEL_MAX)

        actual_control_to_apply = nominal_control_sequence[0, :].copy()
        applied_controls.append(actual_control_to_apply)

        current_state = dynamics(current_state, actual_control_to_apply)
        actual_trajectory.append(current_state.copy())
        time_points.append((step + 1) * DT)

        nominal_control_sequence = np.roll(nominal_control_sequence, -1, axis=0)
        nominal_control_sequence[-1, :] = nominal_control_sequence[-2, :] # 或用零填充，或重复最后一个有效控制

        if step % 10 == 0:
            print(f"Step: {step}, Pos: {current_state[:2]}, Vel: {current_state[2:]}, Control: {actual_control_to_apply}, MinCost: {min_cost:.2f}")
            
    print("仿真结束.")
    return np.array(actual_trajectory), np.array(applied_controls), np.array(time_points)

a=time.time()

# --- 运行和绘图 (与之前相同) ---
if __name__ == "__main__":
    try:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("中文字体 WenQuanYi Micro Hei 未找到，将使用默认字体。")


    trajectory, controls, times = run_mppi_simulation()
    b=time.time()
    print(b-a)
    fig, axs = plt.subplots(2, 2, figsize=(16, 13)) # 调整图像大小
    fig.suptitle('MPPI控制二维质点运动仿真结果', fontsize=16)

    # 1. 运动轨迹
    axs[0, 0].plot(trajectory[:, 0], trajectory[:, 1], marker='.', linestyle='-', label='实际轨迹')
    axs[0, 0].plot(START_POS[0], START_POS[1], 'go', markersize=10, label='起点')
    axs[0, 0].plot(TARGET_POS[0], TARGET_POS[1], 'rx', markersize=10, mew=2, label='目标点')
    axs[0, 0].set_xlabel('X 位置')
    axs[0, 0].set_ylabel('Y 位置')
    axs[0, 0].set_title('运动轨迹')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    axs[0, 0].axis('equal')

    # 2. 位置 vs. 时间
    # trajectory 包含了初始状态，所以长度是 N_STEPS_TAKEN + 1
    # times 的长度与 trajectory 相同
    axs[0, 1].plot(times, trajectory[:, 0], label='X 位置')
    axs[0, 1].plot(times, trajectory[:, 1], label='Y 位置')
    axs[0, 1].axhline(TARGET_POS[0], color='r', linestyle='--', label='目标 X 位置')
    axs[0, 1].axhline(TARGET_POS[1], color='g', linestyle='--', label='目标 Y 位置')
    axs[0, 1].set_xlabel('时间 (s)')
    axs[0, 1].set_ylabel('位置')
    axs[0, 1].set_title('位置 vs. 时间')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # 3. 速度 vs. 时间
    axs[1, 0].plot(times, trajectory[:, 2], label='X 速度')
    axs[1, 0].plot(times, trajectory[:, 3], label='Y 速度')
    axs[1, 0].axhline(TARGET_VEL[0], color='r', linestyle='--', label='目标 X 速度')
    axs[1, 0].axhline(TARGET_VEL[1], color='g', linestyle='--', label='目标 Y 速度')
    axs[1, 0].set_xlabel('时间 (s)')
    axs[1, 0].set_ylabel('速度')
    axs[1, 0].set_title('速度 vs. 时间')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # 4. 控制输入 vs. 时间
    # controls 的长度是实际执行的步数
    control_times = times[:len(controls)] # 确保时间轴与控制数据对齐
    if len(controls) > 0:
        axs[1, 1].plot(control_times, controls[:, 0], label='X 加速度 (ax)')
        axs[1, 1].plot(control_times, controls[:, 1], label='Y 加速度 (ay)')
    axs[1, 1].set_xlabel('时间 (s)')
    axs[1, 1].set_ylabel('控制输入 (加速度)')
    axs[1, 1].set_title('控制输入 vs. 时间')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    axs[1, 1].set_ylim([-ACCEL_MAX*1.1, ACCEL_MAX*1.1])

    plt.savefig("MPPI, N=100, 权重10,10,1,1.png", format='png', bbox_inches='tight')# 'logs/{}epoch-{}steps.png'.format(epoch,steps))
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
