import numpy as np
import matplotlib.pyplot as plt
import time

"""CEM:基于采样的优化算法，通过迭代的方式，逐步优化一个概率分布，使得从这个分布中采样得到的解越来越好
    1.从一个参数化的概率分布中采样出一批候选解。
    2.评估这些候选解的好坏（计算它们的成本或奖励）。
    3.选取表现最好的一部分解（称为“精英样本”）。
    4.利用这些精英样本来更新概率分布的参数，使得新的分布更倾向于生成类似这些精英样本的解。
    5.重复以上步骤，直到满足某个停止条件（例如，分布参数收敛、解的质量不再显著提升，或达到最大迭代次数）。"""

"""加入了动态目标设计"""

# 系统参数
DT = 0.1  # 时间步长
MAX_SIM_TIME = 30.0 # 增加仿真时间以观察追踪效果
N_STEPS = int(MAX_SIM_TIME / DT)

# 初始目标状态和目标运动特性
INITIAL_TARGET_POS = np.array([15.0, 5.0])  # 目标初始位置
TARGET_VELOCITY_PROFILE = np.array([0.5, 0.3]) # 目标恒定速度 (dx/dt, dy/dt)

# 无人机初始状态
START_POS = np.array([0.0, 0.0])
START_VEL = np.array([0.0, 0.0])
START_STATE = np.concatenate((START_POS, START_VEL))

# 控制限制 (假设加速度限制)
ACCEL_MAX = 2.0

# MPC 参数 (通用)
PREDICTION_HORIZON = 10 # 预测时域 (步数)

# CEM-MPC 特定参数
N_SAMPLES_CEM = 100       # 每个CEM迭代的采样数量
N_ELITES_CEM = 10         # 精英样本数量 (通常是 N_SAMPLES_CEM 的 10-20%)
N_ITER_CEM = 5            # 每个MPC时间步的CEM迭代次数
INITIAL_STD_CEM = 0.5     # 控制序列采样的初始标准差
MIN_STD_CEM = 0.05        # 最小标准差，防止过早收敛
ALPHA_CEM = 0.7           # CEM内部均值和标准差更新的平滑因子 (0到1，1表示完全使用新精英样本的统计量)

# 代价函数权重
R_CONTROL_COST_MATRIX = 0.01 * np.eye(2)
Q_STATE_COST_MATRIX = np.diag([10.0, 10.0, 1.0, 1.0]) # 位置误差权重较高
Q_TERMINAL_COST_MATRIX = np.diag([100.0, 100.0, 10.0, 10.0]) # 终端位置误差权重更高

# 系统动力学模型 (二维质点)
# state = [x, y, vx, vy]
# control = [ax, ay]
def dynamics(state, control):
    x, y, vx, vy = state
    ax, ay = np.clip(control, -ACCEL_MAX, ACCEL_MAX) # 确保控制输入在限制内
    new_x = x + vx * DT
    new_y = y + vy * DT
    new_vx = vx + ax * DT
    new_vy = vy + ay * DT
    return np.array([new_x, new_y, new_vx, new_vy])

# --- 目标状态计算函数 ---
def get_target_state_at_time(sim_time_sec):
    """根据仿真时间计算目标当前的状态 (位置和速度)"""
    current_target_pos = INITIAL_TARGET_POS + TARGET_VELOCITY_PROFILE * sim_time_sec
    # 假设目标期望速度就是其自身的运动速度
    current_target_vel = TARGET_VELOCITY_PROFILE
    return np.concatenate((current_target_pos, current_target_vel))

# --- 代价函数 ---
# 需要接收当前MPC规划开始的仿真时间，以便预测目标未来的位置
def cost_function(state_trajectory, control_sequence, mpc_step_start_time_sec):
    total_cost = 0.0
    for t in range(PREDICTION_HORIZON):
        predicted_drone_state_at_t_plus_1 = state_trajectory[t+1]
        control_input_at_t = control_sequence[t]

        # 计算在预测的未来时间点 (t+1)*DT 时，目标将会处于的状态
        time_in_future = mpc_step_start_time_sec + (t + 1) * DT
        future_target_state_ref = get_target_state_at_time(time_in_future)

        state_error = predicted_drone_state_at_t_plus_1 - future_target_state_ref
        total_cost += state_error.T @ Q_STATE_COST_MATRIX @ state_error
        total_cost += control_input_at_t.T @ R_CONTROL_COST_MATRIX @ control_input_at_t

    # 终端代价：评估预测时域末端的状态
    terminal_drone_state = state_trajectory[-1]
    time_at_horizon_end = mpc_step_start_time_sec + PREDICTION_HORIZON * DT
    terminal_target_state_ref = get_target_state_at_time(time_at_horizon_end)

    terminal_state_error = terminal_drone_state - terminal_target_state_ref
    total_cost += terminal_state_error.T @ Q_TERMINAL_COST_MATRIX @ terminal_state_error
    return total_cost

# --- CEM-MPC 主循环 ---
def run_cem_mpc_simulation():
    current_state = START_STATE.copy()
    
    # 初始化用于CEM热启动的标称控制序列 (均值)
    mean_control_sequence_for_warm_start = np.zeros((PREDICTION_HORIZON, 2))

    actual_trajectory = [current_state.copy()]
    applied_controls = []
    time_points = [0.0]
    
    # 存储目标的轨迹用于绘图
    target_actual_trajectory_plot = [get_target_state_at_time(0.0)[:2]]


    print("Running CEM-MPC Simulation for Dynamic Target Tracking...")
    for step in range(N_STEPS):
        current_sim_time = step * DT # 当前仿真时间

        # 获取目标在当前时刻的实际状态
        current_actual_target_state = get_target_state_at_time(current_sim_time)
        target_actual_trajectory_plot.append(current_actual_target_state[:2])

        # 检查是否接近目标 (终止条件)
        pos_dist = np.linalg.norm(current_state[:2] - current_actual_target_state[:2])
        vel_dist = np.linalg.norm(current_state[2:] - current_actual_target_state[2:4]) # 比较与目标自身速度的差异

        # 对于动态追踪，持续追踪直到仿真结束，或者可以设置一个持续追踪的阈值
        # if pos_dist < 0.5 and vel_dist < 0.5: # 放宽一点追踪到的条件
        #     print(f"目标在第 {step} 步被接近 (位置误差: {pos_dist:.2f}m, 速度误差: {vel_dist:.2f}m/s)!")
            # 对于持续追踪，我们可能不在此处break，除非任务是到达并停留

        if step == N_STEPS -1:
            print("仿真时间到。")

        # CEM 优化循环开始
        cem_iter_mean = mean_control_sequence_for_warm_start.copy()
        cem_iter_std = np.full((PREDICTION_HORIZON, 2), INITIAL_STD_CEM)
        min_cost_this_mpc_step = float('inf')

        for cem_iter in range(N_ITER_CEM):
            # 1. 采样
            perturbations = np.random.normal(loc=0.0, scale=1.0,
                                             size=(N_SAMPLES_CEM, PREDICTION_HORIZON, 2))
            sampled_control_sequences = cem_iter_mean[np.newaxis, :, :] + \
                                        perturbations * cem_iter_std[np.newaxis, :, :]
            sampled_control_sequences = np.clip(sampled_control_sequences, -ACCEL_MAX, ACCEL_MAX)

            # 2. 评估
            costs_cem = np.zeros(N_SAMPLES_CEM)
            for k in range(N_SAMPLES_CEM):
                temp_state_trajectory_cem = [current_state.copy()]
                current_sim_state_for_rollout = current_state.copy()
                for t_pred in range(PREDICTION_HORIZON): # t_pred 是预测时域内的步数
                    next_state = dynamics(current_sim_state_for_rollout, sampled_control_sequences[k, t_pred])
                    temp_state_trajectory_cem.append(next_state)
                    current_sim_state_for_rollout = next_state
                # 调用代价函数时传入当前MPC规划开始的仿真时间
                costs_cem[k] = cost_function(np.array(temp_state_trajectory_cem),
                                             sampled_control_sequences[k],
                                             current_sim_time)
            
            min_cost_this_mpc_step = min(min_cost_this_mpc_step, np.min(costs_cem))

            # 3. 选择精英样本
            elite_indices = np.argsort(costs_cem)[:N_ELITES_CEM]
            elite_sequences = sampled_control_sequences[elite_indices]

            # 4. 更新分布参数
            new_mean = np.mean(elite_sequences, axis=0)
            new_std = np.std(elite_sequences, axis=0)

            cem_iter_mean = ALPHA_CEM * new_mean + (1 - ALPHA_CEM) * cem_iter_mean
            cem_iter_std = ALPHA_CEM * new_std + (1 - ALPHA_CEM) * cem_iter_std
            cem_iter_std = np.maximum(cem_iter_std, MIN_STD_CEM)

        optimal_control_sequence_this_step = cem_iter_mean
        actual_control_to_apply = optimal_control_sequence_this_step[0, :].copy()
        
        applied_controls.append(actual_control_to_apply)
        current_state = dynamics(current_state, actual_control_to_apply)
        actual_trajectory.append(current_state.copy())
        time_points.append(current_sim_time + DT)

        mean_control_sequence_for_warm_start = np.roll(optimal_control_sequence_this_step, -1, axis=0)
        mean_control_sequence_for_warm_start[-1, :] = optimal_control_sequence_this_step[-1, :] # 使用最后一个预测的控制进行填充

        if step % 20 == 0: # 减少打印频率
            print(f"SimTime: {current_sim_time:.1f}s, DronePos: {current_state[:2]}, TargetPos: {current_actual_target_state[:2]}, Control: {actual_control_to_apply}, Dist: {pos_dist:.2f}")
            
    print("仿真结束.")
    # 确保 target_actual_trajectory_plot 的长度与 time_points 一致或少一个（因为它在循环开始时添加）
    # 我们需要 target_actual_trajectory_plot 对应于 time_points
    # 由于 target_actual_trajectory_plot 在循环开始添加，它会比 actual_trajectory 多一个初始点
    # 但它也对应于 time_points 的每个时间点
    # 如果 time_points 长度为 N_STEPS+1, target_actual_trajectory_plot 也应为 N_STEPS+1
    return np.array(actual_trajectory), np.array(applied_controls), np.array(time_points), np.array(target_actual_trajectory_plot)

# --- 运行和绘图 ---
if __name__ == "__main__":
    try:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("中文字体 WenQuanYi Micro Hei 未找到，将使用默认字体。")

    simulation_start_time = time.time()
    trajectory, controls, times, target_positions_history = run_cem_mpc_simulation()
    simulation_end_time = time.time()
    print(f"仿真耗时: {simulation_end_time - simulation_start_time:.2f} 秒")

    fig, axs = plt.subplots(2, 2, figsize=(18, 15)) # 略微调大图像尺寸
    fig.suptitle('CEM-MPC 控制二维质点追踪动态目标仿真结果', fontsize=16)

    # 1. 运动轨迹
    axs[0, 0].plot(trajectory[:, 0], trajectory[:, 1], marker='.', linestyle='-', label='无人机轨迹', markersize=3)
    axs[0, 0].plot(target_positions_history[:, 0], target_positions_history[:, 1], marker='x', linestyle='--', color='magenta', label='目标轨迹', markersize=3,markevery=5)
    axs[0, 0].plot(START_POS[0], START_POS[1], 'go', markersize=10, label='无人机起点')
    axs[0, 0].plot(INITIAL_TARGET_POS[0], INITIAL_TARGET_POS[1], 'mo', markersize=8, label='目标起点')
    # 标记目标终点 (近似)
    if len(target_positions_history) > 0 :
        axs[0, 0].plot(target_positions_history[-1,0], target_positions_history[-1,1], 'mx', markersize=10, mew=2, label='目标终点(近似)')

    axs[0, 0].set_xlabel('X 位置 (m)')
    axs[0, 0].set_ylabel('Y 位置 (m)')
    axs[0, 0].set_title('运动轨迹对比')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    axs[0, 0].axis('equal')

    # 2. 位置 vs. 时间
    axs[0, 1].plot(times, trajectory[:, 0], label='无人机 X 位置')
    axs[0, 1].plot(times, trajectory[:, 1], label='无人机 Y 位置')
    axs[0, 1].plot(times, target_positions_history[:len(times), 0], linestyle='--', color='red', label='目标 X 位置')
    axs[0, 1].plot(times, target_positions_history[:len(times), 1], linestyle='--', color='green', label='目标 Y 位置')
    axs[0, 1].set_xlabel('时间 (s)')
    axs[0, 1].set_ylabel('位置 (m)')
    axs[0, 1].set_title('位置 vs. 时间')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # 3. 速度 vs. 时间
    axs[1, 0].plot(times, trajectory[:, 2], label='无人机 X 速度')
    axs[1, 0].plot(times, trajectory[:, 3], label='无人机 Y 速度')
    # 目标速度是恒定的
    axs[1, 0].axhline(TARGET_VELOCITY_PROFILE[0], color='r', linestyle='--', label='目标 X 速度')
    axs[1, 0].axhline(TARGET_VELOCITY_PROFILE[1], color='g', linestyle='--', label='目标 Y 速度')
    axs[1, 0].set_xlabel('时间 (s)')
    axs[1, 0].set_ylabel('速度 (m/s)')
    axs[1, 0].set_title('速度 vs. 时间')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # 4. 控制输入 vs. 时间
    control_times = times[:len(controls)]
    if len(controls) > 0:
        axs[1, 1].plot(control_times, controls[:, 0], label='X 加速度 (ax)')
        axs[1, 1].plot(control_times, controls[:, 1], label='Y 加速度 (ay)')
    axs[1, 1].axhline(ACCEL_MAX, color='k', linestyle=':', label=f'最大/最小加速度 ({ACCEL_MAX})')
    axs[1, 1].axhline(-ACCEL_MAX, color='k', linestyle=':')
    axs[1, 1].set_xlabel('时间 (s)')
    axs[1, 1].set_ylabel('控制输入 (加速度 m/s²)')
    axs[1, 1].set_title('控制输入 vs. 时间')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    axs[1, 1].set_ylim([-ACCEL_MAX*1.2, ACCEL_MAX*1.2])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("CEM-MPC_Dynamic_Target_Tracking.png", format='png', bbox_inches='tight')
    plt.show()
