import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time

a=time.time()
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei'] # 或者你安装的其他中文字体名，如 'SimHei', 'Noto Sans CJK SC'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 1. 系统定义 (System Definition)
# 我们考虑一个二维质点系统
# 状态 state = [x_pos, y_pos, x_velocity, y_velocity]
# 控制 control = [x_acceleration, y_acceleration]
DT = 0.1  # 离散时间步长 (s)

def system_model(state, control):
    """
    离散时间系统模型 
    预测给定当前状态和控制输入后的下一个状态
    """
    # 状态转移矩阵 A
    A = np.array([
        [1, 0, DT, 0],  # x_pos = x_pos + vx*DT
        [0, 1, 0, DT],  # y_pos = y_pos + vy*DT
        [0, 0, 1, 0],   # vx = vx + ax*DT (ax 来自控制输入 B*u)
        [0, 0, 0, 1]    # vy = vy + ay*DT (ay 来自控制输入 B*u)
    ])
    # 控制输入矩阵 B
    B = np.array([
        [0.5 * DT**2, 0          ], # x 位置受加速度影响 (0.5*ax*DT^2)
        [0,           0.5 * DT**2], # y 位置受加速度影响 (0.5*ay*DT^2)
        [DT,          0          ], # x 速度受加速度影响 (ax*DT)
        [0,           DT         ]  # y 速度受加速度影响 (ay*DT)
    ])
    return A @ state + B @ control # X'=(AX + Bu),X=[x,y,vx,vy]

# 2. MPC 参数定义
N_HORIZON = 5  # 预测时域 (多少个时间步向前看)

# 代价函数权重
# 控制输入的代价权重矩阵 R (通常是对角阵)
R_CONTROL_COST_MATRIX = R_CONTROL_COST = 0.01 * np.eye(2) # 对 ax, ay 的惩罚

# 状态误差的代价权重矩阵 Q (通常是对角阵)
Q_STATE_COST_MATRIX = np.diag([
    10.0,  # x 位置误差的权重
    10.0,  # y 位置误差的权重
    1.0,   # x 速度误差的权重 (鼓励速度在目标点为0)
    1.0    # y 速度误差的权重 (鼓励速度在目标点为0)
])
# 终端状态误差的代价权重矩阵 Q_terminal (通常比Q更大，强调最终目标)
Q_TERMINAL_COST_MATRIX = np.diag([
    100.0, # 终端 x 位置误差
    100.0, # 终端 y 位置误差
    10.0,  # 终端 x 速度误差
    10.0   # 终端 y 速度误差
])


# 3. MPC 优化目标函数
def mpc_objective_function(controls_flat, current_state_np, target_state_np):
    """
    计算在预测时域 N_HORIZON 内，给定一系列控制输入后的总代价。
    controls_flat: 展平的控制输入序列 [ax0,ay0, ax1,ay1, ..., ax(N-1),ay(N-1)]
    current_state_np: 预测开始时的当前状态
    target_state_np: 期望的目标状态 [target_x, target_y, target_vx, target_vy]
    """ 
    total_cost = 0.0
    predicted_state = np.copy(current_state_np) # 
    num_control_inputs_per_step = 2 # ax, ay
    
    # 将展平的控制序列重塑为 (N_HORIZON, num_control_inputs_per_step)
    controls_sequence = controls_flat.reshape(N_HORIZON, num_control_inputs_per_step)

    for i in range(N_HORIZON):
        control_input = controls_sequence[i, :] # 当前步的控制输入，两个
        
        # 计算状态误差 (predicted_state - target_state)
        state_error = predicted_state - target_state_np
        
        # 累加代价
        if i < N_HORIZON - 1: # 非终端状态的代价
            total_cost += state_error.T @ Q_STATE_COST_MATRIX @ state_error
        else: # 终端状态的代价 (预测时域的最后一个状态)
            total_cost += state_error.T @ Q_TERMINAL_COST_MATRIX @ state_error
            
        # 累加控制输入的代价
        total_cost += control_input.T @ R_CONTROL_COST_MATRIX @ control_input # 代价函数只有一个，全加到一起
        
        # 预测下一个状态
        predicted_state = system_model(predicted_state, control_input)
        
    return total_cost

# 4. MPC 控制器
def get_mpc_control_action(current_state, target_state, control_input_bounds_list):
    """
    计算当前状态下的最优控制输入。
    current_state: 当前系统状态
    target_state: 目标系统状态
    control_input_bounds_list: 单个时间步控制输入的界限，例如 [(-max_ax, max_ax), (-max_ay, max_ay)]
    """
    num_control_inputs_per_step = 2 # ax, ay
    # 控制输入的初始猜测值 (例如，全零) SQP对初值还挺敏感...
    initial_guess_controls = np.zeros(N_HORIZON * num_control_inputs_per_step)

    # 为优化器定义整个预测时域内每个控制输入的界限
    # control_input_bounds_list 是单个时间步的界限, 我们需要为 N_HORIZON 步重复它
    bounds_for_optimizer = control_input_bounds_list * N_HORIZON

    # 执行优化
    optimization_result = minimize(
        mpc_objective_function, # 代价函数，最小化目标
        initial_guess_controls, # 猜测初值
        args=(current_state, target_state), # args=(current_state, target_state) 会被传递给 mpc_objective_function
        method='SLSQP',  # 序列最小二乘规划，适合处理带边界约束的非线性问题
        bounds=bounds_for_optimizer, # 控制变量范围
        options={'disp': False, 'maxiter': 150} # disp控制是否显示优化器输出, maxiter是最大迭代次数
    )

    if not optimization_result.success:
        print(f"警告: MPC 优化失败或未收敛: {optimization_result.message}")
        # 可以选择返回零控制或上次的有效控制作为备用策略
        # 为简化演示，我们仍然使用可能次优的结果 (如果存在)
        # return np.zeros(num_control_inputs_per_step)


    optimal_controls_flat = optimization_result.x
    # 提取并返回预测时域内的第一个控制输入，用于施加到实际系统
    first_control_input_to_apply = optimal_controls_flat[:num_control_inputs_per_step]
    
    return first_control_input_to_apply

# --- 5. 仿真与演示 ---
def run_mpc_simulation_demo():
    """
    运行一个MPC控制二维质点到达目标的仿真。
    """
    # 初始状态: [x_pos, y_pos, x_vel, y_vel]
    initial_state = np.array([0.0, 0.0, 0.0, 0.0])  # 从原点开始，初始速度为0
    # 目标状态: [target_x, target_y, target_vx, target_vy]
    target_state_final = np.array([20.0, 10.0, 0.0, 0.0])   # 目标位置，期望在目标点速度为0

    # 控制输入约束 (例如，最大加速度)
    max_abs_acceleration = 2.0  # x或y方向的最大绝对加速度值
    control_bounds = [(-max_abs_acceleration, max_abs_acceleration), # ax 的界限
                      (-max_abs_acceleration, max_abs_acceleration)] # ay 的界限

    num_simulation_steps = 200 # 最大仿真步数
    current_state_sim = np.copy(initial_state)

    # 用于存储历史数据以供绘图
    sim_state_history = [np.copy(current_state_sim)]
    sim_control_history = []

    print(f"开始MPC仿真：二维质点运动控制")
    print(f"初始状态: {initial_state}")
    print(f"目标状态: {target_state_final}")
    print(f"预测时域 (N): {N_HORIZON}, 时间步长 (DT): {DT}")
    print(f"控制代价权重 (R) 对角线: {np.diag(R_CONTROL_COST_MATRIX)}")
    print(f"状态代价权重 (Q) 对角线: {np.diag(Q_STATE_COST_MATRIX)}")
    print(f"终端状态代价权重 (Q_terminal) 对角线: {np.diag(Q_TERMINAL_COST_MATRIX)}")


    for step in range(num_simulation_steps):
        # 检查是否到达目标 (位置和速度都在一个小的容忍范围内)
        position_error_norm = np.linalg.norm(current_state_sim[:2] - target_state_final[:2])
        velocity_error_norm = np.linalg.norm(current_state_sim[2:] - target_state_final[2:])
        
        if position_error_norm < 0.2 and velocity_error_norm < 0.1:
            print(f"\n在第 {step} 步成功到达目标!")
            break

        # 从MPC获取最优控制动作
        optimal_control_action = get_mpc_control_action(current_state_sim, target_state_final, control_bounds)
        
        # 将计算出的第一个控制输入施加到系统模型
        current_state_sim = system_model(current_state_sim, optimal_control_action)

        # 存储历史数据
        sim_state_history.append(np.copy(current_state_sim))
        sim_control_history.append(np.copy(optimal_control_action))
        
        if step % 10 == 0 or step == num_simulation_steps -1 : # 每10步打印一次信息
            print(f"步 {step:3d}: 位置: [{current_state_sim[0]:6.2f}, {current_state_sim[1]:6.2f}], "
                  f"速度: [{current_state_sim[2]:6.2f}, {current_state_sim[3]:6.2f}], "
                  f"控制: [{optimal_control_action[0]:6.2f}, {optimal_control_action[1]:6.2f}]")
            
    if step == num_simulation_steps - 1 and (position_error_norm >= 0.2 or velocity_error_norm >= 0.1):
        print("\n仿真结束: 已达到最大步数但未精确到达目标。")
    b=time.time()
    print(b-a)
    # 将历史数据转换为numpy数组，方便后续处理和绘图
    sim_state_history_np = np.array(sim_state_history)
    sim_control_history_np = np.array(sim_control_history)

    # --- 6. 结果可视化 ---
    plt.figure(figsize=(12, 10))
    plt.suptitle("MPC 控制二维质点运动仿真结果", fontsize=16)

    # 子图1: 运动轨迹
    plt.subplot(2, 2, 1)
    plt.plot(sim_state_history_np[:, 0], sim_state_history_np[:, 1], 'b-o', markersize=3, label='实际轨迹')
    plt.scatter([initial_state[0]], [initial_state[1]], color='green', s=100, label='起点', zorder=5)
    plt.scatter([target_state_final[0]], [target_state_final[1]], color='red', s=100, marker='x', label='目标点', zorder=5)
    plt.xlabel('X 位置')
    plt.ylabel('Y 位置')
    plt.title('运动轨迹')
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # 保持X和Y轴等比例缩放

    # 子图2: 位置随时间变化
    time_axis = np.arange(sim_state_history_np.shape[0]) * DT
    plt.subplot(2, 2, 2)
    plt.plot(time_axis, sim_state_history_np[:, 0], label='X 位置')
    plt.plot(time_axis, sim_state_history_np[:, 1], label='Y 位置')
    plt.axhline(target_state_final[0], color='r', linestyle='--', linewidth=0.8, label='目标 X 位置')
    plt.axhline(target_state_final[1], color='g', linestyle='--', linewidth=0.8, label='目标 Y 位置')
    plt.xlabel('时间 (s)')
    plt.ylabel('位置')
    plt.title('位置 vs. 时间')
    plt.legend()
    plt.grid(True)

    # 子图3: 速度随时间变化
    plt.subplot(2, 2, 3)
    plt.plot(time_axis, sim_state_history_np[:, 2], label='X 速度')
    plt.plot(time_axis, sim_state_history_np[:, 3], label='Y 速度')
    plt.axhline(target_state_final[2], color='r', linestyle='--', linewidth=0.8, label='目标 X 速度')
    plt.axhline(target_state_final[3], color='g', linestyle='--', linewidth=0.8, label='目标 Y 速度')
    plt.xlabel('时间 (s)')
    plt.ylabel('速度')
    plt.title('速度 vs. 时间')
    plt.legend()
    plt.grid(True)

    # 子图4: 控制输入随时间变化 (注意控制输入比状态少一个时间点)
    if sim_control_history_np.shape[0] > 0:
        control_time_axis = np.arange(sim_control_history_np.shape[0]) * DT
        plt.subplot(2, 2, 4)
        plt.plot(control_time_axis, sim_control_history_np[:, 0], label='X 加速度 (ax)')
        plt.plot(control_time_axis, sim_control_history_np[:, 1], label='Y 加速度 (ay)')
        plt.xlabel('时间 (s)')
        plt.ylabel('控制输入 (加速度)')
        plt.title('控制输入 vs. 时间')
        plt.legend()
        plt.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局防止标题重叠
    plt.show()
    
    # --- 7. 如何用于指导无人机训练 ---
    print("\n--- 如何使用此MPC的输出指导您的无人机训练算法 ---")
    print("仿真生成的 `sim_state_history_np` (状态历史) 和 `sim_control_history_np` (控制历史)")
    print("可以作为高质量的“专家演示数据”，用于：")
    print("1. 行为克隆 (Behavioral Cloning): 训练一个神经网络来模仿MPC的行为，即从状态映射到动作。")
    print("2. 离线强化学习 (Offline Reinforcement Learning): 使用这些数据训练强化学习智能体，而无需与环境实时交互。")
    print("3. 在线强化学习中的奖励塑造 (Reward Shaping in Online RL):")
    print("   - 可以将MPC计算出的代价作为RL智能体奖励函数的一部分。")
    print("   - 或者，将智能体的行为与MPC生成的轨迹进行比较，偏差越小奖励越高。")
    print("\n对于您的无人机“穿门”任务，您需要：")
    print("  - 调整 `system_model` 以更精确地反映您的无人机动力学。")
    print("  - 将 `target_state_final` 设置为门的位置（可能是门中心或门后一点）。")
    print("  - 如果门是动态的，`target_state_final` 在MPC的每个优化步骤中可能需要基于对门未来位置的预测进行更新。")
    print("  - 考虑在代价函数中加入对碰到门框的惩罚，或者通过约束来避免碰撞。")
    print("  - 您之前提到的时变代价矩阵 Qtr(ttra, h) 对于精确穿门非常重要，可以在 `mpc_objective_function` 中实现，")
    print("    使得在接近“最佳穿越时间 (ttra)”时，跟踪门中心的状态误差权重最高。")

    return sim_state_history_np, sim_control_history_np

run_mpc_simulation_demo()