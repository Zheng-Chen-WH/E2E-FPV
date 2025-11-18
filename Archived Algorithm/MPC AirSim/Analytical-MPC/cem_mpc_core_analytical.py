import numpy as np
import torch
import time
import math
import config as cfg
from analytical_model_gpu import SimpleFlightDynamicsTorch

""" 将参数从 config.py传递到 main.py，再由 main.py 传递给 mpc_core.py
    这种方式通常被称为**依赖注入（Dependency Injection）**的一种形式
    以下是这样做的一些主要原因和好处：
    1.清晰的依赖关系 (Clearer Dependencies / Explicitness):
        当 adaptive_cem_mpc_episode 函数的参数列表明确列出了它所需要的配置项（即使是通过几个字典参数传入）时，
        任何阅读或使用这个函数的人都能立即明白这个函数依赖哪些外部配置才能工作。
        如果函数内部直接从全局的 cfg 模块导入，这些依赖关系就变得不那么明显，你需要深入函数内部去查找它实际使用了 cfg 中的哪些变量。
    2.增强的模块化和解耦 (Increased Modularity and Decoupling):
        mpc_core.py 模块变得更加独立。它不再强依赖于一个名为 cfg (或 config) 的特定文件或模块必须存在,
        只要调用者能提供符合其参数接口要求的数据，mpc_core.py 就能工作。
    3.更高的可测试性 (Improved Testability):
        测试 adaptive_cem_mpc_episode 函数时，可以非常容易地在测试脚本中创建不同的配置字典，并将它们传递给这个函数，来测试它在不同参数设置下的行为。
        如果函数内部直接导入 cfg，那么在测试时，你可能需要修改全局的 cfg 文件/模块，这会使测试变得更加困难和脆弱。
    4.灵活性和可配置性 (Enhanced Flexibility and Configurability):
        main.py 作为程序的入口和协调者，可以有更复杂的逻辑来决定传递给 mpc_core 的参数。
        例如，这些参数可能部分来自 config.py，部分来自命令行参数，部分来自环境变量，或者根据某些条件动态计算得出。
        main.py 负责整合这些配置，然后统一传递给核心逻辑。
        如果想在同一个程序中运行多个MPC实例，每个实例使用不同的参数集，通过参数传递的方式会更容易实现。
    5.避免全局状态问题 (Avoiding Global State Issues):
        过度依赖全局变量（如直接从 config.py 导入）会使代码更难推理，因为函数行为可能受到程序中任何地方对全局配置的修改的影响。
        通过参数传递，函数的行为主要由其接收到的参数决定，更加可预测。"""

def cost_function_gpu(
    predicted_states_batch,   
    control_sequences_batch,      
    current_mpc_target_state_sequence_gpu,
    q_state_cost_matrix_gpu,
    r_control_cost_matrix_gpu,
    q_terminal_cost_matrix_gpu
    ):
    """
    在GPU上批量计算轨迹成本。
    Args:
        predicted_states_batch: 预测的状态轨迹批次 (scaled)。
                                 形状: (n_samples, PREDICTION_HORIZON + 1, state_dim)
        control_sequences_batch: 采样的控制序列批次 (unscaled)。
                                  形状: (n_samples, PREDICTION_HORIZON, action_dim)
        current_mpc_target_state_sequence_gpu: 当前MPC的目标状态序列 (scaled)。
                                       形状: (PREDICTION_HORIZON, state_dim)
    Returns:
        total_costs_batch: 每个样本的总成本。形状: (n_samples,)
    """
    prediction_horizon = control_sequences_batch.shape[1] # 从控制序列获取实际的控制序列长度H

    # 计算状态运行成本
    # 提取用于计算状态成本的预测状态 (从 t=1 到 t=H)
    running_predicted_states = predicted_states_batch[:, 1:prediction_horizon + 1, :]

    # current_mpc_target_state_sequence_gpu的形状是(H, state_dim)以广播到(n_samples, H, state_dim)
    state_error = running_predicted_states - current_mpc_target_state_sequence_gpu.unsqueeze(0)
    # unsqueeze在指定位置加一个1的维度，squeeze只能减去指定位置为1的维度
    
    # 运行状态代价
    # 沿着预测序列horizon (h)求和, 还要对每个样本(k)沿着dimensions(i,j)求和
    # 爱因斯坦和约定einsum: 'khi,ij,khj->k'，通过指定字符串定义张量操作
    # 对h, i, j求和， 稍后对h (horizon) 求和.
    # 输入部分 (khi,ij,khj): 描述了参与运算的三个输入张量的维度。逗号分隔每个张量的标签。
    # 输出部分 (k): 描述了运算结果张量的维度。
    # 爱因斯坦求和的核心规则：
    # 1.重复的索引意味着乘积和求和（缩并 Contract）: 
    # 如果一个索引字母同时出现在输入部分的不同张量标签中，或者在同一个张量标签中多次出现（这里没有这种情况），那么运算结果将沿着这些重复的维度进行乘积累加。
    # 2.未出现在输出部分的索引意味着被求和掉: 如果一个索引字母出现在输入部分的标签中，但没有出现在 -> 右边的输出部分标签中，那么结果张量将沿着这个维度进行求和。
    # 3.出现在输出部分的索引会被保留: 如果一个索引字母出现在输入部分，并且也出现在输出部分，那么这个维度将在结果张量中被保留下来。
    state_costs = torch.einsum('khi,ij,khj->k',
                               state_error,
                               q_state_cost_matrix_gpu,
                               state_error)

    # 控制代价
    # control_sequences_batch is (n_samples, H, action_dim)
    # r_control_cost_matrix_gpu is (action_dim, action_dim)
    control_costs = torch.einsum('khi,ij,khj->k',
                                 control_sequences_batch,
                                 r_control_cost_matrix_gpu,
                                 control_sequences_batch)

    # 终端状态代价
    terminal_state_batch = predicted_states_batch[:, -1, :]  # 目标是最后一个状态
    terminal_target_state = current_mpc_target_state_sequence_gpu[-1, :] # 形状: (n_samples, state_dim)
    
    terminal_state_error = terminal_state_batch - terminal_target_state # 广播
    
    terminal_costs = torch.einsum('ki,ij,kj->k',
                                  terminal_state_error,
                                  q_terminal_cost_matrix_gpu,
                                  terminal_state_error)
    # print("state:", state_costs,"\ncontrol", control_costs, "\nterminal", terminal_costs)

    # 总成本
    total_costs_batch = state_costs + control_costs + terminal_costs
    return total_costs_batch

# Adaptive CEM MPC主算法
def adaptive_cem_mpc_episode(episode_num, airsim_env, cem_hyperparams, mpc_params):
    # 从参数字典中解包
    # CEM参数
    prediction_horizon = cem_hyperparams['prediction_horizon']
    n_samples_cem = cem_hyperparams['n_samples']
    n_elites_cem = cem_hyperparams['n_elites']
    n_iter_cem = cem_hyperparams['n_iter']
    initial_std_cem = cem_hyperparams['initial_std'] # This will depend on action space
    min_std_cem = cem_hyperparams['min_std']         # This will depend on action space
    alpha_cem = cem_hyperparams['alpha']

    # MPC参数
    waypoint_pass_threshold_y = mpc_params['waypoint_pass_threshold_y']
    max_sim_time_per_episode = mpc_params['max_sim_time_per_episode']
    dt_mpc = mpc_params['dt']
    control_max = mpc_params['control_max']
    control_min = mpc_params['control_min']
    q_state_matrix_gpu=mpc_params["q_state_matrix_gpu"]
    r_control_matrix_gpu=mpc_params["r_control_matrix_gpu"]
    q_terminal_matrix_gpu=mpc_params["q_terminal_matrix_gpu"]
    static_q_state_matrix_gpu=mpc_params["static_q_state_matrix_gpu"]
    static_r_control_matrix_gpu=mpc_params["static_r_control_matrix_gpu"]
    static_q_terminal_matrix_gpu=mpc_params["static_q_terminal_matrix_gpu"]
    action_dim = mpc_params['action_dim'] # 4马达PWM输入
    state_dim = mpc_params['state_dim'] #姿态用wxyz四元数表示

    # 环境初始化
    (current_true_state, final_target_state, waypoints_y, # These are numpy arrays
     door_z_positions, door_x_positions, door_x_velocities,
     episode_start_time, door_parameters_dict) = airsim_env.reset()

    device = cfg.device  # 从config中获取设备

    n_steps_this_episode = int(max_sim_time_per_episode / dt_mpc)
    mean_control_sequence_warm_start = np.zeros((prediction_horizon, action_dim))
    
    # 记录轨迹画图用
    actual_trajectory_log = [current_true_state.copy()]
    applied_controls_log = []
    time_points_log = [0.0]
    
    reached_final_target_flag = False
    steps_taken_in_episode = 0

    # 航路点管理
    def _get_current_waypoint_index(current_y_pos, waypoints_y_list, threshold):
        # 航路点：[start_y, door1_y, door2_y, final_target_y]
        # index 0: target is door1 (at waypoints_y_list[1])
        # index 1: target is door2 (at waypoints_y_list[2])
        # index 2: target is final_target (at waypoints_y_list[3])
        if current_y_pos < waypoints_y_list[1] + threshold: # 靠近第一个门
            return 0
        elif current_y_pos < waypoints_y_list[2] + threshold: # elif确保已经越过了第一个门
            return 1
        else: # else确保越过了第二个门
            return 2

    def get_mpc_target_sequence(current_drone_state):
        # 确定当前目标
        # waypoints_y: [drone_start_y, door1_y, door2_y, final_target_y]
        # door_parameters_dict: "initial_x_pos", "amplitude", "frequency", "deviation"
        # door_z_positions: [door1_z, door2_z]
        
        current_idx = _get_current_waypoint_index(current_drone_state[1], waypoints_y, waypoint_pass_threshold_y)
        target_sequence_np = np.zeros((prediction_horizon, state_dim))

        if current_idx < len(airsim_env.door_frames): # 目标是门
            door_info_idx = current_idx # 门的索引（0或1）
            target_door_y = waypoints_y[current_idx + 1] # 目标门的y坐标
            time_this_step = time.time()-door_parameters_dict["start_time"]

            for i in range(prediction_horizon):
                # 预测门在t + dt*i时的位置
                t_future = time_this_step + dt_mpc * i
                
                # 预测门的x位置
                pred_door_x = door_parameters_dict["initial_x_pos"][door_info_idx] + \
                              door_parameters_dict["amplitude"] * math.sin(
                                  2 * math.pi * door_parameters_dict["frequency"] * t_future +
                                  door_parameters_dict["deviation"][door_info_idx]
                              )
                # 计算门的x速度
                pred_door_x_vel = 2 * math.pi * door_parameters_dict["frequency"] * \
                                  door_parameters_dict["amplitude"] * math.cos(
                                      2 * math.pi * door_parameters_dict["frequency"] * t_future +
                                      door_parameters_dict["deviation"][door_info_idx]
                                  )
                
                # MPC目标：以指定速度穿过指定点
                target_pos_x = pred_door_x
                target_pos_y = target_door_y + waypoint_pass_threshold_y # 对准门的y位置+阈值
                target_pos_z = door_z_positions[door_info_idx] - 2 # z位置在门底部，-2m大约在门中心

                target_vel_x = pred_door_x_vel # 匹配门的x速度
                target_vel_y = 4.0  # 穿越门的目标速度
                target_vel_z = 0.0

                target_sequence_np[i, :] = [ # 13维状态
                    target_pos_x, target_pos_y, target_pos_z,
                    target_vel_x, target_vel_y, target_vel_z,
                    0.707, 0.0 ,0.0 ,0.707 ,0.0 ,0.0 ,0.0 # Zero attitude and angular velocity target
                ]
        else: # 目标是最终目标
            target_sequence_np = np.tile(final_target_state, (prediction_horizon, 1))
            
        return torch.tensor(target_sequence_np, dtype=torch.float32, device=device)

    # 初始化MPC目标
    current_mpc_target_sequence_gpu = get_mpc_target_sequence(current_true_state) # (H, 13)
    
    # 初始化解析模型类
    analytical_model_instance = None
    analytical_model_instance = SimpleFlightDynamicsTorch(
        n_samples_cem, dt=0.1, dtype=torch.float32       # dt是动力学模型内部积分步长
    )
    examiner_instance = SimpleFlightDynamicsTorch(       # 检验并行化之后模型可靠性
        1, dt=0.05, dtype=torch.float32
    )

    print(f"\n--- 开始第 {episode_num + 1} 次训练 ---")

    for step_idx in range(n_steps_this_episode):
        steps_taken_in_episode = step_idx + 1

        actual_control_to_apply = np.random.uniform(control_min, control_max, size=action_dim)

        cem_iter_mean_gpu = torch.tensor(mean_control_sequence_warm_start, dtype=torch.float32, device=device)
        cem_iter_std_gpu = torch.full((prediction_horizon, action_dim), initial_std_cem, dtype=torch.float32, device=device)
        
        # CEM优化
        for cem_idx in range(n_iter_cem):
            
            # 采样控制序列
            perturbations_gpu = torch.normal(mean=0.0, std=1.0,
                                                size=(n_samples_cem, prediction_horizon, action_dim),
                                                device=device)
            sampled_controls_gpu = cem_iter_mean_gpu.unsqueeze(0) + \
                                                perturbations_gpu * cem_iter_std_gpu.unsqueeze(0)
            
            # 剪裁动作范围
            sampled_controls_gpu = torch.clip(sampled_controls_gpu, control_min, control_max)

            # 初始状态
            current_true_state_gpu = torch.tensor(current_true_state, dtype=torch.float32, device=device)
            current_true_state_gpu = current_true_state_gpu.unsqueeze(0).repeat(n_samples_cem, 1)

            # simulate_horizon需求动作序列维度(H, K, ActionDim=4)
            # sampled_controls_gpu的维度是(K, H, ActionDim=4)
            predicted_trajectory_batch = analytical_model_instance.simulate_horizon(
                    current_true_state_gpu, # 初始状态
                    sampled_controls_gpu.permute(1, 0, 2), # 转置成(H, K, 4)
                    dt_mpc)

            # 计算代价
            current_idx = _get_current_waypoint_index(current_true_state[1], waypoints_y, waypoint_pass_threshold_y)
            if current_idx == 2:
                costs_cem_gpu = cost_function_gpu(
                predicted_trajectory_batch,  # (K, H+1, 12) SCALED states
                sampled_controls_gpu,      # (K, H, ActionDim) UNSCALED controls
                current_mpc_target_sequence_gpu,
                static_q_state_matrix_gpu,
                static_r_control_matrix_gpu, # Use the R matrix appropriate for current action space
                static_q_terminal_matrix_gpu)
            else:
                costs_cem_gpu = cost_function_gpu(
                    predicted_trajectory_batch,  # (K, H+1, 12) SCALED states
                    sampled_controls_gpu,      # (K, H, ActionDim) UNSCALED controls
                    current_mpc_target_sequence_gpu,
                    q_state_matrix_gpu,
                    r_control_matrix_gpu, # Use the R matrix appropriate for current action space
                    q_terminal_matrix_gpu
                )
            
            # 选择精英群体
            elite_indices = torch.argsort(costs_cem_gpu)[:n_elites_cem]
            elite_sequences_gpu = sampled_controls_gpu[elite_indices]
            
            # 更新mean和std
            new_mean_gpu = torch.mean(elite_sequences_gpu, dim=0)
            new_std_gpu = torch.std(elite_sequences_gpu, dim=0) # 有偏方差
            
            cem_iter_mean_gpu = alpha_cem * new_mean_gpu + (1 - alpha_cem) * cem_iter_mean_gpu
            cem_iter_std_gpu = alpha_cem * new_std_gpu + (1 - alpha_cem) * cem_iter_std_gpu
            cem_iter_std_gpu = torch.maximum(cem_iter_std_gpu, torch.tensor(min_std_cem, dtype=torch.float32, device=device))

        optimal_control_sequence = cem_iter_mean_gpu.cpu().numpy()
        actual_control_to_apply = optimal_control_sequence[0, :].copy()
        print("动作：", actual_control_to_apply)
        
        # warm start
        mean_control_sequence_warm_start = np.roll(optimal_control_sequence, -1, axis=0)
        mean_control_sequence_warm_start[-1, :] = optimal_control_sequence[-1, :].copy()

        # 检验并行化模型可靠性
        examined_trajectory_batch = examiner_instance.simulate_horizon(
                    current_true_state_gpu[0,:].unsqueeze(0), # 初始状态取一个
                    cem_iter_mean_gpu.unsqueeze(1), # 转置成(H, K, 4)
                    dt_mpc)
        print("model prediction:", examined_trajectory_batch[0][1])

        # AirSim执行指令
        next_true_state, _, _, collided = airsim_env.step(actual_control_to_apply)
        print("actual state:", next_true_state)
        # 终止条件
        pos_dist_to_final = np.linalg.norm(next_true_state[:3] - final_target_state[:3])
        
        if pos_dist_to_final < cfg.POS_TOLERANCE:
            print(f"第 {episode_num + 1} 次训练: 最终目标已在第 {steps_taken_in_episode} 步到达!")
            reached_final_target_flag = True
            break
        if collided: # or np.linalg.norm(current_true_state[0:2]-next_true_state[0:2])<0.01: # Original condition
            print(f"第 {episode_num + 1} 次训练: 在第 {steps_taken_in_episode} 步发生碰撞。")
            if steps_taken_in_episode < 10: 
                time.sleep(0.5)
            break

        if steps_taken_in_episode >= n_steps_this_episode:
            print(f"第 {episode_num + 1} 次训练: 仿真时间到，最终目标状态: {'到达' if reached_final_target_flag else '未到达'}")
            break

        # 更新状态
        applied_controls_log.append(actual_control_to_apply.copy())
        current_true_state = next_true_state.copy()
        actual_trajectory_log.append(current_true_state.copy())
        time_points_log.append(steps_taken_in_episode * dt_mpc)

        # 更新目标
        current_mpc_target_sequence_gpu = get_mpc_target_sequence(current_true_state)
        # print(current_mpc_target_sequence_gpu)
        if step_idx % 1 == 0: # Print frequency
             print(f"\nEp {episode_num+1}, Step {steps_taken_in_episode},"
                  f"Pos: [{current_true_state[0]:.1f},{current_true_state[1]:.1f},{current_true_state[2]:.1f}],"
                  f"Action: {actual_control_to_apply[0]:.2f},{actual_control_to_apply[1]:.2f},{actual_control_to_apply[2]:.2f},{actual_control_to_apply[3]:.2f}"
                  )

    return (np.array(actual_trajectory_log), np.array(applied_controls_log),
            np.array(time_points_log), reached_final_target_flag, steps_taken_in_episode)