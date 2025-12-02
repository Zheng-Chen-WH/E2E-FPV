import numpy as np
import torch
import time
import math
import config as cfg
from analytical_model_gpu import SimpleFlightDynamicsTorch, pt_quat_rotate_vector
import torch.nn.functional as F

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

# Adaptive CEM MPC主算法，用类表示
class CEM_MPC():
    def __init__(self, cem_hyperparams, mpc_params): 
        # 从参数字典中解包
        # CEM参数
        self.prediction_horizon = cem_hyperparams['prediction_horizon']
        self.n_samples_cem = cem_hyperparams['n_samples']
        self.n_elites_cem = cem_hyperparams['n_elites']
        self.n_iter_cem = cem_hyperparams['n_iter']
        self.initial_std_cem = cem_hyperparams['initial_std'] # This will depend on action space
        self.min_std_cem = cem_hyperparams['min_std']         # This will depend on action space
        self.alpha_cem = cem_hyperparams['alpha']

        # MPC参数
        self.waypoint_pass_threshold_y = mpc_params['waypoint_pass_threshold_y']
        self.dt_mpc = mpc_params['dt']
        self.control_max = mpc_params['control_max']
        self.control_min = mpc_params['control_min']
        self.q_state_matrix_gpu=mpc_params["q_state_matrix_gpu"]
        self.r_control_matrix_gpu=mpc_params["r_control_matrix_gpu"]
        self.q_terminal_matrix_gpu=mpc_params["q_terminal_matrix_gpu"]
        self.q_state_matrix_gpu_two=mpc_params["q_state_matrix_gpu_two"]
        self.r_control_matrix_gpu_two=mpc_params["r_control_matrix_gpu_two"]
        self.q_terminal_matrix_gpu_two=mpc_params["q_terminal_matrix_gpu_two"]
        self.static_q_state_matrix_gpu=mpc_params["static_q_state_matrix_gpu"]
        self.static_r_control_matrix_gpu=mpc_params["static_r_control_matrix_gpu"]
        self.static_q_terminal_matrix_gpu=mpc_params["static_q_terminal_matrix_gpu"]
        self.action_dim = mpc_params['action_dim'] # 4马达PWM输入
        self.state_dim = mpc_params['state_dim'] #姿态用wxyz四元数表示
        self.device = mpc_params['device']  # 获取设备
        self.mean_control_sequence_warm_start = np.zeros((self.prediction_horizon, self.action_dim))
        self.pos_tolerence = mpc_params['pos_tolerence']
        self.align_cost = mpc_params['align_cost']
        self.vel_align_cost = mpc_params['vel_align_cost']
     
        # 记录轨迹画图用
        # actual_trajectory_log = [current_true_state.copy()]
        # applied_controls_log = []
        # time_points_log = [0.0]
        
        self.reached_final_target_flag = False
        self.steps_taken_in_episode = 0

    def cost_function_gpu(self,
        predicted_states_batch,   
        control_sequences_batch,      
        current_mpc_target_state_sequence_gpu,
        q_state_cost_matrix_gpu,
        r_control_cost_matrix_gpu,
        q_terminal_cost_matrix_gpu,
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

        # 新增相机对准代价计算】
        
        # 提取轨迹中每个时间步的位置和姿态
        # running_predicted_states: (K, H, 13)
        pred_positions = running_predicted_states[:, :, 0:3]   # (K, H, 3)
        pred_quats_wxyz = running_predicted_states[:, :, 6:10]  # (K, H, 4)

        # 计算每个时间步的目标方向向量
        # target_positions: (H, 3)
        target_positions = current_mpc_target_state_sequence_gpu[:, 0:3] # (H, 3)
        # target_direction_vectors: (K, H, 3)
        target_direction_vectors = target_positions.unsqueeze(0) - pred_positions
        # 归一化得到单位方向向量
        target_direction_vectors = F.normalize(target_direction_vectors, p=2, dim=-1)

        # 计算每个时间步无人机实际朝向
        # 无人机机体坐标系的前向轴 [1, 0, 0]
        body_forward_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=torch.float32)
        # 广播到与轨迹批次匹配的形状
        body_forward_vec_batch = body_forward_vec.view(1, 1, 3).expand(self.n_samples_cem, prediction_horizon, -1) # (K, H, 3)

        # 为了使用 pt_quat_rotate_vector，需要将 (K, H, 4) 和 (K, H, 3) 的形状拉平
        # pred_quats_flat: (K*H, 4)
        # body_forward_vec_flat: (K*H, 3)
        pred_quats_flat = pred_quats_wxyz.reshape(-1, 4)
        body_forward_vec_flat = body_forward_vec_batch.reshape(-1, 3)
        
        # 使用四元数旋转得到世界坐标系下的前向向量
        world_forward_vectors_flat = pt_quat_rotate_vector(pred_quats_flat, body_forward_vec_flat)
        # 转换回 (K, H, 3) 的形状
        world_forward_vectors = world_forward_vectors_flat.reshape(self.n_samples_cem, prediction_horizon, 3)

        # 计算对准误差 (点积的负值)，代价是 1 - 点积
        # dot_products: (K, H)
        dot_products = torch.sum(world_forward_vectors * target_direction_vectors, dim=-1)
        
        # alignment_errors: (K, H), 范围在 [0, 2] 之间
        alignment_errors = 1.0 - dot_products
        
        # 将整个时域的对准误差求和，得到每个样本的总对准代价
        # alignment_costs: (K,)
        alignment_costs = self.align_cost * torch.sum(alignment_errors, dim=1)

        # 新增速度矢量对准代价计算
        # pred_velocities: (K, H, 3)
        pred_velocities = running_predicted_states[:, :, 3:6]
        
        # 计算预测速度的模长
        pred_vel_norms = torch.norm(pred_velocities, dim=-1, keepdim=True)
        
        # 归一化预测速度向量 (避免除以零)
        pred_vel_directions = pred_velocities / (pred_vel_norms + 1e-6)
        
        # target_direction_vectors 已经在上面计算过了: (K, H, 3) 指向目标的单位向量
        
        # 计算速度方向与目标方向的点积
        vel_dot_products = torch.sum(pred_vel_directions * target_direction_vectors, dim=-1)
        
        # 速度对准误差，代价是 1 - 点积
        vel_alignment_errors = 1.0 - vel_dot_products
        
        # 只有当速度大于一定阈值时才计算方向代价，防止悬停时方向抖动导致的误差
        # 创建掩码: 速度 > 0.1 m/s
        vel_mask = (pred_vel_norms.squeeze(-1) > 0.1).float()
        
        # 应用掩码
        vel_alignment_errors = vel_alignment_errors * vel_mask
        
        # 求和得到总代价
        vel_alignment_costs = self.vel_align_cost * torch.sum(vel_alignment_errors, dim=1)

        # 总成本

        total_costs_batch = state_costs + control_costs + terminal_costs + alignment_costs + vel_alignment_costs
        # print(alignment_costs)
        return total_costs_batch

    def get_mpc_target_sequence(self, current_idx, elapsed_time):
        # 确定当前目标
        # waypoints_y: [drone_start_y, door1_y, door2_y, final_target_y]
        # door_parameters: "initial_x_pos", "amplitude", "frequency", "deviation"
        # door_z_positions: [door1_z, door2_z]
        target_sequence_np = np.zeros((self.prediction_horizon, self.state_dim))

        if current_idx < self.door_parameters["num"]: # 目标是门
            door_info_idx = current_idx # 门的索引（0或1）
            target_door_y = self.waypoints_y[current_idx + 1] # 目标门的y坐标

            for i in range(self.prediction_horizon):
                # 预测门在t + dt*i时的位置
                t_future = elapsed_time + self.dt_mpc * i
                
                # 预测门的x位置
                pred_door_x = self.door_parameters["initial_x_pos"][door_info_idx] + \
                            self.door_parameters["amplitude"] * math.sin(
                                2 * math.pi * self.door_parameters["frequency"] * t_future +
                                self.door_parameters["deviation"][door_info_idx]
                            )
                # 计算门的x速度
                pred_door_x_vel = 2 * math.pi * self.door_parameters["frequency"] * \
                                self.door_parameters["amplitude"] * math.cos(
                                    2 * math.pi * self.door_parameters["frequency"] * t_future +
                                    self.door_parameters["deviation"][door_info_idx]
                                )
                
                # MPC目标：以指定速度穿过指定点
                target_pos_x = pred_door_x
                target_pos_y = target_door_y + self.waypoint_pass_threshold_y # 对准门的y位置+阈值
                target_pos_z = self.door_z_positions[door_info_idx] - self.door_parameters["center"]
                # print(f'target:{target_pos_z}')

                target_vel_x = pred_door_x_vel # 匹配门的x速度
                target_vel_y = 4.0  # 穿越门的目标速度
                target_vel_z = 0.0

                target_sequence_np[i, :] = [ # 13维状态
                    target_pos_x, target_pos_y, target_pos_z,
                    target_vel_x, target_vel_y, target_vel_z,
                    0.707, 0.0 ,0.0 ,0.707 ,0.0 ,0.0 ,0.0 # Zero attitude and angular velocity target
                ]
        else: # 目标是最终目标
            target_sequence_np = np.tile(self.final_target_state, (self.prediction_horizon, 1))
            # print(target_sequence_np[0][1])
            
        return torch.tensor(target_sequence_np, dtype=torch.float32, device=self.device)
    
    def reset(self, current_drone_state, final_target_state, waypoints_y,\
                door_z_positions, door_param):
        self.current_true_state = current_drone_state
        self.final_target_state = final_target_state
        self.waypoints_y = waypoints_y # These are numpy arrays
        self.door_z_positions = door_z_positions
        self.door_parameters = door_param
        # 初始化解析模型类
        self.analytical_model_instance = None
        self.analytical_model_instance = SimpleFlightDynamicsTorch(
            self.n_samples_cem, dt=0.1, dtype=torch.float32       # dt是动力学模型内部积分步长
        )
        self.examiner_instance = SimpleFlightDynamicsTorch(       # 检验并行化之后模型可靠性
            1, dt=0.05, dtype=torch.float32
        )
        self.mean_control_sequence_warm_start = np.ones((self.prediction_horizon, self.action_dim))*0.64

    def step(self, current_true_state, current_idx, elapsed_time):
        # 分阶段
        self.current_idx = current_idx

        # MPC目标
        current_mpc_target_sequence_gpu = self.get_mpc_target_sequence(self.current_idx, elapsed_time) # (H, 13)

        cem_iter_mean_gpu = torch.tensor(self.mean_control_sequence_warm_start, dtype=torch.float32, device=self.device)
        cem_iter_std_gpu = torch.full((self.prediction_horizon, self.action_dim), self.initial_std_cem, dtype=torch.float32, device=self.device)
        
        # CEM优化
        
        for _ in range(self.n_iter_cem):
            
            # 采样控制序列
            perturbations_gpu = torch.normal(mean=0.0, std=1.0,
                                                size=(self.n_samples_cem, self.prediction_horizon, self.action_dim),
                                                device=self.device)
            sampled_controls_gpu = cem_iter_mean_gpu.unsqueeze(0) + \
                                                perturbations_gpu * cem_iter_std_gpu.unsqueeze(0)
            
            # 剪裁动作范围
            sampled_controls_gpu = torch.clip(sampled_controls_gpu, self.control_min, self.control_max)

            # 初始状态
            current_true_state_gpu = torch.tensor(current_true_state, dtype=torch.float32, device=self.device)
            current_true_state_gpu = current_true_state_gpu.unsqueeze(0).repeat(self.n_samples_cem, 1)

            # simulate_horizon需求动作序列维度(H, K, ActionDim=4)
            # sampled_controls_gpu的维度是(K, H, ActionDim=4)
            predicted_trajectory_batch = self.analytical_model_instance.simulate_horizon(
                    current_true_state_gpu, # 初始状态
                    sampled_controls_gpu.permute(1, 0, 2), # 转置成(H, K, 4)
                    self.dt_mpc)

            # 计算代价
            if self.current_idx ==0:
                costs_cem_gpu = self.cost_function_gpu(
                    predicted_trajectory_batch,  # (K, H+1, 12) SCALED states
                    sampled_controls_gpu,      # (K, H, ActionDim) UNSCALED controls
                    current_mpc_target_sequence_gpu,
                    self.q_state_matrix_gpu,
                    self.r_control_matrix_gpu, # Use the R matrix appropriate for current action space
                    self.q_terminal_matrix_gpu)
            
            if self.current_idx ==1:
                costs_cem_gpu = self.cost_function_gpu(
                    predicted_trajectory_batch,  # (K, H+1, 12) SCALED states
                    sampled_controls_gpu,      # (K, H, ActionDim) UNSCALED controls
                    current_mpc_target_sequence_gpu,
                    self.q_state_matrix_gpu_two,
                    self.r_control_matrix_gpu_two, # Use the R matrix appropriate for current action space
                    self.q_terminal_matrix_gpu_two)
        
            elif self.current_idx == 2:
                costs_cem_gpu = self.cost_function_gpu(
                    predicted_trajectory_batch,  # (K, H+1, 12) SCALED states
                    sampled_controls_gpu,      # (K, H, ActionDim) UNSCALED controls
                    current_mpc_target_sequence_gpu,
                    self.static_q_state_matrix_gpu,
                    self.static_r_control_matrix_gpu, # Use the R matrix appropriate for current action space
                    self.static_q_terminal_matrix_gpu)
            
            
            # 选择精英群体
            elite_indices = torch.argsort(costs_cem_gpu)[:self.n_elites_cem]
            elite_sequences_gpu = sampled_controls_gpu[elite_indices]
            
            # 更新mean和std
            new_mean_gpu = torch.mean(elite_sequences_gpu, dim=0)
            new_std_gpu = torch.std(elite_sequences_gpu, dim=0) # 有偏方差
            
            cem_iter_mean_gpu = self.alpha_cem * new_mean_gpu + (1 - self.alpha_cem) * cem_iter_mean_gpu
            cem_iter_std_gpu = self.alpha_cem * new_std_gpu + (1 - self.alpha_cem) * cem_iter_std_gpu
            cem_iter_std_gpu = torch.maximum(cem_iter_std_gpu, torch.tensor(self.min_std_cem, dtype=torch.float32, device=self.device))

        optimal_control_sequence = cem_iter_mean_gpu.cpu().numpy()
        actual_control_to_apply = optimal_control_sequence[0, :].copy()
        actual_control_to_apply = np.clip(actual_control_to_apply, self.control_min, self.control_max)
        
        # warm start
        self.mean_control_sequence_warm_start = np.roll(optimal_control_sequence, -1, axis=0)
        self.mean_control_sequence_warm_start[-1, :] = optimal_control_sequence[-1, :].copy()

        # 检验并行化模型可靠性
        # examined_trajectory_batch = self.examiner_instance.simulate_horizon(
        #             current_true_state_gpu[0,:].unsqueeze(0), # 初始状态取一个
        #             cem_iter_mean_gpu.unsqueeze(1), # 转置成(H, K, 4)
        #             self.dt_mpc)
        # print("————————————————————\nmodel prediction:", examined_trajectory_batch[0][1])

        # 更新状态
        # applied_controls_log.append(actual_control_to_apply.copy())
        # actual_trajectory_log.append(current_true_state.copy())
        # time_points_log.append(steps_taken_in_episode * dt_mpc)

        return actual_control_to_apply