import torch
import config as cfg

# 四元数乘法，假设q1_batch, q2_batch 的形状为 (K, 4)，格式为 [w, x, y, z]
def pt_quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), dim=1)

# 四元数共轭，假设q_batch 的形状为 (K, 4)，格式为 [w, x, y, z]
def pt_quat_conjugate(q):
    return torch.cat((q[:, 0:1], -q[:, 1:4]), dim=1)

# 通过四元数对三维向量进行旋转，假设q_batch 的形状为 (K, 4)，格式为 [w, x, y, z]
# v_batch 的形状为 (K, 3)
def pt_quat_rotate_vector(q, v):
    q_conj = pt_quat_conjugate(q)
    # 将v转换为纯四元数 [0, vx, vy, vz]
    v_quat = torch.cat((torch.zeros(v.shape[0], 1, device=v.device, dtype=v.dtype), v), dim=1)
    # 将四元数 q 与纯四元数 v_quat 相乘，得到一个旋转后的四元数；
    # 再将结果与四元数的共轭 q_conj 相乘，得到旋转后的四元数表示
    rotated_v = pt_quat_multiply(pt_quat_multiply(q, v_quat), q_conj)
    return rotated_v[:, 1:4] # 提取向量部分，标量部分一般为0

# 旋转矢量转四元数，假设rotvec_batch 的形状为 (K, 3)
# 返回形状为 (K, 4) 的四元数，格式为 [w, x, y, z]
def pt_rotvec_to_quat(rotvec, device, dtype=torch.float32):
    angle = torch.linalg.norm(rotvec, dim=1, keepdim=True)
    axis = torch.div(rotvec, angle, out=torch.zeros_like(rotvec)) # 转为单位向量（即转轴），同时处理零向量
    
    w = torch.cos(angle / 2.0)
    xyz = axis * torch.sin(angle / 2.0)
    
    # 构造结果，对 angle 接近0的情况特殊处理
    # 当 angle 非常小时，sin(angle/2) ~ angle/2, cos(angle/2) ~ 1
    # axis * sin(angle/2) ~ rotvec / angle * angle / 2 = rotvec / 2
    # w ~ 1
    # 我们需要一个mask来识别这些情况
    mask_angle_zero = angle < 1e-7 # 选择一个小的阈值
    
    # 正常计算
    q_calculated = torch.cat((w, xyz), dim=1)
    
    # 对 angle 为 0 的情况，使用单位四元数
    # 对于非零角度，使用计算值
    # 注意：直接用 torch.where(mask_angle_zero, identity_quat_part, q_calculated) 可能因为nan值传播导致问题
    # 因此，我们分别构造
    
    # 初始化为单位四元数
    final_q = torch.zeros_like(q_calculated)
    final_q[:,0] = 1.0

    # 对非零角度的情况，填充计算值
    non_zero_angle_mask = ~mask_angle_zero.squeeze() # 逻辑取反 mask_angle_zero
    if torch.any(non_zero_angle_mask):
        final_q[non_zero_angle_mask] = q_calculated[non_zero_angle_mask]

    return final_q

# # 将 scipy 格式 [x,y,z,w] 转换为内部 [w,x,y,z]
# def pt_convert_scipy_quat_to_wxyz(scipy_quat_batch):
#     return torch.cat((scipy_quat_batch[:, 3:4], scipy_quat_batch[:, 0:3]), dim=1)

# # 将内部 [w,x,y,z] 转换为 scipy 格式 [x,y,z,w]
# def pt_convert_wxyz_quat_to_scipy(wxyz_quat_batch):
#     return torch.cat((wxyz_quat_batch[:, 1:4], wxyz_quat_batch[:, 0:1]), dim=1)

class SimpleFlightDynamicsTorch:
    def __init__(self, num_samples, dt=0.005, dtype=torch.float32):
        """
        初始化动力学模型参数。
        Args:
            num_samples (int): K, 并行处理的样本数量。
            dt (float): 模拟时间步长。
            dtype (torch.dtype): PyTorch数据类型。
        """
        self.K = num_samples
        self.dt = dt
        self.device = cfg.device
        self.dtype = dtype

        self.mass = torch.tensor(cfg.UAV_mass, device=self.device, dtype=self.dtype)
        self.inertia_b = torch.diag(cfg.UAV_inertia_diag) # 本体系下惯量矩阵
        self.inertia_inv_b = torch.linalg.inv(self.inertia_b)

        self.L_eff = torch.tensor(
            cfg.UAV_arm_length * torch.cos(torch.tensor(torch.pi / 4.0)), 
            device=self.device, dtype=self.dtype
        )
        
        # 线性阻力盒
        self.drag_box_body = cfg.drag_box
        # 角阻力系数 (可以是标量或(3,)向量)
        # ang_drag_coeffs = torch.tensor([0.0, 0.0, 0.0], device=self.device, dtype=self.dtype)
        self.ang_drag_coeffs=0.0
        self.gravity_w = torch.tensor([0, 0, 9.81], device=self.device, dtype=self.dtype)
        # 电机参数
        self.rotor_max_thrust = torch.tensor(cfg.UAV_max_thrust, device=self.device, dtype=self.dtype)
        self.rotor_max_torque = torch.tensor(cfg.UAV_max_torque, device=self.device, dtype=self.dtype)
        self.rotor_turning_directions = torch.tensor([1.0, 1.0, -1.0, -1.0], device=self.device, dtype=self.dtype) # Shape (4,)
        # 状态维度定义 (用于内部转换和输出)
        self.state_dim = 13 # pos(3) vel(3) quat_wxyz(4) ang_vel(3)
        # 低通滤波时间系数
        self.tc = cfg.UAV_tc

    def control_filter_update(self, current_filter_outputs, new_filter_inputs):
        """
        更新滤波器状态。
        Args:
            current_filter_outputs: 当前滤波器的输出张量 (K, num_filters_per_sample)
            new_filter_inputs: 新的滤波器输入张量 (K, num_filters_per_sample)
        Returns:
            next_filter_outputs: 更新后的滤波器输出张量 (K, num_filters_per_sample)
        """
        if self.tc > 1e-9:
            alpha = 1.0 - torch.exp(torch.tensor(-self.dt / self.tc, device=self.device, dtype=self.dtype))
            next_filter_outputs = current_filter_outputs + alpha * (new_filter_inputs - current_filter_outputs)
        else:
            next_filter_outputs = new_filter_inputs
        return next_filter_outputs

    def _dynamics_step(self, current_states, current_motor_filtered_outputs, target_motor_pwms):
        """
        执行单个动力学模拟步骤。
        Args:
            current_states (K,13): 当前的解析状态批次。
                (pos_w:(K,0:3), vel_w:(K,3:6), orient_q_bw_wxyz:(K,6:10), ang_vel_b:(K,10:13))
            current_motor_filtered_outputs (torch.Tensor): (K,4) 当前电机滤波器的输出。
            motor_pwms (torch.Tensor): (K,4) 当前步骤的电机PWM目标值。
        Returns:
            next_states (K,13): 下一步的解析状态批次。
            next_motor_filtered_outputs (torch.Tensor): (K,4) 更新后的电机滤波器输出。
        """
        # 从总状态切片各项参数
        pos_w = current_states[:, 0:3]
        vel_w = current_states[:, 3:6]
        q_bw_wxyz = current_states[:, 6:10] # [w,x,y,z]
        ang_vel_b = current_states[:, 10:13]

        # 更新电机滤波器状态
        next_motor_filtered_outputs = self.control_filter_update(
            current_motor_filtered_outputs, target_motor_pwms)
        control_signal = next_motor_filtered_outputs # (K, 4)

        # 计算电机推力和扭矩 (与之前版本类似)
        thrusts = control_signal * self.rotor_max_thrust  # 推力(K, 4)
        rotor_torques_b = control_signal * self.rotor_max_torque * self.rotor_turning_directions.unsqueeze(0) # (K, 4)

        total_thrust_z_b = -torch.sum(thrusts, dim=1)  # 总推力，维度(K,)
        total_thrust_vector_b = torch.zeros((self.K, 3), device=self.device, dtype=self.dtype)
        total_thrust_vector_b[:, 2] = total_thrust_z_b  # (K, 3) # 合力

        T_FR = thrusts[:, 0]; T_RL = thrusts[:, 1]
        T_FL = thrusts[:, 2]; T_RR = thrusts[:, 3]
        tau_x_b = self.L_eff * (T_FL + T_RL - T_FR - T_RR)
        tau_y_b = self.L_eff * (T_FR + T_FL - T_RL - T_RR)
        tau_z_b = torch.sum(rotor_torques_b, dim=1)
        torques_actuators_b = torch.stack((tau_x_b, tau_y_b, tau_z_b), dim=-1) # (K, 3)

        # 计算阻力
        torque_drag_b = -self.ang_drag_coeffs * ang_vel_b
        
        q_bw_inv = pt_quat_conjugate(q_bw_wxyz) # 姿态四元数共轭=求逆
        velocity_b = pt_quat_rotate_vector(q_bw_inv, vel_w) # 速度转到本体系
        force_drag_b = -self.drag_box_body.unsqueeze(0) * torch.abs(velocity_b) * velocity_b

        # 4. 总力和总扭矩
        total_force_b = total_thrust_vector_b + force_drag_b
        total_torque_b = torques_actuators_b + torque_drag_b

        # 5. 动力学方程求解 (与之前版本类似)
        total_force_w = pt_quat_rotate_vector(q_bw_wxyz, total_force_b) # 合力转移到世界系
        linear_accel_w = total_force_w / self.mass + self.gravity_w.unsqueeze(0) # 计算加速度

        inertia_omega_b = torch.einsum('ki,ij->kj', ang_vel_b, self.inertia_b) # 角动量
        cross_product_term = torch.cross(ang_vel_b, inertia_omega_b, dim=1)
        torque_net_b = total_torque_b - cross_product_term
        angular_accel_b = torch.einsum('ki,ij->kj', torque_net_b, self.inertia_inv_b)
        
        # 积分更新状态
        next_pos_w = pos_w + (vel_w + linear_accel_w * self.dt / 2.0) * self.dt
        next_vel_w = vel_w + linear_accel_w * self.dt
        
        delta_angle_b = (ang_vel_b + angular_accel_b * self.dt / 2.0) * self.dt # 旋转向量
        next_ang_vel_b = ang_vel_b + angular_accel_b * self.dt

        delta_rotation_q = pt_rotvec_to_quat(delta_angle_b, self.device, self.dtype) # 使用您定义的辅助函数
        next_q_bw_wxyz = pt_quat_multiply(q_bw_wxyz, delta_rotation_q) # 使用您定义的辅助函数
        
        norm_q = torch.linalg.norm(next_q_bw_wxyz, dim=1, keepdim=True)
        norm_q = torch.where(norm_q == 0, torch.ones_like(norm_q), norm_q) # 避免除以零
        next_q_bw_wxyz = next_q_bw_wxyz / norm_q # 归一化

        next_states = torch.cat((next_pos_w, next_vel_w, next_q_bw_wxyz, next_ang_vel_b), dim=1)
        return next_states, next_motor_filtered_outputs

    # --- 公共方法 ---
    def simulate_horizon(self, batch_initial_states, 
                         motor_pwms_horizon, DT,
                         initial_motor_filter_outputs=None):
        """
        在GPU上批量模拟多个样本在给定控制序列下的轨迹。
        Args:
            batch_initial_states (torch.Tensor): (K, 13) 初始状态批次 (pos,vel,quat,ang_vel_body)。
            motor_pwms_horizon (torch.Tensor): (Horizon, K, 4) 电机PWM控制序列。
            initial_motor_filter_outputs (torch.Tensor, optional): (K,4) 电机滤波器的初始输出。
                                                                     如果为None, 则默认为零。
        Returns:
            predicted_trajectories (torch.Tensor): (K, Horizon + 1, 13) 预测的状态轨迹。
                                                              包含初始状态 (t=0)。
        """
        prediction_horizon = motor_pwms_horizon.shape[0]
        self.DT=torch.tensor(DT, dtype=self.dtype, device=self.device)

        # 准备初始解析状态和滤波器状态
        current_analytical_states = batch_initial_states
        
        if initial_motor_filter_outputs is None:
            current_filter_outputs = torch.zeros((self.K, 4), device=self.device, dtype=self.dtype)
        else:
            current_filter_outputs = initial_motor_filter_outputs.clone()

        # 存储轨迹 (K, Horizon + 1, 13)
        predicted_trajectories = torch.zeros(
            (self.K, prediction_horizon + 1, self.state_dim), 
            device=self.device, dtype=self.dtype
        )
        # 存储初始状态 (t=0)
        predicted_trajectories[:, 0, :] = batch_initial_states

        # 循环模拟
        for t in range(prediction_horizon):
            target_motor_pwms = motor_pwms_horizon[t, :, :] # 需要的电机转速
            for k in range(int( self.DT / self.dt )):
                # 对DT内离散化动力学
                next_analytical_states, next_filter_outputs = self._dynamics_step(
                    current_analytical_states, 
                    current_filter_outputs, 
                    target_motor_pwms
                )
                # 更新状态以便下一次迭代
                current_analytical_states = next_analytical_states
                current_filter_outputs = next_filter_outputs
            
            # 将13D解析状态存储
            predicted_trajectories[:, t + 1, :] = next_analytical_states
            
        return predicted_trajectories
