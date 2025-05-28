import numpy as np
from scipy.spatial.transform import Rotation as R

class FirstOrderFilter: # 输入信号低通滤波器，对信号平滑化处理
    def __init__(self, time_constant, initial_output=0.0):
        self.time_constant = time_constant
        self.output = initial_output
        self.input = initial_output

    def set_input(self, new_input):
        self.input = new_input

    def get_output(self):
        return self.output

    def get_input(self): # To match RotorActuator logic
        return self.input

    def update(self, dt):
        if self.time_constant > 1e-9: # 避免除以0
            # alpha = dt / (self.time_constant + dt) # 欧拉后向法离散，见filter.md
            alpha = 1 - np.exp (- dt / self.time_constant)
            self.output = self.output + alpha * (self.input - self.output)
        else:
            self.output = self.input # No filtering if time constant is zero
        return self.output

    def reset(self, initial_output=0.0):
        self.output = initial_output
        self.input = initial_output

class SimpleFlightDynamics:
    def __init__(self, initial_position, initial_velocity, initial_orientation_quat, initial_angular_velocity,
                dt=0.005):
        """
        脚标：
        initial_position (np.ndarray): Shape (3,), [N, E, D] 世界坐标系下位置(NED).
        initial_velocity (np.ndarray): Shape (3,), [vN, vE, vD] 世界系下速度 (NED).
        initial_orientation_quat (np.ndarray): Shape (4,), [w, x, y, z] 本体系(FRD)到世界系(NED)的四元数姿态角.
        initial_angular_velocity (np.ndarray): Shape (3,), [p, q, r] 本体坐标系下角速度 (FRD - Roll, Pitch, Yaw rates).
            - AirSim官方文档说角速度是FLU系，但是代码实现和实际表现都是FRD系
            - 不过AirSim读取的应该就是本体系下角速度
        vehicle_params (dict, optional): 无人机参数字典，默认为Generic QuadX.
        dt (float): 仿真步长.
        """
        self.dt = dt
        self.gravity_w = np.array([0, 0, 9.81])  # NED世界坐标系

        # 状态变量
        self.position_w = np.array(initial_position, dtype=float) 
        self.velocity_w = np.array(initial_velocity, dtype=float)
        self.orientation_q_bw = R.from_quat(initial_orientation_quat) # 创建一个从本体系转到世界系的“旋转对象”;R.from_quat需要[x,y,z,w]格式的四元数，airsim输出的正是
        self.angular_velocity_b = np.array(initial_angular_velocity, dtype=float)

        self.params = self._get_quadx_params()

        if 'max_thrust' not in self.params['rotor_params'] or 'max_torque' not in self.params['rotor_params']:
            self._calculate_max_thrust_torque() # 如果字典里没有最大力和力矩就重新计算

        self.mass = self.params['mass']
        self.inertia_b = np.diag(self.params['inertia_diag']) # 转动惯量对角阵
        self.inertia_inv_b = np.linalg.inv(self.inertia_b) # 转动惯量求逆
        self.linear_drag_coeff = self.params['linear_drag_coefficient']
        self.drag_box = 0.5 * self.linear_drag_coeff * np.array([self.params['yz_area'], self.params['xz_area'], self.params['xy_area']])
        self.angular_drag_coeff_b = self.params['angular_drag_coefficient']

        # 电机X型配置, FRD本体系: X向前, Y向右, Z向下)
        # 输入PWM指令顺序: (前右, 后左, 前左, 后右)
        # 电机编号： 0: 前右,, 1: 后左, 2: 前左, 3: 后右
        L_eff = self.params['arm_length'] * np.cos(np.pi / 4) # 投影长度

        z_offset_val_cg_relative = self.params.get('rotor_z_offset', 0.0) # 电机高度

        self.rotor_positions_b = [ # 电机位置数组
            np.array([ L_eff,  L_eff, -z_offset_val_cg_relative]), # 0: 前右
            np.array([-L_eff, -L_eff, -z_offset_val_cg_relative]), # 1: 后左
            np.array([ L_eff, -L_eff, -z_offset_val_cg_relative]), # 2: 前左
            np.array([-L_eff,  L_eff, -z_offset_val_cg_relative])  # 3: 后右
        ]

        # 从上方看的电机旋转方向:
        # 0: CCW (逆时针) -> 对机体产生反力矩: CW (顺时针) -> 正偏航力矩 (+1)
        # 1: CCW -> 对机体产生反力矩: CW -> 正偏航力矩 (+1)
        # 2: CW  -> 对机体产生反力矩： CCW -> 负偏航力矩 (-1)
        # 3: CW  -> 对机体产生反力矩： CCW -> 负偏航力矩 (-1)
        self.rotor_turning_directions = np.array([1, 1, -1, -1])

        self.motor_filters = [FirstOrderFilter(self.params['rotor_params']['control_signal_filter_tc']) for _ in range(4)] # 为每个发动机分别配置一个滤波器

    def _get_quadx_params(self):
        params = {}
        params['mass'] = 1.0 # 无人机总重量
        params['arm_length'] = 0.2275 # 无人机臂长度
        params['rotor_z_offset'] = 0.025 # 电机高度

        body_mass_fraction = 0.78 # 无人机中心盒重量占比
        body_cg_mass = params['mass'] * body_mass_fraction
        motor_assembly_mass_total = params['mass'] * (1-body_mass_fraction)
        motor_mass = motor_assembly_mass_total / 4.0 # 电机质量
 
        dim_x = 0.180; dim_y = 0.110; dim_z = 0.040 # 机身盒尺寸

        Ixx_body = body_cg_mass / 12.0 * (dim_y**2 + dim_z**2) # 机身对三个轴的转动惯量
        Iyy_body = body_cg_mass / 12.0 * (dim_x**2 + dim_z**2)
        Izz_body = body_cg_mass / 12.0 * (dim_x**2 + dim_y**2)

        L_eff_sq = (params['arm_length'] * np.cos(np.pi/4))**2 # 电机位置偏移量的平方
        rotor_z_dist_sq = params['rotor_z_offset']**2 # 电机高度偏移量

        Ixx_motors = 4 * motor_mass * (L_eff_sq + rotor_z_dist_sq)
        Iyy_motors = 4 * motor_mass * (L_eff_sq + rotor_z_dist_sq)
        Izz_motors = 4 * motor_mass * (2 * L_eff_sq)

        params['inertia_diag'] = np.array([ # 转动惯量矩阵
            Ixx_body + Ixx_motors,
            Iyy_body + Iyy_motors,
            Izz_body + Izz_motors
        ])

        params['linear_drag_coefficient'] = 0.325 # 线阻力系数
        params['angular_drag_coefficient'] = 0.0 # 角阻力系数；0.325的时候每个DT都会导致最后变成推力矩与阻力矩平衡，无人机y方向角速度锁定在0.02

        params['rotor_params'] = { # 电机参数与空气密度
            'C_T': 0.109919, 'C_P': 0.040164, 'air_density': 1.225,
            'max_rpm': 6396.667, 'propeller_diameter': 0.2286,
            'propeller_height':0.01,
            'control_signal_filter_tc': 0.005,
            'max_thrust':4.179446268,
            'max_torque':0.055562
        }
        # 补全电机参数
        params['rotor_params']['revolutions_per_second'] = params['rotor_params']['max_rpm'] / 60.0
        params['rotor_params']['max_speed_rad_s'] = params['rotor_params']['revolutions_per_second'] * 2 * np.pi
        params['rotor_params']['max_speed_sq'] = params['rotor_params']['max_speed_rad_s']**2
        params['xy_area'] = dim_x * dim_y + 4.0 * np.pi * params['rotor_params']['propeller_diameter']**2
        params['yz_area'] = dim_y * dim_z + 4.0 * np.pi * params['rotor_params']['propeller_diameter'] * params['rotor_params']['propeller_height']
        params['xz_area'] = dim_x * dim_z + 4.0 * np.pi * params['rotor_params']['propeller_diameter'] * params['rotor_params']['propeller_height']
        
        return params

    def _calculate_max_thrust_torque(self): # 计算电机最大推力、最大力矩
        rp = self.params['rotor_params']
        n_rps = rp['revolutions_per_second']
        D = rp['propeller_diameter']; rho = rp['air_density']
        rp['max_thrust'] = rp['C_T'] * rho * (n_rps**2) * (D**4)
        rp['max_torque'] = rp['C_P'] * rho * (n_rps**2) * (D**5) / (2 * np.pi)

    def simulate_step(self, motor_pwms, ):
        rp = self.params['rotor_params'] # 电机参数
        air_density_ratio = 1.0 # 基本不用考虑空气密度变化
        thrusts_mag = np.zeros(4) # 推力大小
        rotor_yaw_torques_b = np.zeros(4) # 机体坐标系下电机产生的偏航力矩大小

        motor_input_map = [motor_pwms[0], motor_pwms[1], motor_pwms[2], motor_pwms[3]]

        for i in range(4):
            self.motor_filters[i].set_input(motor_input_map[i])
            self.motor_filters[i].update(self.dt)
            cs_f = self.motor_filters[i].get_output() # 滤波后的推力幅度
            thrusts_mag[i] = cs_f * rp['max_thrust'] * air_density_ratio
            rotor_yaw_torques_b[i] = cs_f * rp['max_torque'] * self.rotor_turning_directions[i] * air_density_ratio # 考虑方向的电机偏航力矩数组

        T_FR, T_RL, T_FL, T_RR = thrusts_mag[0], thrusts_mag[1], thrusts_mag[2], thrusts_mag[3]

        total_thrust_vector_b = np.array([0, 0, -np.sum(thrusts_mag)]) # 总推力在NED系下
        L_eff = self.params['arm_length'] * np.cos(np.pi / 4)

        tau_x_b = L_eff * (T_FR + T_RR - T_FL - T_RL ) # 推力对三轴产生的力矩
        tau_y_b = L_eff * (T_FR + T_FL - T_RL - T_RR)
        tau_z_b = np.sum(rotor_yaw_torques_b)
        torques_actuators_b = np.array([tau_x_b, tau_y_b, tau_z_b]) # 电机在本体系下产生力矩

        # 线性阻力，速度高的时候误差严重，换成drag_box模式；角速度反正系数已经为0了爱咋咋
        # force_drag_w = -self.linear_drag_coeff * self.velocity_w * np.abs(self.velocity_w) # 世界系下简化空气阻力
        # force_drag_b = self.orientation_q_bw.inv().apply(force_drag_w) # 姿态转换求逆就是w→b，空气阻力转到本体系
        torque_drag_b = -self.angular_drag_coeff_b * self.angular_velocity_b # 旋转阻力
        
        # drag_box方式计算阻力
        velocity_b = self.orientation_q_bw.inv().apply(self.velocity_w)
        force_drag_b = - self.drag_box * np.abs(velocity_b) * velocity_b

        total_force_b = total_thrust_vector_b + force_drag_b # 本体系下合力
        total_torque_b = torques_actuators_b + torque_drag_b # 本体系下合力矩
        # print("torque_drag_b", torque_drag_b)
        # print("total_torque_b", total_torque_b)

        total_force_w = self.orientation_q_bw.apply(total_force_b) # 合力转到世界系
        linear_accel_w = total_force_w / self.mass + self.gravity_w # 世界系下加速度

        inertia_omega_b = self.inertia_b @ self.angular_velocity_b # 本体系下角动量
        # print("inertia_omega_b", inertia_omega_b)
        cross_product_term = np.cross(self.angular_velocity_b, inertia_omega_b) # 角速度叉乘角动量
        angular_accel_b = self.inertia_inv_b @ (total_torque_b - cross_product_term) # 本体系角加速度
        # print("angular_accel_b", angular_accel_b)
        
        self.position_w += (self.velocity_w + linear_accel_w * self.dt/2) * self.dt # 中点法计算位置
        self.velocity_w += linear_accel_w * self.dt # 前向欧拉积分，没用到verlet
        delta_angle_b = (self.angular_velocity_b + angular_accel_b * self.dt/2) * self.dt # 中点法计算角度增量
        self.angular_velocity_b += angular_accel_b * self.dt
        # delta_angle_b不是普通向量，是一个旋转向量；
        # 旋转向量的模数表示旋转角度大小，方向表示旋转轴
        angle_norm = np.linalg.norm(delta_angle_b) 
        if angle_norm > 1e-9:
            axis = delta_angle_b / angle_norm # 计算delta_angle的转轴方向向量
            # from_rotvec() 方法接受一个旋转向量，并将其转换为一个 Rotation 对象（内部以四元数表示）
            # 实际就是把delta_angle转成四元数
            delta_rotation_q = R.from_rotvec(axis * angle_norm)
            self.orientation_q_bw = self.orientation_q_bw * delta_rotation_q # 四元数右乘表示在原来的本体系下转delta_rotation_q

    def simulate_duration(self, motor_pwms_tuple, duration, vehicle_name=''):
        motor_pwms = np.array(motor_pwms_tuple, dtype=float)
        num_steps = int(round(duration / self.dt)); num_steps = max(1, num_steps)
        for _ in range(num_steps): self.simulate_step(motor_pwms)
        return self.get_current_state()

    def get_current_state(self):
        # ypr_rad = self.orientation_q_bw.as_euler('zyx', degrees=False) # zyx的顺序转化为欧拉角
        # euler_angles_deg = np.rad2deg([ypr_rad[2], ypr_rad[1], ypr_rad[0]])
        return self.position_w, self.velocity_w, self.orientation_q_bw.as_quat(), self.angular_velocity_b

# --- Example Usage ---
# if __name__ == '__main__':
#     initial_pos_ned = np.array([0.0, 0.0, 0.0])
#     initial_vel_ned = np.array([0.0, 0.0, 0.0])
#     initial_orient_quat_frd_to_ned = np.array([0.0, 0.0, 0.0, 1.0])
#     initial_ang_vel_frd = np.array([0.0, 0.0, 0.0])

#     drone = SimpleFlightDynamics(initial_pos_ned, initial_vel_ned, initial_orient_quat_frd_to_ned, initial_ang_vel_frd)
#     print("Initial State (NED world, FRD body):"); print(drone.get_current_state())

#     # Max thrust per motor from default params ~4.179 N
#     # Hover PWM ~0.587
#     hover_pwm = 0.587
#     pwm_offset = 0.1 # For maneuvers

#     # --- Takeoff Example ---
#     print(f"\nSimulating TAKEOFF...")
#     takeoff_pwm = 0.65
#     final_state_takeoff = drone.simulate_duration((takeoff_pwm, takeoff_pwm, takeoff_pwm, takeoff_pwm), 2.0)
#     print("\nFinal State after takeoff simulation:"); print(final_state_takeoff)


#     # --- Roll Right Example ---
#     # tau_x_b = L_eff * (T_FL + T_RL - T_FR - T_RR) -> For positive roll (right), FL,RL > FR,RR
#     # Motor order: (FR, RL, FL, RR) -> (low, high, high, low)
#     drone = SimpleFlightDynamics(initial_pos_ned, initial_vel_ned, initial_orient_quat_frd_to_ned, initial_ang_vel_frd) # Reset
#     print("\nSimulating a ROLL RIGHT command...")
#     roll_commands = (hover_pwm - pwm_offset, hover_pwm + pwm_offset, hover_pwm + pwm_offset, hover_pwm - pwm_offset)
#     roll_commands = tuple(np.clip(c, 0.0, 1.0) for c in roll_commands)
#     final_state_roll = drone.simulate_duration(roll_commands, 0.5)
#     print("\nFinal State after roll command:"); print(final_state_roll)


#     # --- Pitch Up Example ---
#     # tau_y_b = L_eff * (T_FR + T_FL - T_RL - T_RR) -> For positive pitch (up), FR,FL > RL,RR
#     # Motor order: (FR, RL, FL, RR) -> (high, low, high, low)
#     drone = SimpleFlightDynamics(initial_pos_ned, initial_vel_ned, initial_orient_quat_frd_to_ned, initial_ang_vel_frd) # Reset
#     print("\nSimulating a PITCH UP command...")
#     pitch_commands = (hover_pwm + pwm_offset, hover_pwm - pwm_offset, hover_pwm + pwm_offset, hover_pwm - pwm_offset)
#     pitch_commands = tuple(np.clip(c, 0.0, 1.0) for c in pitch_commands)
#     final_state_pitch = drone.simulate_duration(pitch_commands, 0.5)
#     print("\nFinal State after pitch up command:"); print(final_state_pitch)


#     # --- Yaw Right Example (Corrected for new motor directions) ---
#     # rotor_turning_directions = [1, 1, -1, -1] for (FR, RL, FL, RR)
#     # tau_z_b = Torque_FR(cs_FR*1) + Torque_RL(cs_RL*1) + Torque_FL(cs_FL*-1) + Torque_RR(cs_RR*-1)
#     # For positive yaw (right): Increase cs_FR, cs_RL; Decrease cs_FL, cs_RR
#     # Motor order: (FR, RL, FL, RR) -> (high, high, low, low)
#     drone = SimpleFlightDynamics(initial_pos_ned, initial_vel_ned, initial_orient_quat_frd_to_ned, initial_ang_vel_frd) # Reset
#     print("\nSimulating a YAW RIGHT command...")
#     yaw_commands = (hover_pwm + pwm_offset, hover_pwm + pwm_offset, hover_pwm - pwm_offset, hover_pwm - pwm_offset)
#     yaw_commands = tuple(np.clip(c, 0.0, 1.0) for c in yaw_commands)
#     final_state_yaw = drone.simulate_duration(yaw_commands, 0.5)
#     print("\nFinal State after yaw right command:"); print(final_state_yaw)
