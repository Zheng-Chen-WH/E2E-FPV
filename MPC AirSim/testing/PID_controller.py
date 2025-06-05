import torch
import math
 
class SimpleFlightParams:
    def __init__(self, device='cpu'):
        self.device = device

        # 电机参数
        self.motor_count = 4
        self.min_motor_output = 0.0
        self.max_motor_output = 1.0
        self.min_armed_throttle_val = 0.1 # 解锁情况下最小油门推力
        self.min_angling_throttle_motor = self.min_armed_throttle_val / 2.0 # 避免低油门下因姿态调整导致推力不足而坠机

        # 角速度AngleRatePID控制器参数
        _arp_kMaxLimit = 2.5 # 最大角速度限制
        _arp_kP = 0.25 # 比例增益
        _arp_kI = 0.0 # 积分增益
        _arp_kD = 0.0 # 微分增益
        self.angle_rate_pid_max_limit = torch.tensor([_arp_kMaxLimit, _arp_kMaxLimit, _arp_kMaxLimit], dtype=torch.float32, device=self.device) # roll, pitch, yaw
        
        # 存储PID增益值，前三个值分别对应滚转、俯仰、偏航轴，第四个为1.0或0.0（默认值），可以在这里修改以实现三轴使用不同pid增益
        self.angle_rate_pid_p = torch.tensor([_arp_kP, _arp_kP, _arp_kP, 1.0], dtype=torch.float32, device=self.device)
        self.angle_rate_pid_i = torch.tensor([_arp_kI, _arp_kI, _arp_kI, 0.0], dtype=torch.float32, device=self.device)
        self.angle_rate_pid_d = torch.tensor([_arp_kD, _arp_kD, _arp_kD, 0.0], dtype=torch.float32, device=self.device)
        # Assuming AngleRateController output limits are important for PID output directly
        self.angle_rate_pid_output_min = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32, device=self.device) # Placeholder, adjust as needed
        self.angle_rate_pid_output_max = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=self.device)  # Placeholder, adjust as needed


        # 角度AngleLevelPID控制器参数
        _alp_pi = 3.14159265359
        _alp_kP = 2.5
        _alp_kI = 0.0
        _alp_kD = 0.0
        self.angle_level_pid_max_limit = torch.tensor([_alp_pi/5.5, _alp_pi/5.5, _alp_pi, 1.0], dtype=torch.float32, device=self.device) # 角度限制
        self.angle_level_pid_p = torch.tensor([_alp_kP, _alp_kP, _alp_kP, 1.0], dtype=torch.float32, device=self.device)
        self.angle_level_pid_p = torch.tensor([_alp_kP, _alp_kP, _alp_kP, 0.0], dtype=torch.float32, device=self.device)
        self.angle_level_pid_p = torch.tensor([_alp_kP, _alp_kP, _alp_kP, 0.0], dtype=torch.float32, device=self.device)

        # 速度VelocityPID控制器参数
        _vp_kMaxLimit = 6.0 # 最大速度限制m/s
        _vp_kMinThrottle = self.min_armed_throttle_val * 3.0
        _vp_kP = 0.2
        _vp_kI = 2.0
        _vp_kD = 0.0
        self.velocity_pid_max_limit = torch.tensor([_vp_kMaxLimit, _vp_kMaxLimit, 0, _vp_kMaxLimit], dtype=torch.float32, device=self.device) # x, y, yaw, z, 米表示，偏航轴为0，不由速度控制
        self.velocity_pid_p = torch.tensor([_vp_kP, _vp_kP, 0.0, 2.0], dtype=torch.float32, device=self.device) # 同样是x,y,yaw,z
        self.velocity_pid_i = torch.tensor([0.0, 0.0, 0.0, _vp_kI], dtype=torch.float32, device=self.device)
        self.velocity_pid_d = torch.tensor([_vp_kD, _vp_kD, _vp_kD, _vp_kI], dtype=torch.float32, device=self.device)

        # 位置PositionPID控制器参数
        _pp_kMaxLimit = 8.8e26 # 最大位置限制，m
        _pp_kP = 0.25
        _pp_kI = 0.0
        _pp_kD = 0.0
        self.position_pid_max_limit = torch.tensor([_pp_kMaxLimit, _pp_kMaxLimit, _pp_kMaxLimit, 1.0], dtype=torch.float32, device=self.device) # x, y, yaw, z, 米表示，偏航轴为0，不由速度控制
        self.position_pid_p = torch.tensor([_pp_kP, _pp_kP, 0.0, _pp_kP], dtype=torch.float32, device=self.device)
        self.position_pid_i = torch.tensor([_pp_kI, _pp_kI, _pp_kI, _pp_kI], dtype=torch.float32, device=self.device)
        self.position_pid_d = torch.tensor([_pp_kD, _pp_kD, _pp_kD, _pp_kD], dtype=torch.float32, device=self.device)

        # QuadXMixer矩阵，用于混合油门指令和roll, pitch, yaw速率导致的扭矩并分配给各电机
        # 电机顺序: FRONT_R, REAR_L, FRONT_L, REAR_R
        self.mixer_quad_x = torch.tensor([
            # 推力  滚转  俯仰   偏航
            [1.0, -1.0,  1.0,  1.0],  # FR，0号电机
            [1.0,  1.0, -1.0,  1.0],  # RL，1号电机
            [1.0,  1.0,  1.0, -1.0],  # FL，2号电机
            [1.0, -1.0, -1.0, -1.0]   # RR，3号电机
        ], dtype=torch.float32, device=self.device)

# PID 控制器 (来自 PidController.hpp)
class PidController:
    def __init__(self, kp, ki, kd, output_min, output_max, device='cpu'):
        self.kp = kp.to(device)
        self.ki = ki.to(device)
        self.kd = kd.to(device)
        self.output_min = output_min.to(device)
        self.output_max = output_max.to(device)
        self.device = device

        self.integral = torch.zeros_like(self.ki, device=self.device)
        self.last_error = torch.zeros_like(self.kp, device=self.device)
        self.last_output = torch.zeros_like(self.kp, device=self.device) # For first derivative step

    def reset(self):
        self.integral = torch.zeros_like(self.ki, device=self.device)
        self.last_error = torch.zeros_like(self.kp, device=self.device)
        self.last_output = torch.zeros_like(self.kp, device=self.device)

    def update(self, goal, measured, dt):
        if dt <= 1e-6: # 避免除以0
            return self.last_output

        error = goal - measured
        
        # 比例项
        p_term = self.kp * error

        # 积分项
        self.integral += error * dt
        i_term = self.ki * self.integral

        # 微分项
        # SimpleFlight的PID微分项是(error - last_error_) / dt
        error_derivative = (error - self.last_error) / dt
        d_term = self.kd * error_derivative
        
        output = p_term + i_term + d_term
        output = torch.clamp(output, self.output_min, self.output_max)

        self.last_error = error
        self.last_output = output
        return output

# 轴控制器 (来自 AngleRateController.hpp 和 PassthroughController)
class AngleRateController:
    def __init__(self, params: SimpleFlightParams, axis_idx: int): # 轴参数: 0=roll, 1=pitch, 2=yaw
        # 使用了类型注解（Type Hints）：
        # params: SimpleFlightParams 表示 params 参数应该是 SimpleFlightParams 类型的对象。在 Python 中，类不仅可以作为对象的蓝图，还可以用作类型提示。
        # axis_idx: int 表示 axis_idx 参数应该是一个整数（int）
        self.params = params
        self.axis_idx = axis_idx
        self.device = params.device

        # 为目前计算的轴指明三个增益系数
        kp = params.angle_rate_pid_p[axis_idx]
        ki = params.angle_rate_pid_i[axis_idx]
        kd = params.angle_rate_pid_d[axis_idx]
        
        # 目前轴PID输出的限制
        output_min = params.angle_rate_pid_output_min[axis_idx]
        output_max = params.angle_rate_pid_output_max[axis_idx]

        self.pid = PidController(kp, ki, kd, output_min, output_max, device=self.device)
        self.output = torch.tensor(0.0, device=self.device)

    def reset(self):
        self.pid.reset()
        self.output = torch.tensor(0.0, device=self.device)

    def update(self, goal_angular_rate, measured_angular_rate, dt):
        # goal_angular_rate and measured_angular_rate are scalars for this axis
        self.output = self.pid.update(goal_angular_rate, measured_angular_rate, dt)

    def get_output(self):
        return self.output

class PassthroughController:
    def __init__(self, device='cpu'):
        self.output = torch.tensor(0.0, device=device)
        self.device = device

    def reset(self):
        self.output = torch.tensor(0.0, device=self.device)

    def update(self, goal_value, measured_value, dt): # measured_value, dt not used
        self.output = goal_value

    def get_output(self):
        return self.output

# 级联控制器 (简化版，只处理 AngleRate 和 Passthrough)

class SimplifiedCascadeController:
    def __init__(self, params: SimpleFlightParams):
        self.params = params
        self.device = params.device
        self.axis_controllers = [None] * 4 # Roll, Pitch, Yaw, Throttle

        # Goal modes for (roll_rate, pitch_rate, yaw_rate, throttle)
        # 0: AngleRate, 1: Passthrough (simplified enum)
        self.current_goal_modes = [0, 0, 0, 1] # Default: RPY=AngleRate, Thr=Passthrough

        self._setup_axis_controllers()
        self.output_controls = torch.zeros(4, device=self.device) # R, P, Y, Thr

    def _setup_axis_controllers(self):
        for axis in range(3): # Roll, Pitch, Yaw
            if self.current_goal_modes[axis] == 0: # AngleRate
                self.axis_controllers[axis] = AngleRateController(self.params, axis)
        
        if self.current_goal_modes[3] == 1: # Passthrough for Throttle
             self.axis_controllers[3] = PassthroughController(device=self.device)

    def reset(self):
        for controller in self.axis_controllers:
            if controller:
                controller.reset()
        self.output_controls = torch.zeros(4, device=self.device)

    def update(self, goal_values: torch.Tensor, measured_angular_rates: torch.Tensor, dt: float):
        # goal_values: tensor of [target_roll_rate, target_pitch_rate, target_yaw_rate, target_throttle]
        # measured_angular_rates: tensor of [current_roll_rate, current_pitch_rate, current_yaw_rate]
        
        # Roll Controller
        if self.axis_controllers[0]:
            self.axis_controllers[0].update(goal_values[0], measured_angular_rates[0], dt)
            self.output_controls[0] = self.axis_controllers[0].get_output() # Output for Roll
        
        # Pitch Controller
        if self.axis_controllers[1]:
            self.axis_controllers[1].update(goal_values[1], measured_angular_rates[1], dt)
            self.output_controls[1] = self.axis_controllers[1].get_output() # Output for Pitch

        # Yaw Controller
        if self.axis_controllers[2]:
            self.axis_controllers[2].update(goal_values[2], measured_angular_rates[2], dt)
            self.output_controls[2] = self.axis_controllers[2].get_output() # Output for Yaw

        # Throttle Controller (Passthrough)
        if self.axis_controllers[3]:
            self.axis_controllers[3].update(goal_values[3], torch.tensor(0.0, device=self.device), dt) # measured not used for passthrough
            self.output_controls[3] = self.axis_controllers[3].get_output() # Output for Throttle
            
    def get_output(self):
        # Returns a tensor [roll_command, pitch_command, yaw_command, throttle_command]
        # where roll/pitch/yaw commands are typically torque/rate outputs from PID
        # and throttle_command is the direct throttle value
        return self.output_controls


# 混控器 (来自 Mixer.hpp)
class Mixer:
    def __init__(self, params: SimpleFlightParams):
        self.params = params
        self.device = params.device

    def get_motor_output(self, controls: torch.Tensor):
        # controls: tensor of [roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd]
        # throttle_cmd is base throttle
        # roll/pitch/yaw_cmd are outputs from PID (e.g. desired torque/acceleration)

        throttle = controls[3]
        motor_outputs = torch.zeros(self.params.motor_count, device=self.device)

        if throttle < self.params.min_angling_throttle_motor:
            motor_outputs.fill_(throttle) # Assign throttle value to all motors
            return torch.clamp(motor_outputs, self.params.min_motor_output, self.params.max_motor_output)

        # Mixer logic: motor_out = base_throttle + roll_adj + pitch_adj + yaw_adj
        # Base throttle part (from controls[3]) is applied via the 'throttle' coefficient in mixer_quad_x
        # The other terms (roll, pitch, yaw from controls[0,1,2]) are adjustments.
        
        # controls tensor: [roll_contrib, pitch_contrib, yaw_contrib, base_throttle]
        # mixer_quad_x columns: [throttle_factor, roll_factor, pitch_factor, yaw_factor]
        
        # Re-arrange controls to match mixer_quad_x column interpretation if necessary.
        # SimpleFlight's mixer formula is:
        # motor[i] = controls.throttle * mix[i].throttle + controls.pitch * mix[i].pitch + ...
        # Here, controls[3] is throttle, controls[0] is roll, controls[1] is pitch, controls[2] is yaw
        
        # motor_outputs = throttle * mixer_coeffs_throttle +
        #                 roll_cmd * mixer_coeffs_roll +
        #                 pitch_cmd * mixer_coeffs_pitch +
        #                 yaw_cmd * mixer_coeffs_yaw
        
        # Let's define controls to be [throttle, roll_cmd, pitch_cmd, yaw_cmd] formatmul
        # or apply it element-wise as in C++
        
        for i in range(self.params.motor_count):
            motor_outputs[i] = (controls[3] * self.params.mixer_quad_x[i, 0] +  # Throttle component
                                controls[0] * self.params.mixer_quad_x[i, 1] +  # Roll component
                                controls[1] * self.params.mixer_quad_x[i, 2] +  # Pitch component
                                controls[2] * self.params.mixer_quad_x[i, 3])   # Yaw component

        # Normalize/Clip motor outputs (as in C++ Mixer)
        min_motor_val = torch.min(motor_outputs)
        if min_motor_val < self.params.min_motor_output:
            undershoot = self.params.min_motor_output - min_motor_val
            motor_outputs += undershoot

        max_motor_val = torch.max(motor_outputs)
        # The C++ code has: if (scale > params_->motor.max_motor_output)
        # This seems like a typo and should be: if (max_motor_val > params_->motor.max_motor_output)
        if max_motor_val > self.params.max_motor_output:
            # The C++ scale logic: scale = max_motor / params_->motor.max_motor_output;
            # And then if scale > max_motor_output (again, typo, should be scale > 1.0 or max_motor_val > max_motor_output)
            # motor_outputs[motor_index] /= scale;
            # This implies if max_motor_val exceeds max_motor_output, all motor outputs are scaled down.
            if max_motor_val > 0: # Avoid division by zero if all outputs somehow became zero
                scale_factor = self.params.max_motor_output / max_motor_val
                motor_outputs *= scale_factor


        # Final clamp
        motor_outputs = torch.clamp(motor_outputs, self.params.min_motor_output, self.params.max_motor_output)
        
        return motor_outputs

# 模拟单步控制 (简化版 Firmware::update() 逻辑)
def run_simpleflight_control_step(
    target_roll_rate: float,
    target_pitch_rate: float,
    target_yaw_rate: float,
    target_throttle: float,
    current_roll_rate: float,
    current_pitch_rate: float,
    current_yaw_rate: float,
    dt: float,
    params: SimpleFlightParams,
    cascade_controller: SimplifiedCascadeController,
    mixer: Mixer,
    device: str ='cpu'
):
    # 1. 准备输入张量
    goal_values = torch.tensor([
        target_roll_rate, target_pitch_rate, target_yaw_rate, target_throttle
    ], dtype=torch.float32, device=device)
    
    measured_angular_rates = torch.tensor([
        current_roll_rate, current_pitch_rate, current_yaw_rate
    ], dtype=torch.float32, device=device)

    # 2. 更新级联控制器 (它内部会更新各个轴的 PID)
    # OffboardAPI的setGoalAndMode在这里被简化为直接将goal_values传入
    cascade_controller.update(goal_values, measured_angular_rates, dt)

    # 3. 获取控制器输出 (抽象的滚转、俯仰、偏航指令和油门)
    controller_outputs = cascade_controller.get_output()
    # controller_outputs is [roll_pid_out, pitch_pid_out, yaw_pid_out, throttle_passthrough]

    # 4. 通过混控器计算电机输出
    # The C++ mixer_.getMotorOutput(output_controls, motor_outputs_);
    # output_controls from controller IS [roll_cmd, pitch_cmd, yaw_cmd, throttle_cmd]
    # where roll_cmd, pitch_cmd, yaw_cmd are the PID outputs for angular rates
    # and throttle_cmd is the passthrough throttle.
    # The mixer formula is:
    # motor_outputs[i] = controls.throttle() * mixer_coeffs.throttle +  <- This is base throttle
    #                    controls.pitch()   * mixer_coeffs.pitch   +  <- This is pitch adjustment
    #                    controls.roll()    * mixer_coeffs.roll    +  <- This is roll adjustment
    #                    controls.yaw()     * mixer_coeffs.yaw       <- This is yaw adjustment
    # So, the 'controls' argument to mixer should be:
    # [roll_pid_out, pitch_pid_out, yaw_pid_out, throttle_passthrough]
    # which is exactly what controller_outputs is.

    motor_signals = mixer.get_motor_output(controller_outputs)

    return motor_signals, controller_outputs

# -----------------------------------------------------------------------------
# 示例用法
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    selected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #selected_device = 'cpu' # Force CPU for testing
    print(f"Using device: {selected_device}")

    # 初始化参数和控制器
    sf_params = SimpleFlightParams(device=selected_device)
    
    # 在C++中，Firmware会根据params->controller_type创建CascadeController或AdaptiveController
    # 这里我们直接创建简化的级联控制器
    sfc_controller = SimplifiedCascadeController(params=sf_params)
    sf_mixer = Mixer(params=sf_params)

    # 模拟输入
    # 假设我们想让无人机：
    # - 以 0.5 rad/s 的速度滚转
    # - 保持俯仰和偏航速率为 0
    # - 油门设置为 0.6 (60%)
    target_rr = 0.5  # rad/s
    target_pr = 0.0  # rad/s
    target_yr = 0.0  # rad/s
    target_thr = 0.6 # 0.0 to 1.0

    # 假设无人机当前状态
    current_rr = 0.1 # rad/s
    current_pr = 0.05# rad/s
    current_yr = -0.02# rad/s
    
    # 时间步长
    simulation_dt = 0.01 # seconds (与PID的dt对应)

    print(f"Target: RollRate={target_rr:.2f}, PitchRate={target_pr:.2f}, YawRate={target_yr:.2f}, Throttle={target_thr:.2f}")
    print(f"Current: RollRate={current_rr:.2f}, PitchRate={current_pr:.2f}, YawRate={current_yr:.2f}")
    print(f"DT: {simulation_dt}s")

    # 重置控制器状态（如果需要多次运行或在循环中）
    sfc_controller.reset()

    # 执行单步控制计算
    motor_outputs, ctrl_outputs = run_simpleflight_control_step(
        target_rr, target_pr, target_yr, target_thr,
        current_rr, current_pr, current_yr,
        simulation_dt,
        sf_params,
        sfc_controller,
        sf_mixer,
        device=selected_device
    )

    print("\n--- Controller Outputs (Abstract Commands to Mixer) ---")
    print(f"Roll Command (PID out) : {ctrl_outputs[0].item():.4f}")
    print(f"Pitch Command (PID out): {ctrl_outputs[1].item():.4f}")
    print(f"Yaw Command (PID out)  : {ctrl_outputs[2].item():.4f}")
    print(f"Throttle (Passthrough) : {ctrl_outputs[3].item():.4f}")
    
    print("\n--- Final Motor Outputs (0.0 to 1.0) ---")
    for i, output in enumerate(motor_outputs):
        motor_name = ["FRONT_R", "REAR_L ", "FRONT_L", "REAR_R "][i] # 确保与mixer_quad_x的行对应
        print(f"Motor {i+1} ({motor_name}): {output.item():.4f}")

    # 再次运行以查看PID积分项和last_error的影响
    print("\n--- Running a second step with same inputs to see PID state changes ---")
    # 假设状态没有改变，只是为了看PID内部状态
    motor_outputs2, ctrl_outputs2 = run_simpleflight_control_step(
        target_rr, target_pr, target_yr, target_thr,
        current_rr, current_pr, current_yr, # 保持当前状态不变，观察PID的I和D项变化（如果启用）
        simulation_dt,
        sf_params,
        sfc_controller, # 控制器状态会被保留
        sf_mixer,
        device=selected_device
    )
    print("\n--- Controller Outputs (Step 2) ---")
    print(f"Roll Command (PID out) : {ctrl_outputs2[0].item():.4f}")
    print(f"Pitch Command (PID out): {ctrl_outputs2[1].item():.4f}")
    print(f"Yaw Command (PID out)  : {ctrl_outputs2[2].item():.4f}")
    print(f"Throttle (Passthrough) : {ctrl_outputs2[3].item():.4f}")

    print("\n--- Final Motor Outputs (Step 2) ---")
    for i, output in enumerate(motor_outputs2):
        motor_name = ["FRONT_R", "REAR_L ", "FRONT_L", "REAR_R "][i]
        print(f"Motor {i+1} ({motor_name}): {output.item():.4f}")
