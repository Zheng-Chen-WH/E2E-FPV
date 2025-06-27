import numpy as np
import torch
import torch.nn as nn

# 仿真与通用参数
DT = 0.10  # MPC轨迹每步步长
img_time = 0.02
"""在GPU并行生成轨迹处理之后N=100时计算时间减少到了0.005s以下
   N=1000时为0.005-0.007s
   N=10000时为0.009s左右
   网络每次train耗时大约0.003s"""
MAX_STEP_PER_EPISODE = 500  # 单个Episode最大时间
POS_TOLERANCE = 1  # 判定抵达目标的位置误差限 (meters)
VELO_TOLERANCE = 1  # 判定抵达目标的速度误差限 (m/s)
CONTROL_MAX = 0.66 # 最大控制指令范围（simpleflight为速度范围，PX4为加速度，现在是油门信号）
CONTROL_MIN = 0.62 # 油门信号下限
SCALED_CONTROL_MAX = 5.0 # 网络输出信号放大以便增大损失函数
SCALED_CONTROL_MIN = -5.0

# CEM参数
PREDICTION_HORIZON = 5  # MPC预测长度 (N_steps)
N_SAMPLES_CEM = 200000  # 每个CEM采样过程采样数量
N_ELITES_CEM = int(0.1 * N_SAMPLES_CEM)  # CEM精英群体数量
N_ITER_CEM = 1  # 每个MPC优化步的CEM迭代轮数
INITIAL_STD_CEM = CONTROL_MAX  # CEM采样初始标准差，给大一点利于探索
MIN_STD_CEM = 0.05  # CEM标准差最小值
ALPHA_CEM = 0.8  # CEM方差均值更新时软参数，新的值所占比重
MPC_STATE_DIM=13

# 神经网络与训练参数 
PI_STATE_DIM = 15  # Pi网络状态:暂定之前三次PWM和目标位置
Q_STATE_DIM = 27 # Q网络状态：暂定无人机姿态（6D连续表示）、世界系速度和本体系角速度、相对两个门的位置、速度、相对目标的位置
RESNET_AUX_DIM = 9 # ResNet辅助头输出维度，6D连续表示姿态+相对下一目标的位置
GRU_AUX_DIM = 6 # GRU辅助头输出维度，相对下一目标的速度+3D角速度
GRU_LAYER = 2 # GRU层数
DROP_OUT = 0.3 # GRU的dropout概率
ACTION_DIM = 4  # 动作向量维度，4个PWM
NN_HIDDEN_SIZE = [128,128,128, 64]  # 隐藏层大小
LEARNING_RATE = 1e-4  # 学习率
BUFFER_SIZE = 100000  # buffer大小
BATCH_SIZE = 128  # 训练batch size
NN_TRAIN_EPOCHS_PER_STEP = 1  # 每次训练时训练epoch数
MIN_EPISODES_FOR_TRAINING = 10  # 开始训练时最小episode数
EPISODE_EXPLORE = 5  # 随机探索episode数
SCALER_REFIT_FREQUENCY = 10  # 归一化参数更新频率
FIT_SCALER_SUBSET_SIZE = 2000  # 用于更新归一化参数的样本数
NUM_TRANSFORMER_FRAMES = 4
NUM_EPISODES = 1000  # 训练最大episode数
WARM_UP = 10000 # 学习率预热，在这些updates内学习率线性提升到设定的lr值
AUX_LOSS_WEIGHT = 1.0 # 辅助头总损失权重
POS_LOSS_WEIGHT = 0.1 # 相对位置损失权重
ROT_LOSS_WEIGHT = 20.0  # 相对姿态损失权重
VEL_LOSS_WEIGHT = 1.0  # 相对速度损失权重
ANG_VEL_LOSS_WEIGHT = 20.0 # 相对角速度损失权重


# 穿门任务专用参数
WAYPOINT_PASS_THRESHOLD_Y = 0.5  # 判定无人机穿门的阈值

# PyTorch设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 代价权重矩阵
# # 控制代价矩阵
# R_CONTROL_COST_NP = R_CONTROL_COST_NP = np.diag([
#     1000,  # FR电机控制量
#     1000,  # RL控制量
#     1000,  # FL控制量
#     1000   # RR控制量
# ])
# R_CONTROL_COST_MATRIX_GPU = torch.tensor(R_CONTROL_COST_NP, dtype=torch.float32, device=device)

# # 运行状态代价矩阵
# # 索引：0-2: 位置 (x,y,z), 3-5: 速度 (vx,vy,vz), 6-8: 姿态 (p,r,y), 9-11: 角速度 (wx,wy,wz)
# Q_STATE_COST_NP = np.diag([
#     250.0, 1.0, 10.0,  # x,y,z位置
#     100.0, 10.0, 100.0,   # x,y,z速度
#     10.0, 100.0, 100.0, 100.0,     # 姿态
#     100.0, 10.0, 100.0      # 角速度
# ])
# Q_STATE_COST_MATRIX_GPU = torch.tensor(Q_STATE_COST_NP, dtype=torch.float32, device=device)

# # 终端状态代价矩阵
# Q_TERMINAL_COST_NP = np.diag([
#     250.0, 1.0, 10.0,  # x,y,z位置
#     100.0, 10.0, 100.0,   # x,y,z速度
#     10.0, 100.0, 100.0, 100.0,     # 姿态
#     100.0, 10.0, 100.0      # 角速度
# ])
# Q_TERMINAL_COST_MATRIX_GPU = torch.tensor(Q_TERMINAL_COST_NP, dtype=torch.float32, device=device)

# 减0后的代价权重矩阵
# 控制代价矩阵
R_CONTROL_COST_NP = np.diag([
    0.1,  # FR电机控制量
    0.1,  # RL控制量
    0.1,  # FL控制量
    0.1   # RR控制量
])
R_CONTROL_COST_MATRIX_GPU = torch.tensor(R_CONTROL_COST_NP, dtype=torch.float32, device=device)

# 运行状态代价矩阵
# 索引：0-2: 位置 (x,y,z), 3-5: 速度 (vx,vy,vz), 6-9: 姿态 (p,r,y), 10-12: 角速度 (wx,wy,wz)
Q_STATE_COST_NP = np.diag([
    250.0, 1.0, 50.0,  # x,y,z位置
    100.0, 10.0, 100.0,   # x,y,z速度
    10.0, 100.0, 100.0, 100.0,     # 姿态
    100.0, 10.0, 100.0      # 角速度
])
Q_STATE_COST_MATRIX_GPU = torch.tensor(Q_STATE_COST_NP, dtype=torch.float32, device=device)

# 终端状态代价矩阵
Q_TERMINAL_COST_NP = np.diag([
    250.0, 1.0, 50.0,  # x,y,z位置
    100.0, 10.0, 100.0,   # x,y,z速度
    10.0, 100.0, 100.0, 100.0,     # 姿态
    100.0, 10.0, 100.0      # 角速度
])
Q_TERMINAL_COST_MATRIX_GPU = torch.tensor(Q_TERMINAL_COST_NP, dtype=torch.float32, device=device)

# 第二扇门特供
# 索引：0-2: 位置 (x,y,z), 3-5: 速度 (vx,vy,vz), 6-9: 姿态 (p,r,y), 10-12: 角速度 (wx,wy,wz)
Q_STATE_COST_NP_TWO = np.diag([
    500.0, 1.0, 200.0,  # x,y,z位置
    150.0, 5.0, 100.0,   # x,y,z速度
    10.0, 100.0, 100.0, 100.0,     # 姿态
    100.0, 10.0, 100.0      # 角速度
])
Q_STATE_COST_MATRIX_GPU_TWO = torch.tensor(Q_STATE_COST_NP_TWO, dtype=torch.float32, device=device)

# 终端状态代价矩阵
Q_TERMINAL_COST_NP_TWO = np.diag([
    500.0, 1.0, 200.0,  # x,y,z位置
    150.0, 5.0, 100.0,   # x,y,z速度
    10.0, 100.0, 100.0, 100.0,     # 姿态
    100.0, 10.0, 100.0      # 角速度
])
Q_TERMINAL_COST_MATRIX_GPU_TWO = torch.tensor(Q_TERMINAL_COST_NP_TWO, dtype=torch.float32, device=device)

# 静态代价权重矩阵
# 静态目标控制代价矩阵
STATIC_R_CONTROL_COST_NP = np.diag([
    0.1,  # FR电机控制量
    0.1,  # RL控制量
    0.1,  # FL控制量
    0.1   # RR控制量
])
STATIC_R_CONTROL_COST_MATRIX_GPU = torch.tensor(STATIC_R_CONTROL_COST_NP, dtype=torch.float32, device=device)

# 静态目标运行状态代价矩阵
# 索引：0-2: 位置 (x,y,z), 3-5: 速度 (vx,vy,vz), 6-9: 姿态 (p,r,y), 10-12: 角速度 (wx,wy,wz)
STATIC_Q_STATE_COST_NP = np.diag([
    5.0, 5.0, 10.0,  # x,y,z位置
    2.0, 2.0, 10.0,   # x,y,z速度
    5.0, 10.0, 10.0, 10.0,     # 姿态
    10.0, 5.0, 10.0      # 角速度
])
STATIC_Q_STATE_COST_MATRIX_GPU = torch.tensor(STATIC_Q_STATE_COST_NP, dtype=torch.float32, device=device)

# 静态目标终端状态代价矩阵
STATIC_Q_TERMINAL_COST_NP = np.diag([
    200.0, 150.0, 800.0,  # x,y,z位置
    20.0, 20.0, 150.0,   # x,y,z速度
    50.0, 100.0, 100.0, 100.0,     # 姿态
    100.0, 50.0, 100.0      # 角速度
])
STATIC_Q_TERMINAL_COST_MATRIX_GPU = torch.tensor(STATIC_Q_TERMINAL_COST_NP, dtype=torch.float32, device=device)

# AirSim参数
door_frames_names = ["men_Blueprint", "men2_Blueprint"]
door_param= { # 门的正弦运动参数
            "num": len(door_frames_names),
            "amplitude": 2,  # 运动幅度（米）
            "frequency": 0.1,  # 运动频率（Hz）
            "deviation": None,  # 两个门的初始相位 (set in reset)
            "initial_x_pos": None,  # 门的初始x位置 (set in reset)
            "start_time":None # 门运动的初始时间
        }

# 无人机动力学模型参数

UAV_mass=1.0 # 无人机总重量
UAV_arm_length = 0.2275 # 无人机臂长度
UAV_rotor_z_offset = 0.025 # 电机高度

# 电机参数与空气密度
UAV_rotor_C_T = 0.109919 # 螺旋桨推力系数
UAV_rotor_C_P = 0.040164 # 螺旋桨扭矩系数
air_density = 1.225 # 空气密度
UAV_rotor_max_rpm = 6396.667 # 电机最大转速
UAV_propeller_diameter = 0.2286 # 螺旋桨直径
UAV_propeller_height = 0.01 # 螺旋桨截面高度
UAV_tc = 0.005 # 无人机电机滤波时间常数, 越大滤波效果越明显
UAV_max_thrust = 4.179446268 # 无人机单电机最大推力
UAV_max_torque = 0.055562 # 无人机单电机最大扭矩
UAV_linear_drag_coefficient = 0.325 # 线阻力系数
UAV_angular_drag_coefficient = 0.0 # 角阻力系数；有值的时候每个DT都会导致最后变成推力矩与阻力矩平衡，无人机y方向角速度锁定在0.02}
UAV_body_mass_fraction = 0.78 # 无人机中心盒重量占比
UAV_body_mass = UAV_mass * UAV_body_mass_fraction # 无人机中心盒质量
UAV_motor_mass = UAV_mass * (1-UAV_body_mass_fraction) / 4.0 # 电机质量
UAV_dim_x = 0.180; UAV_dim_y = 0.110; UAV_dim_z = 0.040 # 机身盒尺寸
Ixx_body = UAV_body_mass / 12.0 * (UAV_dim_y**2.0 + UAV_dim_z**2.0) # 机身对三个轴的转动惯量
Iyy_body = UAV_body_mass / 12.0 * (UAV_dim_x**2.0 + UAV_dim_z**2.0)
Izz_body = UAV_body_mass / 12.0 * (UAV_dim_x**2.0 + UAV_dim_y**2.0)
L_eff_sq = (UAV_arm_length * torch.cos(torch.tensor(torch.pi / 4.0)))**2.0 # 电机位置偏移量的平方
rotor_z_dist_sq = UAV_rotor_z_offset**2.0 # 电机高度偏移量
Ixx_motors = 4 * UAV_motor_mass * (L_eff_sq + rotor_z_dist_sq)
Iyy_motors = 4 * UAV_motor_mass * (L_eff_sq + rotor_z_dist_sq)
Izz_motors = 4 * UAV_motor_mass * (2.0 * L_eff_sq)
UAV_inertia_diag = torch.tensor([ # 转动惯量矩阵
            Ixx_body + Ixx_motors,
            Iyy_body + Iyy_motors,
            Izz_body + Izz_motors
        ], device=device, dtype=torch.float32)
UAV_xy_area = UAV_dim_x * UAV_dim_y + 4.0 * torch.pi * UAV_propeller_diameter**2
UAV_yz_area = UAV_dim_y * UAV_dim_z + 4.0 * torch.pi * UAV_propeller_diameter * UAV_propeller_height
UAV_xz_area = UAV_dim_x * UAV_dim_z + 4.0 * torch.pi * UAV_propeller_diameter * UAV_propeller_height
drag_box = 0.5 * UAV_linear_drag_coefficient * torch.tensor([UAV_yz_area, UAV_xz_area, UAV_xy_area], device=device, dtype=torch.float32) # 三轴阻力系数“盒”

# 补全电机参数
# UAV_revolutions_per_second = UAV_rotor_max_rpm / 60.0
# UAV_max_speed_rad_s = UAV_revolutions_per_second * 2 * np.pi
# UAV_max_speed_sq = UAV_max_speed_rad_s**2
# UAV_max_thrust = UAV_rotor_C_T * air_density * (UAV_revolutions_per_second**2) * (UAV_propeller_diameter**4) # 电机最大推力计算
# UAV_max_torque = UAV_rotor_C_P * air_density * (UAV_revolutions_per_second**2) * (UAV_propeller_diameter**5) / (2 * np.pi) # 电机最大扭矩计算