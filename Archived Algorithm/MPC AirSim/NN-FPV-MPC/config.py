import numpy as np
import torch

# 仿真与通用参数
DT = 0.5  # MPC轨迹每步步长
"""在GPU并行生成轨迹处理之后N=100时计算时间减少到了0.005s以下
   N=1000时为0.005-0.007s
   N=10000时为0.009s左右
   网络每次train耗时大约0.003s"""
MAX_SIM_TIME_PER_EPISODE = 40  # 单个Episode最大时间
NUM_EPISODES = 10000  # 训练最大episode数
POS_TOLERANCE = 1  # 判定抵达目标的位置误差限 (meters)
VELO_TOLERANCE = 1  # 判定抵达目标的速度误差限 (m/s)
ACCEL_MAX = 10.0  # 最大控制指令范围（simpleflight为速度范围，PX4为加速度）

# CEM参数
PREDICTION_HORIZON = 5  # MPC预测长度 (N_steps)
N_SAMPLES_CEM = 10000  # 每个CEM采样过程采样数量
N_ELITES_CEM = int(0.1 * N_SAMPLES_CEM)  # CEM精英群体数量
N_ITER_CEM = 3  # 每个MPC优化步的CEM迭代轮数
INITIAL_STD_CEM = ACCEL_MAX  # CEM采样初始标准差，给大一点利于探索
MIN_STD_CEM = 0.05  # CEM标准差最小值
ALPHA_CEM = 0.9  # CEM方差均值更新时软参数，新的值所占比重

# 神经网络与训练参数
STATE_DIM = 12  # 状态向量维度
ACTION_DIM = 3  # 动作向量维度
NN_HIDDEN_SIZE = 128  # 隐藏层大小
LEARNING_RATE = 5e-5  # 学习率
BUFFER_SIZE = 100000  # buffer大小
BATCH_SIZE = 64  # 训练batch size
NN_TRAIN_EPOCHS_PER_STEP = 5  # 每次训练时训练epoch数
MIN_BUFFER_FOR_TRAINING = BATCH_SIZE  # 开始训练时buffer最小容量
EPISODE_EXPLORE = 5  # 随机探索episode数
SCALER_REFIT_FREQUENCY = 10  # 归一化参数更新频率
FIT_SCALER_SUBSET_SIZE = 2000  # 用于更新归一化参数的样本数

# 穿门任务专用参数
WAYPOINT_PASS_THRESHOLD_Y = -0.5  # 判定无人机穿门的阈值，负值拟合到达门前一点然后冲过去

# PyTorch设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 代价权重矩阵
# 控制代价矩阵
R_CONTROL_COST_NP = R_CONTROL_COST_NP = np.diag([
    -0.01,  # x控制量，取负鼓励
    0.01,   # y控制量
    0.01     # z控制量
])
R_CONTROL_COST_MATRIX_GPU = torch.tensor(R_CONTROL_COST_NP, dtype=torch.float32, device=device)

# 运行状态代价矩阵
# 索引：0-2: 位置 (x,y,z), 3-5: 速度 (vx,vy,vz), 6-8: 姿态 (p,r,y), 9-11: 角速度 (wx,wy,wz)
Q_STATE_COST_NP = np.diag([
    200.0, 10.0, 100.0,  # x,y,z位置
    50.0, 100.0, 0.0,   # x,y,z速度
    0.0, 0.0, 0.0,     # 姿态
    0.0, 0.0, 0.0      # 角速度
])
Q_STATE_COST_MATRIX_GPU = torch.tensor(Q_STATE_COST_NP, dtype=torch.float32, device=device)

# 终端状态代价矩阵
Q_TERMINAL_COST_NP = np.diag([
    2000.0, 100.0, 1000.0, # x,y,z位置
    500.0, 1000.0, 0.0,   # x,y,z速度
    0.0, 0.0, 0.0,       # 姿态
    0.0, 0.0, 0.0        # 角速度
])
Q_TERMINAL_COST_MATRIX_GPU = torch.tensor(Q_TERMINAL_COST_NP, dtype=torch.float32, device=device)

# AirSim参数
door_frames_names = ["men_Blueprint", "men2_Blueprint"]
door_param= { # 门的正弦运动参数
            "amplitude": 2,  # 运动幅度（米）
            "frequency": 0.1,  # 运动频率（Hz）
            "deviation": None,  # 两个门的初始相位 (set in reset)
            "initial_x_pos": None,  # 门的初始x位置 (set in reset)
            "start_time":None # 门运动的初始时间
        }