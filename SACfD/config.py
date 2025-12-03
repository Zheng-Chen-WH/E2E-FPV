import numpy as np
import torch
import torch.nn as nn

# 仿真与通用参数（多个模块调用，所以单独设变量，方便统一修改）
dt = 0.1                                                    # MPC轨迹每步步长
"""在GPU并行生成轨迹处理之后N=100时，计算时间减少到了0.005s以下
   N=1000时为0.005-0.007s
   N=10000时为0.009s左右
   网络每次train耗时大约0.003s"""
POS_TOLERANCE = 2                                         # 判定抵达目标的位置误差限 (meters)
VELO_TOLERANCE = 10                                          # 判定抵达目标的速度误差限 (m/s)
ACTION_DIM = 4                                              # 动作为4个PWM值
CONTROL_MAX = 0.73                                          # default:0.66, 最大控制指令范围（simpleflight为速度范围，PX4为加速度，现在是油门信号）
CONTROL_MIN = 0.55                                          # default:0.62, 油门信号下限
# 穿门任务专用参数
WAYPOINT_PASS_THRESHOLD_Y = 0.2                             # 判定无人机穿门的阈值
SCALED_CONTROL_MAX = 1.                                     # 网络输出信号放大以便增大损失函数，对称缩放
SCALED_CONTROL_MIN = -1.                                    # 网络输出信号放大以便增大损失函数，对称缩放
sequence_len = 4                                            # 图像序列长度

# PyTorch设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MPC参数
R_CONTROL_COST_MATRIX_GPU = torch.tensor(np.diag([0.1,      # FR电机控制量
                                                0.1,        # RL控制量
                                                0.1,        # FL控制量
                                                0.1]),      # RR控制量
                                                dtype=torch.float32, device=device) 
mpc_params = { # MPC参数字典
    'waypoint_pass_threshold_y': WAYPOINT_PASS_THRESHOLD_Y,  
    'dt': dt,  
    'control_max': CONTROL_MAX,
    'control_min': CONTROL_MIN,
    'q_state_matrix_gpu': torch.tensor(np.diag([            # 运行状态代价矩阵
            450.0, 0.5, 500.0,                              # x,y,z位置
            50.0, 10.0, 100.0,                              # x,y,z速度
            0.0, 0.0, 0.0, 0.0,                             # 姿态, default (10.0, 100.0, 100.0, 100.0)
            100.0, 10.0, 100.0                              # 角速度
            ]), dtype=torch.float32, device=device),
    'r_control_matrix_gpu':R_CONTROL_COST_MATRIX_GPU,
    'q_terminal_matrix_gpu':torch.tensor(np.diag([          # 终端状态代价矩阵
            450.0, 0.5, 500.0,                              # x,y,z位置
            50.0, 10.0, 100.0,                              # x,y,z速度
            0.0, 0.0, 0.0, 0.0,                             # 姿态, default (10.0, 100.0, 100.0, 100.0)
            100.0, 10.0, 100.0                              # 角速度
            ]), dtype=torch.float32, device=device),
    'q_state_matrix_gpu_two':torch.tensor(np.diag([         # 第二扇门运行状态代价矩阵
            700.0, 0.5, 300.0,                              # x,y,z位置
            50.0, 5.0, 100.0,                               # x,y,z速度
            0.0, 0.0, 0.0, 0.0,                             # 姿态, default (10.0, 100.0, 100.0, 100.0)
            100.0, 10.0, 100.0                              # 角速度
            ]), dtype=torch.float32, device=device),
    'r_control_matrix_gpu_two':R_CONTROL_COST_MATRIX_GPU,
    'q_terminal_matrix_gpu_two':torch.tensor(np.diag([      # 第二扇门终端状态代价矩阵
            700.0, 0.5, 300.0,                              # x,y,z位置
            50.0, 5.0, 100.0,                               # x,y,z速度
            0.0, 0.0, 0.0, 0.0,                             # 姿态, default (10.0, 100.0, 100.0, 100.0)
            100.0, 10.0, 100.0                              # 角速度
            ]), dtype=torch.float32, device=device),
    'static_q_state_matrix_gpu':torch.tensor(np.diag([      # 撞球运行状态代价矩阵
            5.0, 5.0, 10.0,                                 # x,y,z位置
            2.0, 2.0, 10.0,                                 # x,y,z速度
            0.0, 0.0, 0.0, 0.0,                             # 姿态, default (5.0, 10.0, 10.0, 10.0)
            10.0, 5.0, 10.0                                 # 角速度
            ]), dtype=torch.float32, device=device),
    'static_r_control_matrix_gpu':R_CONTROL_COST_MATRIX_GPU,
    'static_q_terminal_matrix_gpu':torch.tensor(np.diag([   # 撞球终端状态代价矩阵
            200.0, 150.0, 800.0,                            # x,y,z位置
            20.0, 20.0, 150.0,                              # x,y,z速度
            0.0, 0.0, 0.0, 0.0,                             # 姿态, default (50.0, 100.0, 100.0, 100.0)
            100.0, 50.0, 100.0                              # 角速度
            ]), dtype=torch.float32, device=device),
    'action_dim': ACTION_DIM,
    'state_dim': 13,                                        # MPC输入状态维度
    'device': device,
    'pos_tolerence': POS_TOLERANCE,
    'align_cost': 300,                                      # 鼓励瞄准目标中心飞
    'vel_align_cost': 500                                   # 鼓励速度矢量指向目标中心
    }

# CEM参数
N_SAMPLE = 100000                                           # 每个CEM采样过程采样数量
CEM_param = {'prediction_horizon': 5,                       # MPC预测长度 (N_steps)
            'n_samples': N_SAMPLE,                          # 每个CEM采样过程采样数量，需要更改上面的 N_SAMPLE
            'n_elites': int(0.1 * N_SAMPLE),                # 精英群体比例
            'n_iter': 1,                                    # 每个MPC优化步的CEM迭代轮数
            'initial_std': mpc_params['control_max'],       # CEM采样初始标准差，给大一点利于探索
            'min_std': 0.05,                                # CEM标准差最小值
            'alpha': 0.8,                                   # CEM方差均值更新时软参数，新的值所占比重
            }

# airsim环境参数
DOOR_FRAMES = ["men_Blueprint", "men_Blueprint2"]           # 门框名称列表
env_params = {'DT': dt,                                     # MPC轨迹每步步长,
            'img_time': 0.025,                              # 拍照间隔
            'door_frames': DOOR_FRAMES,                     # 门框名称列表，在上面的参数处单独修改
            'door_param': {                             # 门的正弦运动参数
                "num": len(DOOR_FRAMES),
                "amplitude": 1.5,                             # 运动幅度（米）
                "frequency": 0.1,                           # 运动频率（Hz）
                "deviation": None,                          # 两个门的初始相位 (set in reset)
                "initial_x_pos": None,                      # 门的初始x位置 (set in reset)
                "start_time":None,                          # 门运动的初始时间
                "width": 2.5,                               # 定义门框宽度
                "height": 2.4,                              # 定义门框高度
                "center": 2.6                               # 门框中心相对地面高度，修正门框物理中心和建模原点在地面的误差
                },
            'POS_TOLERANCE': POS_TOLERANCE,                 # 判定抵达目标的位置误差限 (meters)
            'frames': sequence_len,                         # 照片序列帧数
            'pass_threshold_y': WAYPOINT_PASS_THRESHOLD_Y,  # 判定无人机穿门的阈值
            'control_max': CONTROL_MAX,
            'control_min': CONTROL_MIN,
            'reward_weight': {                          # 奖励函数参数字典
                'W_POS_PROG': 1.0,                          # 位置接近奖励权重
                'W_VEL_ALIGN': 0.5,                         # 速度对齐奖励权重
                'W_VEL_DIR_ALIGN': 0.1,                     # 速度方向指向目标中心奖励权重
                'W_FINAL_PULL': 0.5,                        # 一个始终存在的、朝向最终目标的“微弱引力”
                # 成本惩罚权重
                'W_ACTION_MAG': 0.05,                       # 控制指令幅度
                'W_BODY_RATE': 0.1,                         # 角速度大小
                'W_TIME_COST': 0.05,                        # 时间
                'W_ALIGNMENT': 0.1,                         # 角度对准
                # 终端奖励/惩罚
                'SUCCESS_BONUS': 10,
                'CRASH_PENALTY': -10,
                # 奖励归一化分母
                'REWARD_NORMALIZATION': 5.0,
                # 穿门
                'GATE_PASS_BONUS': 5.0,                      # 穿过每个门的奖励
                'GATE_BYPASS_PENALTY': -5.0,                 # 绕过门的惩罚
                }
            }

# 神经网络与训练参数
first_output_dim = 256                                      # 第一模块（resnet/transformer）输出主向量维度
first_aux_dim = 9                                           # 第一模块辅助头输出维度，6D连续表示姿态+相对下一目标的位置
second_output_dim = 128                                     # 第二模块（GRU）输出主向量维度
second_aux_dim = 6                                          # 第二模块辅助头输出维度，相对下一目标的速度+3D角速度 
actor_param = {"action_dim": ACTION_DIM,                    # 动作网络输出：4个PWM
               "first_module": "ViT",                       # 第一模块类型，ResNet/ViT
               "second_module": "TempT",                    # 第二模块类型，GRU/TempT
               "first_aux_dim": first_aux_dim, 
               "second_aux_dim": second_aux_dim,
               "scaled_max_action": SCALED_CONTROL_MAX,     # 缩放后动作上限值
               "scaled_min_action": SCALED_CONTROL_MIN,     # 缩放后动作下限值
               "print_aux_output": False,                   # 打印辅助头输出结果
               "MLP":{                                  # MLP参数（第三模块）
                    "input_feature_dim": second_output_dim, # 第二模块输出是MLP输入
                    "mlp_state_dim": 3,                     # Pi网络拼接定量状态:目标3d位置
                    "hidden_size": [256, 128, 64],          # MLP隐藏层大小
                    "activation": nn.ReLU,                  # 激活函数
                }, 
               "ResNet":{                               # ResNet网络参数，卷积→最大池化→多卷积阶段→平均池化→输出
                   "frames": sequence_len,                  # 帧数
                    "input_channels": 3,                    # default=3（RGB），输入通道数
                    "first_CNN_layer": {                    # ResNet输入第一层的参数，默认值参考ResNet18
                        "out_put_channel": 64,              # 卷积核数（输出通道数）
                        "kernel_size": 7,                   # 卷积核大小
                        "stride":2,                         # 滑动步幅
                        "padding":3,                        # 边缘填充
                        },
                    "max_pool":{                        # 最大池化层，接在第一层后面
                        "kernel_size": 3,                   # 核尺寸
                        "stride": 2,                        # 步幅
                        "padding": 1                        # 填充
                        },
                    "block_counts": [1,1,1],                # 列表表示resnet阶段数，每个阶段处理相同尺寸特征图；数字表示每个阶段有几个残差块
                    # 阶段之间进行下采样（第一个残差块stride = 2)
                    "channel_scales": [64, 128, first_output_dim], # 每阶段的卷积核数，除第一个残差块外其他残差块输入通道=输出通道
                    # 推荐第一个阶段不进行下采样，保持与第一层的输出通道相同
                    "num_aux_output": first_aux_dim,        # 第一层辅助头
                    "embed_dim": first_output_dim,          # 输出特征向量维度
                    },
                "ViT":{                                 # ViT参数
                    "img_size": (256, 144),                 # 输入图像尺寸
                    "frames": sequence_len,                 # 帧数
                    "patch_size": 16,                       # 一个patch边长
                    "input_channels": 3,                    # 输入图像颜色通道
                    "num_aux_outputs": first_aux_dim,       # 辅助输出头维度
                    "embed_dim": first_output_dim,          # 输出特征向量维度  
                    "depth": 3,                             # ViT中transformer层数
                    "num_heads": 8,                         # 注意力头数
                    "mlp_ratio": 4.0,                       # FFN隐藏层大小比例因数，hidden_dimension = embed_dim * mlp_ratio
                    "dropout": 0.1,                         # dropout比例
                    "activation": 'relu',                   # 激活函数，或者gelu
                    "batch_first": True,                    # 设定输入和输出张量的维度顺序为(Batch, Seq, Dim)
                    "norm_first": True,                     # Pre-Layer Normalization，在自注意力层和FFN之前进行层归一化，能更稳定一些
                    },
               "GRU":{                                  # GRU网络参数
                    "input_dim": first_output_dim,          # 第一模块输出是第二模块输入
                    "gru_hidden_dim": second_output_dim,    # GRU主输出向量维度    
                    "aux_out_dim": second_aux_dim,          # GRU辅助输出维度
                    "layer_num": 2,                         # GRU层数
                    "batch_first": True,                    # default = True, 指定输入和输出张量的维度顺序为 (batch, seq_len, features)
                    "drop_out": 0.3,                        # dropout概率
                    },
                "TemporalTransformer":{                 # 时序transformer参数，具有一个线性层将input_dim对齐到output_dim
                    "input_dim": first_output_dim,
                    "num_aux_outputs": second_aux_dim,      # 辅助输出头维度
                    "embed_dim": second_output_dim,
                    "depth": 3,                             # transformer层数
                    "num_heads": 8,                         # 注意力头数
                    "mlp_ratio": 4.0,                       # FFN隐藏层大小比例因数，hidden_dimension = embed_dim * mlp_ratio
                    "dropout": 0.1,                         # dropout比例
                    "activation": 'relu',                   # 激活函数，或者gelu
                    "batch_first": True,                    # 设定输入和输出张量的维度顺序为(Batch, Seq, Dim)
                    "norm_first": True,                     # Pre-Layer Normalization，在自注意力层和FFN之前进行层归一化，能更稳定一些
                    }
               } 

critic_param = {"state_dim": 21,                            # Q网络状态：世界系速度、无人机姿态（6D连续表示）、本体系角速度、相对下一目标的位置、速度、相对最终目标的位置
                "action_dim": ACTION_DIM,                   # Q网络还需要输入动作
                "hidden_size": [256, 128, 64],              # Q网络隐藏层大小
                "activation": nn.ReLU,                      # 激活函数
                }

SAC_param = {"gamma":0.99,                                  # reward长期衰减因数 (default: 0.99)
            'tau':0.01,                                     # 目标网络软更新时的平滑系数(τ) (default: 0.005), 控制新网络在更新目标网络时与旧目标值的融合程度。
            'alpha':0.1,                                    # 温度系数，控制熵正则项相对Q值重要性 (default: 0.2)
            'seed':20000323,                                # 网络初始化的时候用的随机数种子
            'mu_init_boundary': 0.1,                        # policy的mu层初始化时界限
            'target_update_interval': 20,                   # 目标网络更新的间隔
            'automatic_entropy_tuning': True,              # 自动调整温度系数alpha (default: False)
            'chunk_update': False,                           # 是否使用序列更新，如果使用GRU且想维持长期记忆就需要True；TempT和短序列GRU都为False（否则与现在的buffer不匹配）
            'warm_up_steps': 100000,                        # 学习率预热，在这些updates内学习率线性提升到设定的lr值
            'lr': 1e-4,                                     # 学习率 (default: 0.0003)
            'action_dim': ACTION_DIM,                       # 动作维度
            'loss_dynamic_change_window': 500000,           # il+rl动态权重机制下，loss的线性变换周期长度
            'rl_loss_weight_target': 0.8,                   # 线性衰减时rl_loss占比最终值
            'max_norm_grad': 1.0,                           # default=1.0, 梯度裁剪阈值，当计算出的梯度范数超过这个值时，所有梯度会被等比例缩放 
            'buffer_param':{
                'buffer_configs':{                      # 定义memory结构
                    'expert': 15000,                        # 纯粹收集MPC示范过程的数据，实际执行MPC动作
                    'dagger': 15000                         # 收集dagger过程的数据，现在同时收集NN和MPC动作了，实际执行NN动作
                    }, 
                'recent_size': 32                           # 定义“最近期数据池”范围
                },
            'batch_size': {                             # 定义每个buffer取样数量
                'expert': 64,                               # default=64, expert buffer取样数，用于IL
                'dagger_old': 32,                           # default=64, dagger buffer中，[0:-'dagger_recent']中取样数，解包MPC action用于IL，解包NN action用于RL
                'dagger_recent': 32                         # default=32, 在dagger中抽取最后value组数据
            },
            'loss_weight':{                             # 损失函数权重
                'aux_loss_weight': 0.5,                     # 辅助头总损失权重
                'pos_loss_weight': 1.0,                     # 相对位置损失权重
                'rot_loss_weight': 1.0,                     # 相对姿态损失权重
                'vel_loss_weight': 1.0,                     # 相对速度损失权重
                'ang_vel_loss_weight': 1.0,                 # 相对角速度损失权重
                'il_loss_weight': 5.0                       # 模仿学习损失权重，用于调整量级
                },
            }

PPO_param = {
            # 基础参数
            "gamma": 0.99,                                  # reward长期衰减因数 (default: 0.99)
            "lr": 3e-4,                                     # 学习率 (PPO论文推荐3e-4)
            "seed": 20000323,                               # 网络初始化的时候用的随机数种子
            # PPO 核心参数
            "n_steps": 64,                                  # default=2048, Rollout Buffer 长度 (每次更新前收集多少步数据)
            "mini_batch_size": 32,                          # default=64, 每次更新时的 Mini-batch 大小，现在是"序列块数量"
            "ppo_epoch": 10,                                # 每次 update 循环更新多少次 (K epochs)
            "clip": 0.2,                                    # PPO 裁剪范围 (epsilon), 通常 0.1~0.2
            "gae_lambda": 0.95,                             # GAE 平滑系数 (λ)
            'adv_normalization': True,                      # 是否对优势函数进行归一化
            # 损失函数系数
            "value_loss_coef": 0.5,                         # Value Loss 的权重系数 (c1)
            "entropy_coef": 0.01,                           # 熵正则项的权重系数 (c2)，鼓励探索
            "max_norm_grad": 0.5,                           # 梯度裁剪阈值 (PPO通常比SAC小)
            # 网络初始化
            "mu_init_boundary": 0.01,                       # policy的mu层初始化时界限（PPO通常更小）
            'warm_up_steps': 10000,                         # 学习率预热，在这些updates内学习率线性提升到设定的lr值
            'buffer_param':{                            # Imitation Learning (IL) Buffer参数
                'buffer_configs':{                          # 定义memory结构
                    'expert': 15000,                        # 纯粹收集MPC示范过程的数据
                    'dagger': 15000                         # 收集dagger过程的数据
                }, 
                'recent_size': 32                           # 定义“最近期数据池”范围
            },
            'batch_size': {                             # 定义每个buffer取样数量
                'expert': 64,                               # expert buffer取样数，用于IL
                'dagger_old': 32,                           # dagger buffer中，[0:-'dagger_recent']中取样数
                'dagger_recent': 32                         # 在dagger中抽取最后value组数据
            },
            'rollout_buffer':{                          # Rollout Buffer 参数                                                      
                'device': device,
                'gamma': 0.99,                              # 折扣因子 
                'gae_lambda': 0.95,                         # GAE平滑系数, 1为完全蒙特卡洛（高方差，低偏差），0为时序差分(TD)（低方差，高偏差）
                'seq_len': 8,                               # Recurrent PPO 序列块长度，每个 mini-batch 中的连续时间步数
            },
            # Loss 权重
            'loss_dynamic_change_window': 50000,            # il+rl动态权重机制下，loss的线性变换周期长度
            'rl_loss_weight_target': 0.8,                   # 线性衰减时rl_loss占比最终值
            'loss_weight':{
                'aux_loss_weight': 0.5,                     # 辅助头总损失权重
                'pos_loss_weight': 1.0,                     # 相对位置损失权重
                'rot_loss_weight': 1.0,                     # 相对姿态损失权重
                'vel_loss_weight': 1.0,                     # 相对速度损失权重
                'ang_vel_loss_weight': 1.0,                 # 相对角速度损失权重
                'il_loss_weight': 0.0                       # 模仿学习损失权重（纯PPO可设为0）
            },
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

# 历史遗留参数
SCALER_REFIT_FREQUENCY = 10  # 归一化参数更新频率
FIT_SCALER_SUBSET_SIZE = 2000  # 用于更新归一化参数的样本数

# 补全电机参数
# UAV_revolutions_per_second = UAV_rotor_max_rpm / 60.0
# UAV_max_speed_rad_s = UAV_revolutions_per_second * 2 * np.pi
# UAV_max_speed_sq = UAV_max_speed_rad_s**2
# UAV_max_thrust = UAV_rotor_C_T * air_density * (UAV_revolutions_per_second**2) * (UAV_propeller_diameter**4) # 电机最大推力计算
# UAV_max_torque = UAV_rotor_C_P * air_density * (UAV_revolutions_per_second**2) * (UAV_propeller_diameter**5) / (2 * np.pi) # 电机最大扭矩计算