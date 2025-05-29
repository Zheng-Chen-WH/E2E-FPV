import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import math
import airsim
from datetime import datetime
import time
import pickle

# 系统参数和常量
DT = 0.1 # N=15时计算耗时约0.18-0.19s，N=10时0.12-0.13秒（加上AirSim只会更慢）
"""但是在GPU并行生成轨迹处理之后N=100时计算时间减少到了0.005s以下
   N=1000时为0.005-0.007s
   N=10000时为0.009s左右
   网络每次train耗时大约0.003s"""
MAX_SIM_TIME_PER_EPISODE = 40 # 单个Episode最大时间
NUM_EPISODES = 10000 # 训练最大episode数
POS_TOLERANCE = 1 # 接近到1m内就是成功
VELO_TOLERANCE = 1 # 速度相差1m/s以内
ACCEL_MAX = 10.0

PREDICTION_HORIZON = 15
N_SAMPLES_CEM = 1000
N_ELITES_CEM = int(0.1*N_SAMPLES_CEM)
N_ITER_CEM = 5
INITIAL_STD_CEM = ACCEL_MAX # 初始方差给大一点利于探索
MIN_STD_CEM = 0.05
ALPHA_CEM = 0.9 # 软更新时新的值所占比重

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
R_CONTROL_COST_MATRIX = 0.001 * np.eye(3) # 控制量代价矩阵
# R_CONTROL_COST_MATRIX = 0.0 * np.eye(3) # 不在乎代价的控制量代价矩阵
R_CONTROL_COST_MATRIX_GPU = torch.tensor(R_CONTROL_COST_MATRIX, dtype=torch.float32, device=device)
Q_STATE_COST_MATRIX = np.diag([10.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # 状态包含无人机位置、速度、角速度，仅对位置和x方向速度有要求
Q_STATE_COST_MATRIX_GPU=torch.tensor(Q_STATE_COST_MATRIX, dtype=torch.float32, device=device)
Q_TERMINAL_COST_MATRIX = np.diag([50.0, 50.0, 50.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # 终端也仅考虑位置和x方向速度
Q_TERMINAL_COST_MATRIX_GPU=torch.tensor(Q_TERMINAL_COST_MATRIX, dtype=torch.float32, device=device)

STATE_DIM = 12
ACTION_DIM = 3
NN_HIDDEN_SIZE = 128
LEARNING_RATE = 5e-5 # 学习率
BUFFER_SIZE = 100000 # buffer容量
BATCH_SIZE = 64
NN_TRAIN_EPOCHS_PER_STEP = 5 # 每个step对网络进行几次更新
MIN_BUFFER_FOR_TRAINING = BATCH_SIZE # buffer中有这些样本之后才开始训练
EPISODE_EXPLORE = 5 # 开始训练前自由探索的次数
WAYPOINT_PASS_THRESHOLD_Y = -0.5 # 判定无人机穿门的阈值
SCALER_REFIT_FREQUENCY=10 # 每10episode重新拟合一次scaler
FIT_SCALER_SUBSET_SIZE=2000 # 用来拟合scaler的样本数



# AirSim仿真环境
class env:
    def __init__(self, dt):
        # 连接到AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        # 获取当前实际时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 设置仿真时间为实际时间
        self.client.simSetTimeOfDay(True, start_datetime=current_time, celestial_clock_speed=1)
        self.target_distance=1
        self.i=0
        # 门框的名称（确保与UE4中的名称一致）
        self.door_frames=["men_Blueprint","men2_Blueprint"]
        self.DT=dt
        # 门框正弦运动运动参数
        self.door_param={"amplitude": 2,   # 运动幅度（米）
                         "frequency": 0.1, # 运动频率（Hz）
                         "deviation": None, # 两个门的初始相位
                         "initial_x_pos": None, # 门的初始x位置
                         } 

    def movedoor(self, door_frame, position):
        new_pose = airsim.Vector3r(position[0], position[1], position[2])
        self.client.simSetObjectPose(door_frame, airsim.Pose(new_pose, self.initial_pose.orientation), True)
    
    def state_fpv(self):
        # 获取无人机状态
        FPV_state = self.client.getMultirotorState()
        # 获取位置
        position = FPV_state.kinematics_estimated.position
        FPV_pos=np.array([position.x_val,
                          position.y_val,
                          position.z_val])
        # 获取线速度
        linear_velocity = FPV_state.kinematics_estimated.linear_velocity
        FPV_vel=np.array([linear_velocity.x_val,
                          linear_velocity.y_val,
                          linear_velocity.z_val])
        
        #获取姿态角
        orientation_q = FPV_state.kinematics_estimated.orientation
        # 将四元数转换为欧拉角 (radians)
        # 返回的是 (pitch, roll, yaw) in radians
        # 注意这里的顺序是 pitch, roll, yaw
        pitch, roll, yaw = airsim.to_eularian_angles(orientation_q)
        # roll_deg = math.degrees(roll)
        # pitch_deg = math.degrees(pitch)
        # yaw_deg = math.degrees(yaw)
        FPV_attitude=np.array((pitch, roll, yaw))

        # 获取角速度
        angular_velocity = FPV_state.kinematics_estimated.angular_velocity
        FPV_ww=np.array([angular_velocity.x_val,
                         angular_velocity.y_val,
                         angular_velocity.z_val])
        return np.concatenate((FPV_pos, FPV_vel, FPV_attitude, FPV_ww))

    def reset(self,seed=None): 

        for attempt in range(10):
            # print(f"Attempting to reset and initialize drone (Attempt {attempt + 1}/{10})...")
            try:
                self.client.reset()
                # 短暂等待AirSim完成重置
                time.sleep(0.5) # 根据需要调整
                self.client.enableApiControl(True)
                self.client.armDisarm(True)
                time.sleep(0.5) # 等待armDisarm生效

                # 验证状态
                if not self.client.isApiControlEnabled():
                    print("Failed to enable API control after reset.")
                    continue # 如果apicontrol不成功就直接从最上面的reset重新开始
                
                # (可选) 检查是否解锁，但通常armDisarm(True)后即是。
                # multirotor_state = current_client.getMultirotorState()
                # if multirotor_state.landed_state != airsim.LandedState.Flying and not multirotor_state.rc_data.is_armed: # 检查是否仍在地面且未解锁
                #     print("Drone is not armed or still landed after armDisarm call.")
                #     # 你可能需要更复杂的逻辑来判断arm是否真的成功，取决于具体情况
                #     # 简单起见，这里我们假设armDisarm成功后，无人机可以接收指令
                
                # print("Drone reset and initialized successfully.")
                reset_success = True
                break # 成功，跳出重试循环
            except Exception as e:
                print(f"Error during drone initialization (Attempt {attempt + 1}): {e}")
                # 尝试重新连接，以防是连接问题导致的reset失败
                try:
                    self.client.confirmConnection() # 尝试重新确认连接
                except Exception as conn_err:
                    print(f"Failed to re-confirm connection: {conn_err}")
                    # 如果连不上，可能需要更高级的错误处理，比如退出脚本
                    break # 无法连接，停止尝试reset
                time.sleep(1) # 等待一段时间再重试

        # if seed is None:
        #     np.random.seed(self.i)
        # else:
        #     np.random.seed(seed)

        # 定义无人机的初始位置和方向
        # FPV_position=np.array([np.random.uniform(-3,3), np.random.uniform(-5,5), -1.0])
        # initial_drone_position = airsim.Vector3r(FPV_position[0],FPV_position[1],FPV_position[2])  # 定义位置 (x=10, y=20, z=-0.5)
        # yaw = math.radians(90)  # 90 度（朝向正 y 轴）

        # 标记重要分段点
        self.way_point=[]
        # self.way_point.append(FPV_position[1])
        self.way_point.append(0.0)

        # # 创建 Pose 对象
        # initial_drone_pose = airsim.Pose(initial_drone_position, airsim.to_quaternion(0.0, 0.0, yaw))
        # # 设置无人机初始位置
        # self.client.simSetVehiclePose(initial_drone_pose, ignore_collision=True)
        
        self.client.takeoffAsync().join()
        time.sleep(0.5)
        collision_info = self.client.simGetCollisionInfo()

        # 获取门框的初始位姿并随机修改
        self.initial_door_x_positions=[]
        self.initial_door_y_positions=[]
        self.door_x_positions=[]
        self.door_z_positions=[]
        self.door_x_velocity=np.array([0.0,0.0])
        self.door_param["deviation"] = np.random.uniform(0, 10, size=len(self.door_frames)) # 相位差
        for ii, i in enumerate(self.door_frames):
            try:
                self.initial_pose = self.client.simGetObjectPose(i)
                self.initial_door_position = self.initial_pose.position
                # print(f"Initial Position: X={initial_position.x_val}, Y={initial_position.y_val}, Z={initial_position.z_val}")
                new_x=0+np.random.uniform(-1,1)
                new_y=(ii+1)*15+np.random.uniform(-2,2)
                self.movedoor(i,np.array([new_x,new_y,self.initial_door_position.z_val]))
                self.initial_door_x_positions.append(new_x)
                self.door_x_positions.append(new_x)
                self.door_z_positions.append(self.initial_door_position.z_val)
                self.way_point.append(new_y)
            except Exception as e:
                print(f"Error: {e}") 
                print(f"请确保场景中存在名为 '{i}' 的对象。")
                exit()
        self.door_param["initial_x_pos"] = self.initial_door_x_positions
        self.target_state=np.array([np.random.uniform(-5,5),
                                 np.random.uniform(38,42),
                                 np.random.uniform(-2,-1),
                                 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        new_ball_pose = airsim.Vector3r(self.target_state[0], self.target_state[1], self.target_state[2])
        self.way_point.append(self.target_state[1])
        self.client.simSetObjectPose("OrangeBall_Blueprint", airsim.Pose(new_ball_pose, self.client.simGetObjectPose("OrangeBall_Blueprint")), True)
        
        # 开始时间
        self.start_time = time.time()
        # 计算当前时间
        t = time.time() - self.start_time
        for ii,value in enumerate(self.door_frames):
            # pose = self.client.simGetObjectPose(value) #相当于把门的状态存在ue4里，每次用的时候直接读取
            # position = pose.position
            # 计算门框的新位置、速度
            new_x = self.door_param["initial_x_pos"][ii] + self.door_param["amplitude"] * math.sin(2 * math.pi * self.door_param["frequency"] * t + self.door_param["deviation"][ii])# + deviation)
            self.door_x_velocity[ii] = 2*math.pi * self.door_param["frequency"] * self.door_param["amplitude"] * math.cos(2 * math.pi * self.door_param["frequency"] * t + self.door_param["deviation"][ii])
            
            # 设置门框的新位置
            self.movedoor(value,np.array([new_x, self.way_point[ii+1], self.door_z_positions[ii]]))

        self.info=0
        self.done=False
        state_fpv=self.state_fpv()
        
        return state_fpv, self.target_state, self.way_point, self.door_z_positions, self.door_x_positions, self.door_x_velocity, \
              self.start_time, self.door_param

    def step(self, control):#,red_action

        # 发送速度指令
        self.client.moveByVelocityAsync(float(control[0]), float(control[1]), float(control[2]), 2.0) 

        '''考虑到MPC的计算耗时tc，这里有两种处理方法
           第一种是直接延迟DT，状态转移总时间是DT+tc
           第二种是延迟DT-tc，状态转移总时间仍然是DT
           两种方法下MPC都是用t时刻的St计算控制序列，但是施加控制时的状态都是St+tc
           需要研究哪种处理更好
           祈祷NN能解决tc'''
        time.sleep(self.DT)    

        # 计算当前时间
        t = time.time() - self.start_time
        for ii,value in enumerate(self.door_frames):
            # 计算门框的新位置
            new_x = self.door_param["initial_x_pos"][ii] + self.door_param["amplitude"] * math.sin(2 * math.pi * self.door_param["frequency"] * t + self.door_param["deviation"][ii])# + deviation)
            self.door_x_velocity[ii] = 2*math.pi * self.door_param["frequency"] * self.door_param["amplitude"] * math.cos(2 * math.pi * self.door_param["frequency"] * t + self.door_param["deviation"][ii])
            
            # 设置门框的新位置
            self.movedoor(value,np.array([new_x, self.way_point[ii+1], self.door_z_positions[ii]]))

        state_fpv=self.state_fpv()
        collision_info = self.client.simGetCollisionInfo()
        self.collide=False
        if  collision_info.time_stamp/1e9 > self.start_time + 0.5:
            self.collide=True
        # print(state_fpv)
        return state_fpv, self.door_x_positions, self.door_x_velocity, self.collide

class Scaler:
    def __init__(self, dim):
        self.dim = dim
        self.mean = torch.zeros(dim, dtype=torch.float32)
        self.std = torch.ones(dim, dtype=torch.float32)
        self.fitted = False

    def fit(self, data_tensor):
        """
        从样本中计算均值与方差
        样本需要是2维tensor, 每行都是一个sample.
        """
        if data_tensor.ndim == 1:
            data_tensor = data_tensor.unsqueeze(0)
        if data_tensor.shape[0] == 0:
            print("Warning: Trying to fit scaler with empty data.")
            return

        self.mean = torch.mean(data_tensor, dim=0)
        self.std = torch.std(data_tensor, dim=0)
        # 防止标准差为0 (如果某个特征在所有样本中都一样)
        self.std = torch.where(self.std < 1e-7, torch.ones_like(self.std) * 1e-7, self.std)
        self.fitted = True
        # print(f"Scaler fitted. Mean: {self.mean.cpu().numpy()}, Std: {self.std.cpu().numpy()}")

    def transform(self, data_tensor):
        if not self.fitted:
            # print("Warning: Scaler not fitted yet. Returning original data.")
            return data_tensor
        if data_tensor.device != self.mean.device: # 确保在同一设备上
            self.mean = self.mean.to(data_tensor.device)
            self.std = self.std.to(data_tensor.device)
        return (data_tensor - self.mean) / self.std

    def inverse_transform(self, data_tensor_scaled):
        if not self.fitted:
            return data_tensor_scaled
        if data_tensor_scaled.device != self.mean.device: # 确保在同一设备上
            self.mean = self.mean.to(data_tensor_scaled.device)
            self.std = self.std.to(data_tensor_scaled.device)
        return data_tensor_scaled * self.std + self.mean

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

# 动力学模型网络
class DynamicsNN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(DynamicsNN, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, state_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        next_state_pred = self.fc3(x)
        return next_state_pred

# ReplayBuffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, next_state):
        self.buffer.append((state, action, next_state))
    def sample(self, batch_size):
        state, action, next_state = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(next_state)
    def get_raw_data(self): # 返回所有存储的原始数据
        raw_states, raw_actions, raw_next_states = zip(*list(self.buffer))
        return np.array(raw_states), np.array(raw_actions), np.array(raw_next_states)
    def __len__(self):
        return len(self.buffer)

def cost_function_gpu(
    predicted_states_batch,       # PyTorch Tensor (n_samples, horizon + 1, state_dim) on GPU
    control_sequences_batch,      # PyTorch Tensor (n_samples, horizon, action_dim) on GPU
    current_mpc_target_state_gpu, # PyTorch Tensor (state_dim,) or (1, state_dim) on GPU
    ):
    """
    在GPU上批量计算轨迹成本。

    Args:
        predicted_states_batch: 预测的状态轨迹批次。
                                 形状: (n_samples, PREDICTION_HORIZON + 1, state_dim)
                                 其中 [:, 0, :] 是初始状态。
        control_sequences_batch: 采样的控制序列批次。
                                  形状: (n_samples, PREDICTION_HORIZON, action_dim)
        current_mpc_target_state_gpu: 当前MPC的目标状态。
                                       形状: (state_dim,) 或 (1, state_dim) 以便广播。
        Q_STATE_COST_MATRIX_gpu: 状态成本矩阵 Q。
        R_CONTROL_COST_MATRIX_gpu: 控制成本矩阵 R。
        Q_TERMINAL_COST_MATRIX_gpu: 终端状态成本矩阵 P (或 Q_terminal)。

    Returns:
        total_costs_batch: 每个样本的总成本。形状: (n_samples,)
    """
    n_samples = predicted_states_batch.shape[0] #第一个维度数量即样本数量
    prediction_horizon = control_sequences_batch.shape[1] # 从控制序列获取实际的控制序列长度H

    # 计算状态成本
    
    # 提取用于计算状态成本的预测状态 (从 t=1 到 t=H)
    # predicted_states_batch[:, 0, :] 是初始状态 s_0 （提取了所有样本的第0步s0）
    predicted_states = predicted_states_batch[:, 1 : prediction_horizon + 1, :] # 预测序列长度有N+1，所以是1:N+1
    # 形状: (n_samples, PREDICTION_HORIZON, state_dim)

    # 计算每个时间步的状态误差
    # target 需要扩展到 (1, 1, state_dim) 以便与 (n_samples, PREDICTION_HORIZON, state_dim) 进行广播
    state_error = predicted_states - current_mpc_target_state_gpu.unsqueeze(0) # unsqueeze在指定位置加一个1的维度，squeeze只能减去指定位置为1的维度
    # 形状: (n_samples, PREDICTION_HORIZON, state_dim)

    # 计算状态运行成本: state_error_t.T @ Q @ state_error_t
    # 爱因斯坦求和约定einsum: 'khi,ij,khj->k'，通过指定字符串定义张量操作
    # 输入部分 (khi,ij,khj): 描述了参与运算的三个输入张量的维度。逗号分隔每个张量的标签。
    # 输出部分 (k): 描述了运算结果张量的维度。
    # 爱因斯坦求和的核心规则：
    # 1.重复的索引意味着乘积和求和（缩并 Contract）: 
    # 如果一个索引字母同时出现在输入部分的不同张量标签中，或者在同一个张量标签中多次出现（这里没有这种情况），那么运算结果将沿着这些重复的维度进行乘积累加。
    # 2.未出现在输出部分的索引意味着被求和掉: 如果一个索引字母出现在输入部分的标签中，但没有出现在 -> 右边的输出部分标签中，那么结果张量将沿着这个维度进行求和。
    # 3.出现在输出部分的索引会被保留: 如果一个索引字母出现在输入部分，并且也出现在输出部分，那么这个维度将在结果张量中被保留下来。
    state_costs = torch.einsum('khi,ij,khj->k',
                                       state_error,
                                       Q_STATE_COST_MATRIX_GPU,
                                       state_error)

    # 计算控制运行成本: control_t.T @ R @ control_t
    # control_sequences_batch 形状: (n_samples, PREDICTION_HORIZON, action_dim)
    control_costs = torch.einsum('khi,ij,khj->k',
                                         control_sequences_batch,
                                         R_CONTROL_COST_MATRIX_GPU,
                                         control_sequences_batch)
    
    # 计算终端成本
    terminal_state_batch = predicted_states_batch[:, -1, :] # 取最后一个状态
    # 形状: (n_samples, state_dim)
    
    # 终端目标
    terminal_target_state = current_mpc_target_state_gpu[-1, :]
    terminal_state_error = terminal_state_batch - terminal_target_state # target (1,S) -> (S) for (K,S)-(S)
    # 形状: (n_samples, state_dim)

    # 终端状态成本: terminal_error.T @ P @ terminal_error
    # 使用 einsum: 'ki,ij,kj->k'
    # k: n_samples, i: state_dim, j: state_dim
    terminal_costs = torch.einsum('ki,ij,kj->k',
                                  terminal_state_error,
                                  Q_TERMINAL_COST_MATRIX_GPU,
                                  terminal_state_error)
    # print("state:", state_costs,"\ncontrol", control_costs, "\nterminal", terminal_costs)
    # --- 3. 计算总成本 ---
    total_costs_batch = state_costs + control_costs + terminal_costs
    # 形状: (n_samples,)

    return total_costs_batch

# 网络训练
def train_nn_model(nn_model, optimizer, replay_buffer, batch_size, epochs, min_buffer_size, state_scaler, action_scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(replay_buffer) < min_buffer_size: # 记忆量不够时不训练
        return 0.0
    total_loss = 0
    actual_epochs_run = 0
    for epoch in range(epochs): # 每个step更新次数
        if len(replay_buffer) < batch_size: # Ensure enough samples for a batch
            break
        states, actions, next_states_true = replay_buffer.sample(batch_size)
        states_tensor = torch.FloatTensor(states).to(device)
        actions_tensor = torch.FloatTensor(actions).to(device)
        next_states_true_tensor = torch.FloatTensor(next_states_true).to(device)
        # --- 应用缩放 ---
        scaled_states_tensor = state_scaler.transform(states_tensor)
        scaled_actions_tensor = action_scaler.transform(actions_tensor)
        scaled_next_states_true_tensor = state_scaler.transform(next_states_true_tensor) # 标签也要缩放!
        scaled_next_states_pred_tensor = nn_model(scaled_states_tensor, scaled_actions_tensor)
        loss = nn.MSELoss()(scaled_next_states_pred_tensor, scaled_next_states_true_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        actual_epochs_run +=1
    return total_loss / actual_epochs_run if actual_epochs_run > 0 else 0.0

def save_model_and_buffer(model, replay_buffer, filename):
    # 保存模型
    torch.save(model.state_dict(), filename + "_model.pt")
    # 保存 ReplayBuffer
    with open("replay_buffer.pkl", 'wb') as f:
        pickle.dump(replay_buffer, f)
    
def load_model_and_buffer(model, filename):
    # 加载模型
    model.load_state_dict(torch.load(filename))
    # 加载 ReplayBuffer
    with open("replay_buffer.pkl", 'rb') as f:
        replay_buffer = pickle.load(f)
    return replay_buffer

# Adaptive CEM MPC主算法
def Adaptive_CEM_MPC(
    episode_num, true_system, nn_model, optimizer, replay_buffer,
    waypoint_pass_threshold, # 航路点阈值
    max_sim_time, dt,
    cem_params, nn_train_params, state_scaler, action_scaler, scalers_fitted_once
):
    # 环境初始化
    fpv_state, final_target_state, waypoint, door_z_pos, door_x_pos, door_x_velo, start_time, door_parameter = true_system.reset()

    # GPU并行化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if scalers_fitted_once and episode_num > 0 and episode_num % nn_train_params['scaler refit frequency'] == 0: # 每
        # 从 buffer 获取所有原始数据
        raw_states_np, raw_actions_np, _ = replay_buffer.get_raw_data()
        current_buffer_len = len(raw_states_np)
        # 确定实际用于拟合的样本数量
        actual_fit_size_states = min(current_buffer_len, FIT_SCALER_SUBSET_SIZE)
        if actual_fit_size_states < current_buffer_len:
            indices_fit = np.random.choice(current_buffer_len, actual_fit_size_states, replace=False)
            raw_states_np = raw_states_np[indices_fit]
            raw_actions_np = raw_actions_np[indices_fit]
        raw_states_tensor = torch.tensor(raw_states_np, dtype=torch.float32).to(device)
        state_scaler.fit(raw_states_tensor)
        raw_actions_tensor = torch.tensor(raw_actions_np, dtype=torch.float32).to(device)
        action_scaler.fit(raw_actions_tensor)

    n_steps_this_episode = int(max_sim_time / dt)
    current_true_state = fpv_state.copy()
    mean_control_sequence_for_warm_start = np.zeros((cem_params['prediction_horizon'], ACTION_DIM))
    actual_trajectory = [current_true_state.copy()]
    applied_controls = []
    time_points = [0.0]
    model_losses_this_episode = []
    reached_final_target = False
    steps_taken_in_episode = 0

    # 航路点管理
    def get_index(waypoint, fpv_state, waypoint_pass_threshold):
        y_pos_fpv = fpv_state[1]
        if y_pos_fpv < waypoint[1] + waypoint_pass_threshold:
            index=0
        elif y_pos_fpv > waypoint[1] + waypoint_pass_threshold and y_pos_fpv < waypoint[2] + waypoint_pass_threshold:
            index=1
        elif y_pos_fpv > waypoint[2] + waypoint_pass_threshold:
            index=2
        return index

    def get_target(fpv_state, waypoint_pass_threshold, N_horizon):
        index=get_index(waypoint, fpv_state, waypoint_pass_threshold)
        if index < 2:
            target_sequence=[]
            t = time.time()-start_time
            for i in range(N_horizon):
                new_x = door_parameter["initial_x_pos"][index] + door_parameter["amplitude"] * math.sin(2 * math.pi * door_parameter["frequency"] * (t+dt*i) + door_parameter["deviation"][index])
                door_x_velocity= 2 * math.pi * door_parameter["frequency"] * door_parameter["amplitude"] * math.cos(2 * math.pi * door_parameter["frequency"] * (t+dt*i) + door_parameter["deviation"][index])
                target_sequence.append(np.array([new_x, waypoint[index+1] + waypoint_pass_threshold, door_z_pos[index]-2, door_x_velocity, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            return np.array(target_sequence)
        # 门的坐标中心在基座，门框中心-2m（airsim的z朝下所以减！！！）
        if index == 2:
            return np.tile(final_target_state, (N_horizon, 1))

    current_mpc_target = get_target(current_true_state, waypoint_pass_threshold, cem_params['prediction_horizon'])
    # 将 current_mpc_target 转换为张量，并移至GPU
    # 假设 current_mpc_target 是一个 numpy 数组
    current_mpc_target_gpu = torch.tensor(current_mpc_target, dtype=torch.float32, device=device) 
    print(f"\n--- 开始第 {episode_num + 1} 次训练")

    for step in range(n_steps_this_episode):
        steps_taken_in_episode = step + 1

        if not scalers_fitted_once and len(replay_buffer) >= nn_train_params['batch_size']-1:
            # 从 buffer 获取所有原始数据
            raw_states_np, raw_actions_np, _ = replay_buffer.get_raw_data()
            raw_states_tensor = torch.tensor(raw_states_np, dtype=torch.float32).to(device)
            state_scaler.fit(raw_states_tensor)
            raw_actions_tensor = torch.tensor(raw_actions_np, dtype=torch.float32).to(device)
            action_scaler.fit(raw_actions_tensor)
            scalers_fitted_once = True

        min_cost_step = float('inf') # 无穷大，后面每个都比他小
        use_cem = len(replay_buffer) >= nn_train_params['min_buffer_for_training'] and episode_num>nn_train_params['episode_explore'] # 10epi后才开始训练

        # 将模型移到设备
        nn_model.to(device)
        nn_model.eval() # 通常在推理时使用评估模式
        if not use_cem: # buffer凑够一batch之前随机跑
            # 督战队 主要往前跑
            num_dimensions = ACTION_DIM - 1  # 前几维的数量
            y_move = np.random.uniform(0, ACCEL_MAX, size=1)
            # x-z方向随便跑
            x_move = np.random.uniform(-ACCEL_MAX, ACCEL_MAX, size=1)
            z_move = np.random.uniform(-ACCEL_MAX, ACCEL_MAX, size=1)
            actual_control_to_apply = np.concatenate((x_move, y_move, z_move)) 
        else:
            # 将初始均值和标准差以及目标状态转移到GPU
            cem_iter_mean_gpu = torch.tensor(mean_control_sequence_for_warm_start, dtype=torch.float32, device=device)
            cem_iter_std_gpu = torch.full((cem_params['prediction_horizon'], ACTION_DIM), cem_params['initial_std'], dtype=torch.float32, device=device) # 创建多维数组，填充给定值（方差）

            for cem_iter in range(cem_params['n_iter']): # 每个CEM取样步骤迭代五轮
                # 在GPU上生成扰动和采样控制序列
                perturbations_gpu = torch.normal(mean=0.0, std=1.0,
                                                size=(cem_params['n_samples'], cem_params['prediction_horizon'], ACTION_DIM),
                                                device=device)
                sampled_control_sequences_gpu = cem_iter_mean_gpu.unsqueeze(0) + \
                                                perturbations_gpu * cem_iter_std_gpu.unsqueeze(0)
                
                sampled_control_sequences_gpu = torch.clip(sampled_control_sequences_gpu, -ACCEL_MAX, ACCEL_MAX)
                
                # costs_cem_gpu = torch.zeros(cem_params['n_samples'], device=device) # 移到 cost_function_gpu 内部或外部初始化

                # GPU 并行轨迹前向模拟
                # 初始化当前状态批次 (n_samples, state_dim) .repeat在指定维度重复张量扩维
                current_states_batch_gpu = torch.tensor(current_true_state, dtype=torch.float32, device=device).repeat(cem_params['n_samples'], 1)
                scaled_current_states_batch_gpu = state_scaler.transform(current_states_batch_gpu)
                # 存储所有样本在整个预测时域内的状态轨迹
                # 形状: (n_samples, prediction_horizon + 1, state_dim)
                # +1 是因为包含了初始状态 current_true_state
                scaled_predicted_trajectory_gpu = torch.zeros((cem_params['n_samples'], cem_params['prediction_horizon'] + 1, current_true_state.shape[0]), dtype=torch.float32, device=device)
                scaled_predicted_trajectory_gpu[:, 0, :] = scaled_current_states_batch_gpu

                with torch.no_grad(): # 在推理过程中关闭梯度计算
                    for t_horizon in range(cem_params['prediction_horizon']):
                        # 获取当前时间步的控制输入批次 (n_samples, action_dim)
                        control_inputs_batch_gpu = sampled_control_sequences_gpu[:, t_horizon, :]
                        scaled_control_inputs_batch_gpu = action_scaler.transform(control_inputs_batch_gpu)
                        
                        # 模型预测下一个状态的批次
                        # 假设 nn_model(states, controls) 返回 next_states
                        scaled_next_states_batch_pred_gpu = nn_model(scaled_current_states_batch_gpu, scaled_control_inputs_batch_gpu)
                        
                        scaled_predicted_trajectory_gpu[:, t_horizon + 1, :] = scaled_next_states_batch_pred_gpu
                        scaled_current_states_batch_gpu = scaled_next_states_batch_pred_gpu
                
                # GPU批量计算成本
                scaled_current_mpc_target_gpu = state_scaler.transform(current_mpc_target_gpu)
                costs_cem_gpu = cost_function_gpu(scaled_predicted_trajectory_gpu,
                                                sampled_control_sequences_gpu,
                                                scaled_current_mpc_target_gpu)
                
                min_cost_step_val = torch.min(costs_cem_gpu).item() # .item() 将单个元素的张量转为python标量
                min_cost_step = min(min_cost_step, min_cost_step_val)
                
                # GPU选择精英样本
                elite_indices_gpu = torch.argsort(costs_cem_gpu)[:cem_params['n_elites']]
                elite_sequences_gpu = sampled_control_sequences_gpu[elite_indices_gpu]
                
                # 在GPU上计算新的均值和标准差
                new_mean_gpu = torch.mean(elite_sequences_gpu, dim=0)
                new_std_gpu = torch.std(elite_sequences_gpu, dim=0) # 注意：torch.std 默认计算的是有偏标准差 (n)，如果需要无偏 (n-1)，使用 Bessel's correction (但对于大量样本差异不大)
                
                # 软更新均值和标准差 (在GPU上)
                cem_iter_mean_gpu = cem_params['alpha'] * new_mean_gpu + (1 - cem_params['alpha']) * cem_iter_mean_gpu
                cem_iter_std_gpu = cem_params['alpha'] * new_std_gpu + (1 - cem_params['alpha']) * cem_iter_std_gpu
                cem_iter_std_gpu = torch.maximum(cem_iter_std_gpu, torch.tensor(cem_params['min_std'], dtype=torch.float32, device=device))
                # print(cem_iter_std_gpu)

            # 将最终的优化控制序列从GPU移回CPU，并转换为numpy数组
            optimal_control_sequence = cem_iter_mean_gpu.cpu().numpy()
            # print(optimal_control_sequence)
            actual_control_to_apply = optimal_control_sequence[0, :].copy() # sequece是个二维数组，选第一行就是一个动作
            
            # 更新用于下一次MPC的warm_start序列
            next_mean_control_sequence = np.roll(optimal_control_sequence, -1, axis=0) # 把当前CEM选出的1:end的序列roll作为下一时刻的更新基准(warm start)
            next_mean_control_sequence[-1, :] = optimal_control_sequence[-2, :].copy() # roll之后最后一个动作是a0而不是aN,所以直接用aN-1作为此时的aN
        next_state_fpv, door_x_pos, door_x_velo, collided = true_system.step(actual_control_to_apply)
        # print("状态", next_state_fpv, "\n当前目标", current_mpc_target_gpu, "\n当前动作:", actual_control_to_apply)
        # 阻止卡在原地时的劣质数据污染buffer        
        if np.linalg.norm(current_true_state[0]-0.0)>0.5 and np.linalg.norm(current_true_state[1]-0.0)>0.5 and not collided:
            replay_buffer.push(current_true_state, actual_control_to_apply, next_state_fpv)

        # 检查是否到达最终目标
        pos_dist_to_final = np.linalg.norm(next_state_fpv[:3] - final_target_state[:3])
        vel_dist_to_final = np.linalg.norm(next_state_fpv[3:6] - final_target_state[3:6])
        
        if pos_dist_to_final < POS_TOLERANCE and vel_dist_to_final < VELO_TOLERANCE:
                print(f"第 {episode_num + 1} 次训练: 最终目标已在第 {step} 步到达!")
                reached_final_target = True
                break
        
        # else: # For intermediate gates, "passing" is handled by y-coord check above
        #     if pos_dist_to_current_mpc_target < 0.5 : # Looser check for intermediate gates
        #         print(f"  Ep {episode_num+1}, 步骤 {step}: 到达中间MPC目标 Y={current_mpc_target[1]:.1f}")
        #         # Don't break, let the y-coord check handle switching to next target

        if step == n_steps_this_episode - 1:
            print(f"第 {episode_num + 1} 次训练: 仿真时间到，最终目标状态: {'到达' if reached_final_target else '未到达'}")
            break # End episode if time is up

        if collided:
            print(f"第 {episode_num + 1} 次训练: 发生碰撞")
            if step<10:
                time.sleep(0.5)
            break
        avg_nn_loss = train_nn_model(nn_model, optimizer, replay_buffer,
                                     nn_train_params['batch_size'], nn_train_params['epochs_per_step'],
                                     nn_train_params['min_buffer_for_training'], state_scaler, action_scaler)
        model_losses_this_episode.append(avg_nn_loss if avg_nn_loss > 0 else np.nan)

        applied_controls.append(actual_control_to_apply.copy())
        current_true_state = next_state_fpv.copy()
        current_mpc_target = get_target(current_true_state, waypoint_pass_threshold, cem_params['prediction_horizon'])
        # 将 current_mpc_target 转换为张量，并移至GPU
        # 假设 current_mpc_target 是一个 numpy 数组
        current_mpc_target_gpu = torch.tensor(current_mpc_target, dtype=torch.float32, device=device) 
        actual_trajectory.append(current_true_state.copy())
        time_points.append((step + 1) * dt)

        if step % 20 == 0:
            print(f"Ep {episode_num+1}, step: {step}, MPC代价: {min_cost_step if min_cost_step != float('inf') else -1:.2f}, "
                  f"NN损失: {avg_nn_loss:.4f}, Buffer: {len(replay_buffer)}")
            
    avg_episode_loss = np.nanmean(model_losses_this_episode) if len(model_losses_this_episode) > 0 else 0
    print(f"--- 第 {episode_num + 1} 次训练结束 --- 步数: {steps_taken_in_episode}, "
          f"是否到达最终目标: {reached_final_target}, 平均损失: {avg_episode_loss:.4f}")
    
    # 每20episodes存储一次模型
    if episode_num % 10 == 0 and episode > 1 and step > 10:
        save_model_and_buffer(nn_model, replay_buffer,"master")
        print(f"Model saved.")

    return (np.array(actual_trajectory), np.array(applied_controls),
            np.array(time_points), np.array(model_losses_this_episode),
            reached_final_target, steps_taken_in_episode, avg_episode_loss, scalers_fitted_once)

# --- Main Script ---
if __name__ == "__main__":
    try:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("中文字体 WenQuanYi Micro Hei 未找到，将使用默认字体。")

    true_system = env(DT)
    nn_dynamics_model = DynamicsNN(STATE_DIM, ACTION_DIM, NN_HIDDEN_SIZE)
    
    # 加载模型
    # replay_buffer=load_model_and_buffer(nn_dynamics_model, "master_model.pt")
    
    nn_optimizer = optim.Adam(nn_dynamics_model.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    state_scaler = Scaler(STATE_DIM)
    action_scaler = Scaler(ACTION_DIM)
    state_scaler.to(device)
    action_scaler.to(device)
    scalers_fitted_once = False
    
    cem_parameters = {
        'prediction_horizon': PREDICTION_HORIZON,
        'n_samples': N_SAMPLES_CEM,
        'n_elites': N_ELITES_CEM,
        'n_iter': N_ITER_CEM,
        'initial_std': INITIAL_STD_CEM,
        'min_std': MIN_STD_CEM,
        'alpha': ALPHA_CEM
    }
    nn_training_parameters = {
        'batch_size': BATCH_SIZE,
        'epochs_per_step': NN_TRAIN_EPOCHS_PER_STEP,
        'min_buffer_for_training': MIN_BUFFER_FOR_TRAINING,
        'episode_explore':EPISODE_EXPLORE,
        'scaler refit frequency': SCALER_REFIT_FREQUENCY
    }

    episode_steps_taken = []
    episode_reached_target_flags = []
    episode_avg_losses = []
    last_episode_data = None

    for episode in range(NUM_EPISODES):
        trajectory, controls, times, nn_losses_ep, \
        reached_target_ep, steps_taken_ep, avg_loss_ep, scalers_fitted_once = Adaptive_CEM_MPC(
            episode_num=episode,
            true_system=true_system,
            nn_model=nn_dynamics_model,
            optimizer=nn_optimizer,
            replay_buffer=replay_buffer,
            waypoint_pass_threshold=WAYPOINT_PASS_THRESHOLD_Y,
            max_sim_time=MAX_SIM_TIME_PER_EPISODE,
            dt=DT,
            cem_params=cem_parameters,
            nn_train_params=nn_training_parameters,
            state_scaler=state_scaler,
            action_scaler=action_scaler,
            scalers_fitted_once=scalers_fitted_once
        )
        
        episode_steps_taken.append(steps_taken_ep)
        episode_reached_target_flags.append(1 if reached_target_ep else 0)
        episode_avg_losses.append(avg_loss_ep)

        if episode == NUM_EPISODES - 1:
            last_episode_data = {
                "trajectory": trajectory, "controls": controls,
                "times": times, "nn_losses": nn_losses_ep
            }

    # # 绘图
    # # 最后一条轨迹详细图像
    # if last_episode_data:
    #     fig_last_ep, axs_last_ep = plt.subplots(2, 3, figsize=(22, 12))
    #     fig_last_ep.suptitle(f'自适应CEM-MPC - 第 {NUM_EPISODES} 次训练结果', fontsize=16)
        
    #     traj_data = last_episode_data["trajectory"]
    #     axs_last_ep[0, 0].plot(traj_data[:, 0], traj_data[:, 1], marker='.', linestyle='-', label='实际轨迹')
    #     axs_last_ep[0, 0].plot(START_POS[0], START_POS[1], 'go', markersize=10, label='起点')
    #     # Plot all waypoints
    #     for i in range(len(WAYPOINTS_Y)):
    #         axs_last_ep[0, 0].plot(WAYPOINTS_X[i], WAYPOINTS_Y[i], 'bx', markersize=8, mew=2, label=f'航点 {i+1} (Y={WAYPOINTS_Y[i]})')
    #     axs_last_ep[0, 0].plot(ULTIMATE_TARGET_POS[0], ULTIMATE_TARGET_POS[1], 'rx', markersize=12, mew=3, label='最终目标')
    #     axs_last_ep[0, 0].set_xlabel('X 位置'); axs_last_ep[0, 0].set_ylabel('Y 位置'); axs_last_ep[0, 0].set_title('运动轨迹')
    #     axs_last_ep[0, 0].legend(); axs_last_ep[0, 0].grid(True); axs_last_ep[0, 0].axis('equal')

    #     axs_last_ep[0, 1].plot(last_episode_data["times"], traj_data[:, 0], label='X 位置')
    #     axs_last_ep[0, 1].plot(last_episode_data["times"], traj_data[:, 1], label='Y 位置')
    #     for i in range(len(WAYPOINTS_Y)):
    #          axs_last_ep[0, 1].axhline(WAYPOINTS_Y[i], color='gray', linestyle=':', label=f'航点 Y={WAYPOINTS_Y[i]}')
    #     axs_last_ep[0, 1].axhline(ULTIMATE_TARGET_POS[1], color='r', linestyle='--', label=f'最终目标 Y')
    #     axs_last_ep[0, 1].set_xlabel('时间 (s)'); axs_last_ep[0, 1].set_ylabel('位置'); axs_last_ep[0, 1].set_title('位置 vs. 时间')
    #     axs_last_ep[0, 1].legend(); axs_last_ep[0, 1].grid(True)

    #     axs_last_ep[1, 0].plot(last_episode_data["times"], traj_data[:, 2], label='X 速度')
    #     axs_last_ep[1, 0].plot(last_episode_data["times"], traj_data[:, 3], label='Y 速度')
    #     axs_last_ep[1, 0].axhline(ULTIMATE_TARGET_VEL[0], color='r', linestyle='--', label='目标 X Vel'); axs_last_ep[1, 0].axhline(ULTIMATE_TARGET_VEL[1], color='g', linestyle='--', label='目标 Y Vel')
    #     axs_last_ep[1, 0].set_xlabel('时间 (s)'); axs_last_ep[1, 0].set_ylabel('速度'); axs_last_ep[1, 0].set_title('速度 vs. 时间')
    #     axs_last_ep[1, 0].legend(); axs_last_ep[1, 0].grid(True)

    #     control_times_last_ep = last_episode_data["times"][:len(last_episode_data["controls"])]
    #     if len(last_episode_data["controls"]) > 0:
    #         axs_last_ep[1, 1].plot(control_times_last_ep, last_episode_data["controls"][:, 0], label='X 加速度')
    #         axs_last_ep[1, 1].plot(control_times_last_ep, last_episode_data["controls"][:, 1], label='Y 加速度')
    #     axs_last_ep[1, 1].set_xlabel('时间 (s)'); axs_last_ep[1, 1].set_ylabel('控制输入'); axs_last_ep[1, 1].set_title('控制输入 vs. 时间')
    #     axs_last_ep[1, 1].legend(); axs_last_ep[1, 1].grid(True); axs_last_ep[1, 1].set_ylim([-ACCEL_MAX*1.1, ACCEL_MAX*1.1])

    #     nn_losses_plot = last_episode_data["nn_losses"]
    #     if len(nn_losses_plot) > 0:
    #         valid_loss_indices = ~np.isnan(nn_losses_plot)
    #         times_for_losses = last_episode_data["times"][1:len(nn_losses_plot)+1] # Corrected indexing
    #         if np.any(valid_loss_indices):
    #              axs_last_ep[0, 2].plot(times_for_losses[valid_loss_indices], nn_losses_plot[valid_loss_indices], label='NN 模型损失 (MSE)')
    #         else:
    #              axs_last_ep[0, 2].text(0.5,0.5, '无有效损失', transform=axs_last_ep[0,2].transAxes, ha='center', va='center')
    #     else:
    #         axs_last_ep[0, 2].text(0.5,0.5, '无损失数据', transform=axs_last_ep[0,2].transAxes, ha='center', va='center')
    #     axs_last_ep[0, 2].set_xlabel('时间 (s)'); axs_last_ep[0, 2].set_ylabel('平均 MSE 损失'); axs_last_ep[0, 2].set_title('学习的动力学模型损失（上次训练）')
    #     axs_last_ep[0, 2].legend(); axs_last_ep[0, 2].grid(True); axs_last_ep[0, 2].set_yscale('log')
        
    #     axs_last_ep[1, 2].axis('off')
    #     plt.tight_layout(rect=[0, 0, 1, 0.96])

    # # Plot 2: Learning progress across episodes (remains the same)
    # fig_progress, axs_progress = plt.subplots(1, 2, figsize=(15, 6))
    # fig_progress.suptitle('跨训练周期的学习进度 (多航点)', fontsize=16)
    # axs_progress[0].plot(range(1, NUM_EPISODES + 1), episode_steps_taken, marker='o', linestyle='-')
    # axs_progress[0].set_xlabel('训练周期编号'); axs_progress[0].set_ylabel('到达最终目标所需步数'); axs_progress[0].set_title('每轮训练到达目标的步数'); axs_progress[0].grid(True)
    # ax2_steps = axs_progress[0].twinx()
    # ax2_steps.plot(range(1, NUM_EPISODES + 1), episode_reached_target_flags, marker='x', linestyle='--', color='r', label='是否到达目标 (1=是)')
    # ax2_steps.set_ylabel('是否到达最终目标 (1=是)', color='r'); ax2_steps.tick_params(axis='y', labelcolor='r'); ax2_steps.set_ylim([-0.1, 1.1])
    # axs_progress[1].plot(range(1, NUM_EPISODES + 1), episode_avg_losses, marker='o', linestyle='-')
    # axs_progress[1].set_xlabel('训练周期编号'); axs_progress[1].set_ylabel('平均 NN 模型损失 (MSE)'); axs_progress[1].set_title('每轮训练的平均模型损失'); axs_progress[1].grid(True); axs_progress[1].set_yscale('log')
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.show()