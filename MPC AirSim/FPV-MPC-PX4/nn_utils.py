import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
import pickle

class Scaler:
    def __init__(self, dim):
        self.dim = dim
        self.mean = torch.zeros(dim, dtype=torch.float32)
        self.std = torch.ones(dim, dtype=torch.float32)
        self.fitted = False
        self._device = torch.device("cpu") # 单纯初始化，使用时用to塞到gpu上

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

        self.mean = torch.mean(data_tensor, dim=0).to(self._device) # 确保方差和均值在目标设备上
        self.std = torch.std(data_tensor, dim=0).to(self._device)
        # 防止标准差为0 (如果某个特征在所有样本中都一样)
        self.std = torch.where(self.std < 1e-7, torch.ones_like(self.std) * 1e-7, self.std)
        self.fitted = True
        print(f"Scaler fitted. Mean: {self.mean.cpu().numpy()}, Std: {self.std.cpu().numpy()}")

    def transform(self, data_tensor): # 对数据归一化
        if not self.fitted:
            # print("Warning: Scaler not fitted yet. Returning original data.")
            return data_tensor
        return (data_tensor.to(self._device) - self.mean) / self.std

    def inverse_transform(self, data_tensor_scaled): # 对数据逆归一化
        if not self.fitted:
            return data_tensor_scaled
        return data_tensor_scaled.to(self._device) * self.std + self.mean

    def to(self, device):
        """把scaler参数转移到指定设备"""
        self._device = device
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

# 动力学模型网络
class DynamicsNN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(DynamicsNN, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, state_dim) # Predicts delta_state or next_state

    def forward(self, state, action):
        # 目前直接预测状态，但是预测残差似乎也很可行
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        next_state_pred = self.fc3(x)
        # next_state_pred = state + delta_pred # 如果改成预测残差的话取消注释这行
        return next_state_pred

# ReplayBuffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, next_state):
        self.buffer.append((state, action, next_state))

    def sample(self, batch_size):
        # 确保有足够多样本可以取出
        actual_batch_size = min(batch_size, len(self.buffer))
        if actual_batch_size == 0:
            return None, None, None

        state, action, next_state = zip(*random.sample(self.buffer, actual_batch_size))
        return np.array(state), np.array(action), np.array(next_state)

    def get_raw_data(self): # 返回所有存储的原始数据
        if not self.buffer:
            return np.array([]), np.array([]), np.array([])
        raw_states, raw_actions, raw_next_states = zip(*list(self.buffer))
        return np.array(raw_states), np.array(raw_actions), np.array(raw_next_states)

    def __len__(self):
        return len(self.buffer)

# 网络训练
def train_nn_model(nn_model, optimizer, replay_buffer, batch_size, epochs,
                   min_buffer_size_for_training, state_scaler, action_scaler, device):
    if len(replay_buffer) < min_buffer_size_for_training: # 记忆量不够时不训练
        return 0.0, 0 # Loss, epochs_run

    total_loss = 0.0
    actual_epochs_run = 0
    nn_model.train() # 模型设为训练模式

    for epoch in range(epochs): # 每个step更新次数
        if len(replay_buffer) < batch_size: # 确保buffer数据足够多
            break

        states_np, actions_np, next_states_true_np = replay_buffer.sample(batch_size)
        if states_np is None: # Not enough samples for even one batch
            break

        states_tensor = torch.FloatTensor(states_np).to(device)
        actions_tensor = torch.FloatTensor(actions_np).to(device)
        next_states_true_tensor = torch.FloatTensor(next_states_true_np).to(device)

        # 应用缩放
        scaled_states_tensor = state_scaler.transform(states_tensor)
        scaled_actions_tensor = action_scaler.transform(actions_tensor)
        # 用来生成数据的标签也要缩放! (Target for the NN is also a state)
        scaled_next_states_true_tensor = state_scaler.transform(next_states_true_tensor)

        scaled_next_states_pred_tensor = nn_model(scaled_states_tensor, scaled_actions_tensor)

        loss = nn.MSELoss()(scaled_next_states_pred_tensor, scaled_next_states_true_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        actual_epochs_run += 1

    nn_model.eval() # 训练后模型复位为eval模式
    return total_loss / actual_epochs_run if actual_epochs_run > 0 else 0.0, actual_epochs_run

def save_model_and_buffer(model, replay_buffer, state_scaler, action_scaler, filename="master"):
    # 保存模型状态字典
    torch.save(model.state_dict(), filename + "_model.pt")

    # 保存 ReplayBuffer
    with open(filename + "_replay_buffer.pkl", 'wb') as f:
        pickle.dump(replay_buffer, f)

    # 保存 Scalers
    with open(filename + "_scalers.pkl", 'wb') as f:
        pickle.dump({'state_scaler': state_scaler, 'action_scaler': action_scaler}, f)

    print("Model, Replay Buffer, and Scalers saved")

def load_model_and_buffer(model, base_filename="master", device=torch.device("cpu")):
    # 加载模型状态字典
    model.load_state_dict(torch.load(base_filename + "_model.pt", map_location=device))
    model.to(device) # 确保模型在正确设备上
    model.eval() # 加载后设为测试模式
    print(f"Dynamics model loaded from {base_filename}_model.pt")

    # 加载 ReplayBuffer
    try:
        with open(base_filename + "_replay_buffer.pkl", 'rb') as f:
            replay_buffer = pickle.load(f)
        print(f"Replay buffer loaded from {base_filename}_replay_buffer.pkl")
    except FileNotFoundError:
        print(f"Warning: Replay buffer file {base_filename}_replay_buffer.pkl not found. Returning None.")
        replay_buffer = None

    # 加载 Scalers
    try:
        with open(base_filename + "_scalers.pkl", 'rb') as f:
            scalers_data = pickle.load(f)
            state_scaler = scalers_data['state_scaler']
            action_scaler = scalers_data['action_scaler']
            state_scaler.to(device)
            action_scaler.to(device)
        print(f"Scalers loaded from {base_filename}_scalers.pkl")
    except FileNotFoundError:
        print(f"Warning: Scalers file {base_filename}_scalers.pkl not found. Returning None for scalers.")
        state_scaler, action_scaler = None, None
        
    return replay_buffer, state_scaler, action_scaler