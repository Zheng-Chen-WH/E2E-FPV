import random
import numpy as np
import os
import pickle
import torch

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, pi_img, q_state, action, reward, next_pi_img, 
             next_q_state, done, goal,
              resnet_position, resnet_attitude, gru_velocity, gru_angular):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None) #没有充满时先用none创造出self.position
        self.buffer[int(self.position)] = (pi_img.squeeze(0), q_state, action, reward, next_pi_img.squeeze(0), next_q_state, done, goal,
                                      resnet_position, resnet_attitude, gru_velocity, gru_angular)
        # 把两个视频张量的多余维度挤掉，输出stack的时候才能叠回正常batchsize
        self.position = (self.position + 1) % self.capacity #超出记忆池容量后从第一个开始改写以维持容量不超标

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        pi_img, q_state, action, reward, next_pi_img,\
            next_q_state, done, goal, \
            resnet_position, resnet_attitude, gru_velocity, gru_angular = map(np.stack, zip(*batch))
        return pi_img, q_state, action, reward, next_pi_img,\
            next_q_state, done, goal, \
            resnet_position, resnet_attitude, gru_velocity, gru_angular
        '''zip()函数用于将多个可迭代对象（例如列表、元组等）的对应元素打包成一个元组，返回一个迭代器。
        map()函数将一个函数应用于迭代器中的每个元素，并返回一个新的迭代器。
        这里将np.stack()函数应用于zip(*batch)的每个元素。np.stack()函数用于沿着新的轴堆叠数组序列，返回一个新的数组。
        map(np.stack, zip(*batch))将返回一个包含多个新数组的迭代器，其中每个新数组由批次中对应元素的堆叠组成。
        最后将上一步得到的迭代器中的新数组分别赋值给state、action、reward、next_state和done这五个变量。这意味着每个变量都是一个包含了批次中对应元素的堆叠数组。
        【迭代器】：允许按需逐个访问集合中的元素，而不是一次性获取整个集合；range()函数、zip()函数和字典的items()方法都返回迭代器，
        还可以使用关键字yield来定义生成器函数，生成器函数返回的对象也是迭代器'''

    def __len__(self):
        return len(self.buffer)
        # 双下划线（__）用于表示特殊方法或特殊属性。这些特殊方法和属性具有预定义的名称，它们在对象上具有特殊的行为。
        # __len__() 是一个特殊方法，用于返回对象的长度

    def save_buffer(self, file_name, save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = f"checkpoints/sac_buffer_{file_name}"
            
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
    
    def clear(self):
        self.buffer = []

class DAggerMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, pi_img, action, goal,
              resnet_position, resnet_attitude, gru_velocity, gru_angular):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None) #没有充满时先用none创造出self.position
        self.buffer[self.position] = (pi_img.squeeze(0), action, goal,
                                      resnet_position, resnet_attitude, gru_velocity, gru_angular)
        # 把两个视频张量的多余维度挤掉，输出stack的时候才能叠回正常batchsize
        self.position = (self.position + 1) % self.capacity #超出记忆池容量后从第一个开始改写以维持容量不超标

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        pi_img, action, goal, \
            resnet_position, resnet_attitude, gru_velocity, gru_angular = map(np.stack, zip(*batch))
        return pi_img, action, goal, \
            resnet_position, resnet_attitude, gru_velocity, gru_angular

    def __len__(self):
        return len(self.buffer)
        # 双下划线（__）用于表示特殊方法或特殊属性。这些特殊方法和属性具有预定义的名称，它们在对象上具有特殊的行为。
        # __len__() 是一个特殊方法，用于返回对象的长度

    def save_buffer(self, file_name, save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = f"checkpoints/sac_buffer_{file_name}"
            
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
    
    def clear(self):
        self.buffer = []

class RolloutBuffer:
    def __init__(self, n_steps, state_dims, action_dim, gae_lambda=0.95, gamma=0.99):
        self.n_steps = n_steps
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        
        # 定义存储空间
        self.observations = {'pi_img': np.zeros((n_steps, *state_dims['pi_img']), dtype=np.float32),
                             'goal': np.zeros((n_steps, *state_dims['goal']), dtype=np.float32)}
        self.actions = np.zeros((n_steps, action_dim), dtype=np.float32)
        self.rewards = np.zeros((n_steps,), dtype=np.float32)
        self.dones = np.zeros((n_steps,), dtype=np.bool_)
        self.log_probs = np.zeros((n_steps,), dtype=np.float32)
        self.values = np.zeros((n_steps,), dtype=np.float32)
        
        # GAE计算所需
        self.advantages = np.zeros((n_steps,), dtype=np.float32)
        self.returns = np.zeros((n_steps,), dtype=np.float32)
        
        self.ptr = 0
        self.full = False

    def add(self, obs, action, reward, done, value, log_prob):
        for key, val in obs.items():
            self.observations[key][self.ptr] = val
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value, last_done):
        """
        在rollout结束后，从后往前计算GAE和Returns
        """
        last_advantage = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            self.advantages[t] = last_advantage
        
        self.returns = self.advantages + self.values

    def get(self, batch_size, device):
        """
        创建一个生成器，用于在PPO更新期间按mini-batch产出数据
        """
        # 确保所有数据都在一个大的numpy数组中
        indices = np.random.permutation(self.n_steps)
        
        # 展平数据
        obs_flat = {key: val.reshape(-1, *val.shape[1:]) for key, val in self.observations.items()}
        actions_flat = self.actions.reshape(-1, self.actions.shape[-1])
        
        for start in range(0, self.n_steps, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            # 将numpy数据转为torch tensor
            batch_obs = {key: torch.from_numpy(obs_flat[key][batch_indices]).to(device) for key in obs_flat}
            batch_actions = torch.from_numpy(actions_flat[batch_indices]).to(device)
            batch_log_probs = torch.from_numpy(self.log_probs[batch_indices]).to(device)
            batch_advantages = torch.from_numpy(self.advantages[batch_indices]).to(device)
            batch_returns = torch.from_numpy(self.returns[batch_indices]).to(device)
            
            yield batch_obs, batch_actions, batch_log_probs, batch_advantages, batch_returns