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
        self.position = 0

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
        self.position = 0
class RolloutBuffer:
    """
    为On-Policy算法（如PPO）设计的Rollout Buffer。
    它收集一个固定长度的轨迹，然后计算优势和回报，
    并为训练提供小批次数据的生成器。

    :param buffer_size: Buffer可以存储的最大转换数 (n_steps)
    :param args: 包含状态、动作等维度的参数字典
    :param gae_lambda: GAE算法中的lambda因子
    :param gamma: 折扣因子
    :param device: 计算设备 (cpu or cuda)
    """
    def __init__(self, buffer_size: int, args: dict, gae_lambda: float = 0.95, gamma: float = 0.99, device: str = "cpu"):
        self.buffer_size = buffer_size
        self.args = args
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.device = device

        # 获取维度信息
        pi_img_shape = (args['pi_img_seq_len'], 3, args['pi_img_size'], args['pi_img_size'])
        state_dim = args['Pi_mlp_dim']
        priv_state_dim = args['privileged_dim']
        action_dim = args['action_dim']
        aux_dims = {
            'pos': 3, 'att': 9, 'vel': 3, 'ang': 3
        }

        # 预先分配内存，比list.append()高效得多
        self.pi_imgs = np.zeros((self.buffer_size,) + pi_img_shape, dtype=np.float32)
        self.states = np.zeros((self.buffer_size, state_dim), dtype=np.float32)
        self.privileged_states = np.zeros((self.buffer_size, priv_state_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size,), dtype=np.float32)
        self.values = np.zeros((self.buffer_size,), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size,), dtype=np.float32)
        
        # 存储辅助任务的真值
        self.gt_pos = np.zeros((self.buffer_size, aux_dims['pos']), dtype=np.float32)
        self.gt_att = np.zeros((self.buffer_size, aux_dims['att']), dtype=np.float32)
        self.gt_vel = np.zeros((self.buffer_size, aux_dims['vel']), dtype=np.float32)
        self.gt_ang = np.zeros((self.buffer_size, aux_dims['ang']), dtype=np.float32)

        # GAE计算结果
        self.advantages = np.zeros((self.buffer_size,), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size,), dtype=np.float32)
        
        self.pos = 0
        self.full = False

    def clear(self):
        """清空Buffer，准备下一次rollout"""
        self.pos = 0
        self.full = False

    def push(self, pi_img, state, privileged_state, action, reward, done, value, log_prob,
             gt_pos, gt_att, gt_vel, gt_ang):
        """
        向Buffer中添加一个时间步的数据。

        :param pi_img: 策略网络使用的图像序列 (Actor input)
        :param state: 策略网络使用的常规状态向量 (Actor input)
        :param privileged_state: 价值网络使用的物理真值向量 (Critic input)
        :param action: 执行的动作
        :param reward: 收到的奖励
        :param done: 结束标志
        :param value: Critic对当前状态的价值估计 V(s)
        :param log_prob: Actor执行该动作的对数概率
        :param gt_...: 辅助任务的真值
        """
        # 将数据存入预分配的numpy数组
        self.pi_imgs[self.pos] = pi_img
        self.states[self.pos] = state
        self.privileged_states[self.pos] = privileged_state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.gt_pos[self.pos] = gt_pos
        self.gt_att[self.pos] = gt_att
        self.gt_vel[self.pos] = gt_vel
        self.gt_ang[self.pos] = gt_ang
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(self, last_value: float, last_done: float):
        """
        在Rollout结束后，计算每个时间步的Return和Advantage。
        这个函数必须在调用get()之前被调用。

        :param last_value: 最后一个状态的价值估计 V(s_T)。用于GAE的自举(bootstrap)。
        :param last_done: 最后一个状态的done标志。
        """
        last_gae_lam = 0
        # 从后往前计算
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]
            
            # 广义优势估计 (GAE)
            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        
        # 价值目标 = 优势 + 状态价值
        self.returns = self.advantages + self.values

    def get(self, batch_size: int):
        """
        返回一个生成器，用于在PPO的优化循环中产生小批次数据。
        
        :param batch_size: 小批次的大小
        :return: 一个包含所有数据的字典的生成器
        """
        if not self.full:
            raise ValueError("Buffer not full. Call 'compute_returns_and_advantage' before 'get'.")
            
        # 优势标准化 (非常重要的PPO trick)
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

        # 创建随机索引，用于打乱数据
        indices = np.random.permutation(self.buffer_size)

        start_idx = 0
        while start_idx < self.buffer_size:
            batch_indices = indices[start_idx : start_idx + batch_size]
            
            # 提取一个小批次的数据并转换为torch.Tensor
            yield {
                'pi_imgs': torch.from_numpy(self.pi_imgs[batch_indices]).to(self.device),
                'states': torch.from_numpy(self.states[batch_indices]).to(self.device),
                'privileged_states': torch.from_numpy(self.privileged_states[batch_indices]).to(self.device),
                'actions': torch.from_numpy(self.actions[batch_indices]).to(self.device),
                'values': torch.from_numpy(self.values[batch_indices]).to(self.device),
                'log_probs': torch.from_numpy(self.log_probs[batch_indices]).to(self.device),
                'advantages': torch.from_numpy(self.advantages[batch_indices]).to(self.device),
                'returns': torch.from_numpy(self.returns[batch_indices]).to(self.device),
                'gt_pos': torch.from_numpy(self.gt_pos[batch_indices]).to(self.device),
                'gt_att': torch.from_numpy(self.gt_att[batch_indices]).to(self.device),
                'gt_vel': torch.from_numpy(self.gt_vel[batch_indices]).to(self.device),
                'gt_ang': torch.from_numpy(self.gt_ang[batch_indices]).to(self.device),
            }
            start_idx += batch_size

    def __len__(self):
        return self.buffer_size if self.full else self.pos
