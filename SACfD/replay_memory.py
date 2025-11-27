import random
import numpy as np
import os
import pickle
from collections import deque
from torch.utils.data import BatchSampler, SubsetRandomSampler
import torch

class ReplayMemory:
    def __init__(self, args):
        """
        通过配置字典预先定义所有buffer和它们的容量
        buffer_configs: e.g., {'expert': 10000, 'agent': 500000}
        """
        self.buffer_configs = args['buffer_configs']
        # 使用deque可以高效地实现固定大小的队列
        self.recent_size = args['recent_size']
        self.buffers = {name: deque(maxlen=capacity) for name, capacity in self.buffer_configs.items()}

    def push(self, source_name, transition):
        """
        将一个transition存到指定名称的buffer中。
        如果source_name未在初始化时定义，则会报错。
        """
        self.buffers[source_name].append(transition)

    def sample(self, batch_config):
        """
        根据指定的配置从不同的buffer中采样数据，并返回带有来源标志的batch。
        batch_config: e.g., {'expert': 32, 'agent': 64}
        """
        # 用于收集所有采样出的数据
        all_augmented_transitions = []

        # 遍历采样指令
        for source_name, batch_size in batch_config.items():
            sampling_zone = None
            original_source_name = source_name

            # 关键：检测特殊的采样需求
            if source_name.endswith('_recent'):
                sampling_zone = 'recent'
                original_source_name = source_name.removesuffix('_recent')
            elif source_name.endswith('_old'):
                sampling_zone = 'old'
                original_source_name = source_name.removesuffix('_old')
            
            if original_source_name not in self.buffers or not self.buffers[original_source_name]:
                continue

            buffer = self.buffers[original_source_name]

            # 根据区域定义采样池
            sampling_pool = list(buffer) # 将deque转为list以进行切片
            
            if sampling_zone == 'recent':
                # 最后N个元素构成“最近”池
                pool_to_sample = sampling_pool[-self.recent_size:]
            elif sampling_zone == 'old':
                # 除了最后N个元素，其他所有元素构成“存档”池
                pool_to_sample = sampling_pool[:-self.recent_size]
            else:
                # 默认行为：从整个buffer采样
                pool_to_sample = sampling_pool
            
            if not pool_to_sample:
                continue

            if batch_size > len(pool_to_sample):
                # print(f"{source_name}中数据实际容量为{len(pool_to_sample)}<{batch_size}，不进行训练")
                return None
            
            # 从buffer中随机采样
            sampled_transitions = random.sample(pool_to_sample, batch_size)
            
            # 为每条数据增加来源标志，并存入总列表
            for t in sampled_transitions:
                '''关键步骤
                假设 expert 采样结果为 [e2, e1]
                (*t, source_name) 的意思是：
                - *t: 将元组 t (例如 e2) 的所有元素解包出来。
                - , source_name: 在解包后的元素末尾，追加当前的来源名称（类似append）。
                - (...): 将所有这些元素重新组合成一个【新】的、更长的元组。
                执行完这个循环后，all_augmented_transitions 的内容会是：
                (假设 e1 = (s_e1, a_e1, ...) and a1 = (s_a1, a_a1, ...) )
                [
                (s_e2, a_e2, ..., 'expert'),  # 来自 expert buffer
                (s_e1, a_e1, ..., 'expert'),  # 来自 expert buffer
                (s_a4, a_a4, ..., 'agent'),   # 来自 agent buffer
                (s_a1, a_a1, ..., 'agent'),   # 来自 agent buffer
                (s_a3, a_a3, ..., 'agent')    # 来自 agent buffer
                ]'''
                all_augmented_transitions.append((*t, source_name))

        # 为了训练的稳定性，将来自不同buffer的数据混合在一起
        random.shuffle(all_augmented_transitions)

        # 解压batch，现在最后一项是数据来源的标志
        """*all_augmented_transitions：
            这会把列表 all_augmented_transitions “解包”成独立的参数。就好像这样调用 zip:
            zip( (s_a1, ...), (s_e2, ...), (s_a4, ...), (s_e1, ...), (s_a3, ...) )
            zip(...)：接着，zip会像拉链一样，把每个元组相同位置的元素聚合在一起。
            它会取出所有元组的第0个元素 (s_a1, s_e2, s_a4, s_e1, s_a3)，组成一个新的元组。
            然后取出所有元组的第1个元素 (a_a1, a_e2, a_a4, a_e1, a_a3)，组成一个新的元组。
            ...以此类推...
            最后取出所有元组的最后一个元素 ('agent', 'expert', 'agent', 'expert', 'agent')，组成一个新的元组。
            unzipped_batch 的结果会是一个列表，其内容如下：
            [
            (s_a1, s_e2, s_a4, s_e1, s_a3),      # <-- 所有 state 组成一个元组
            (a_a1, a_e2, a_a4, a_e1, a_a3),      # <-- 所有 action 组成一个元组
            ...                                  # <-- 所有 reward, next_state, done...
            ('agent', 'expert', 'agent', 'expert', 'agent') # <-- 所有 source_name 组成一个元组
            ]"""
        unzipped_batch = list(zip(*all_augmented_transitions))
        
        # 将每个部分转换为numpy array
        """final_batch 的最终形态是：
            [
            np.array([...states...]),      # 形状为 (N, state_dim) 的Numpy数组
            np.array([...actions...]),     # 形状为 (N, action_dim) 的Numpy数组
            ...
            np.array(['agent', ...])       # 形状为 (N,) 的Numpy数组，包含了来源标志
            ]"""
        final_batch = [np.array(part) for part in unzipped_batch]

        return final_batch

    def __len__(self, source_name=None):
        """
        返回指定buffer的长度；如果source_name为None，则返回所有buffer的总长度
        """
        if source_name:
            if source_name not in self.buffers:
                return 0
            return len(self.buffers[source_name])
        
        return sum(len(buffer) for buffer in self.buffers.values())

class RolloutBuffer:
    """
    Recurrent PPO 使用的 Rollout Buffer。
    
    核心设计原则：
    1. 保持 episode 内的时序连续性，不随机打散
    2. 按 episode 存储数据，记录每个 episode 的边界
    3. 采样时按序列块 (sequence chunks) 采样
    4. 在序列块开头重新初始化 hidden state (设为 zero 或使用第一步存储的)
    
    这样可以正确处理 GRU/LSTM 等循环网络的时序依赖。
    """
    def __init__(self, args):
        self.device = args['device']
        self.gamma = args['gamma']
        self.gae_lambda = args['gae_lambda']
        # 序列块长度：每个 mini-batch 中的连续时间步数
        self.seq_len = args['seq_len']  # 一个序列块内的步数
        self.reset()

    def reset(self):
        self.img_seqs = []
        self.V_states = []   # Critic (Value Network) 的输入状态
        self.pi_states = []  # Policy (Actor) 的输入状态
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.hidden_states = []  # 存储每步开始时的 hidden state（用于序列块开头）
        # 辅助任务标签
        self.aux_pos = []
        self.aux_rot = []
        self.aux_vel = []
        self.aux_ang_vel = []
        # GAE 计算结果
        self.returns = []
        self.advantages = []
        # 记录当前已处理的episode截止位置
        self.last_episode_end = 0
        # 记录每个 episode 的边界索引 [(start, end), ...]
        self.episode_boundaries = []

    def push(self, data):
        # 解包数据: (img_seq, V_state, pi_state, action, reward, done, log_prob, value, 
        #            pos, rot, vel, ang, hidden_state)
        img_seq, V_state, pi_state, action, reward, done, \
            log_prob, value, pos, rot, vel, ang, hidden_state = data
        self.img_seqs.append(img_seq)
        self.V_states.append(V_state)  # Critic的输入
        self.pi_states.append(pi_state)  # Policy的输入
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.aux_pos.append(pos)
        self.aux_rot.append(rot)
        self.aux_vel.append(vel)
        self.aux_ang_vel.append(ang)
        self.hidden_states.append(hidden_state)

    def get_data_loader(self, mini_batch_size):
        """
        Recurrent PPO 的数据加载器：按序列块 (sequence chunks) 采样
        
        关键改变:
        1. 不打散时间步，保持 episode 内时序连续
        2. 将每个 episode 划分为长度为 seq_len 的连续序列块
        3. 随机打乱序列块的顺序（而非单个 step）
        4. 每个 mini-batch 包含多个序列块
        
        Args:
            mini_batch_size: 每个 mini-batch 包含的【序列块数量】（不是 step 数量）
        
        Yields:
            batch: 包含形状为 (batch_size, seq_len, ...) 的数据
        """
        # 预处理：构建 Chunk 索引列表
        # chunk_indices: 存储每个 chunk 包含的 step 索引 [ [0,1,2...], [8,9,10...] ]
        # chunk_starts: 存储每个 chunk 的起始索引，用于提取 hidden state
        # 收集所有序列块（chunk）的首尾索引（例：55，99的一个episode）
        chunk_indices_list = []
        chunk_masks_list = []
        chunk_start_indices = []
        for ep_start, ep_end in self.episode_boundaries:
            ep_len = ep_end - ep_start
            # 将这个 episode 划分为多个 seq_len 长度的块
            # 例：0，8，16，...40 (假设 seq_len=8)
            for chunk_start in range(0, ep_len, self.seq_len):
                # 例：(55+0=55), (55+8=63), (55+16=71), ... (55+40=95)
                actual_start = ep_start + chunk_start
                # 例：(55+8=63), (55+16=71), ... (55+48=103【取min】=99)
                actual_end = min(actual_start + self.seq_len, ep_end)
                # 例：本次的chunk_start对应的chunk列表[55,56,57,58,59,60,61,62]
                indices = list(range(actual_start, actual_end))
                current_len = len(indices)
                # Padding 索引：如果长度不足 seq_len，补 0 (或其他安全索引)
                # 注意：补 0 会取到第 0 个数据，但后面会被 mask 掉，所以没关系
                if current_len < self.seq_len:
                    indices += [0] * (self.seq_len - current_len)
                    mask = [1.0] * current_len + [0.0] * (self.seq_len - current_len)
                else:
                    mask = [1.0] * self.seq_len
                
                chunk_indices_list.append(indices)
                chunk_masks_list.append(mask)
                chunk_start_indices.append(actual_start)
        
        if len(chunk_indices_list) == 0:
            return
        
        # 各种索引转换为 Tensor 索引矩阵 (Num_Chunks, Seq_Len)
        # 这一步在 CPU 上做通常更快，最后再转 GPU
        # 在 PyTorch中凡是用来做“索引”的 Tensor，都必须是 LongTensor (int64) 类型。
        all_chunk_indices = torch.LongTensor(chunk_indices_list).to(self.device)
        all_chunk_masks = torch.FloatTensor(chunk_masks_list).to(self.device)
        # start_indices 保持为 list 或 numpy，用于索引 hidden_states 列表
        all_start_indices = np.array(chunk_start_indices)
        
        # 转换所有数据为 Tensor（一次性转换，效率更高）
        # 如果显存紧张，不要在这里 .to(device)，而是在 yield 之前转
        all_img_seqs = torch.cat(self.img_seqs).to(self.device)
        all_V_states = torch.FloatTensor(np.array(self.V_states)).to(self.device)
        all_pi_states = torch.FloatTensor(np.array(self.pi_states)).to(self.device)
        all_actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        all_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        all_returns = torch.FloatTensor(self.returns).to(self.device)
        all_advantages = torch.FloatTensor(self.advantages).to(self.device)
        all_values = torch.FloatTensor(np.array(self.values)).to(self.device)
        all_aux_pos = torch.FloatTensor(np.array(self.aux_pos)).to(self.device)
        all_aux_rot = torch.FloatTensor(np.array(self.aux_rot)).to(self.device)
        all_aux_vel = torch.FloatTensor(np.array(self.aux_vel)).to(self.device)
        all_aux_ang_vel = torch.FloatTensor(np.array(self.aux_ang_vel)).to(self.device)
        
        # 随机打乱序列块的顺序
        num_chunks = len(chunk_indices_list)
        # chunk_indices_list例：[[55,56，...62],[63,...,70],..., [97,...,104], ...]
        perm = torch.randperm(num_chunks) # 生成随机排列的索引
        shuffled_indices = all_chunk_indices[perm]
        # shuffled_masks例：[[1,1，...1],[1,...1,0],..., [1,...,1,0,0], ...]
        shuffled_masks = all_chunk_masks[perm]
        shuffled_start_indices = all_start_indices[perm.cpu().numpy()]
        
        # 按MiniBatch遍历所有chunk
        # shuffled_indices例：[[55,56，...62],[97,...,104],...,[63,...,70],...]
        for batch_start in range(0, num_chunks, mini_batch_size):
            # 对chunk列表计算当前batch的结束索引
            batch_end = min(batch_start + mini_batch_size, num_chunks)
            # 获取当前 batch 的索引矩阵 (Batch_Size, Seq_Len)
            # 例：[[55,56，...62],[97,...,104],...,[63,...,70]]
            batch_idx = shuffled_indices[batch_start:batch_end]
            # PyTorch使用高级索引一次性提取数据
            '''
            假设all_img_seqs 的形状是 (Total_Steps, C, H, W)，比如 (10000, 3, 64, 64)。
            batch_idx 的形状是 (Batch_Size, Seq_Len)，比如 (32, 8)。
            PyTorch 的处理逻辑如下：
            1. 看索引的维度：batch_idx 是一个 LongTensor。
            2. 匹配第一维：PyTorch 会把 batch_idx 中的每一个整数，都当作是去 all_img_seqs 的第 0 维（即 Total_Steps 那一维）查找数据的下标。
            3. 保留索引形状：结果张量的前面几个维度，会直接继承索引张量 (batch_idx) 的形状。
            4. 保留数据剩余维度：结果张量的后面几个维度，会继承数据张量 (all_img_seqs) 剩余的维度。
            Index.shape = (32, 8)
            Data.shape = (10000, 3, 64, 64)，去掉第 0 维后剩下 (3, 64, 64)。
            最终结果 = (32, 8) + (3, 64, 64) = (32, 8, 3, 64, 64)
            '''
            # 结果形状直接为 (Batch_Size, Seq_Len, Feature_Dim)，无需 pad_sequence
            batch_img_seqs = all_img_seqs[batch_idx]
            batch_V_states = all_V_states[batch_idx]
            batch_pi_states = all_pi_states[batch_idx] 
            batch_actions = all_actions[batch_idx]
            batch_log_probs = all_log_probs[batch_idx]
            batch_returns = all_returns[batch_idx]
            batch_advantages = all_advantages[batch_idx]
            batch_values = all_values[batch_idx]
            batch_aux_pos = all_aux_pos[batch_idx]
            batch_aux_rot = all_aux_rot[batch_idx]
            batch_aux_vel = all_aux_vel[batch_idx]
            batch_aux_ang_vel = all_aux_ang_vel[batch_idx]
            
            batch_masks = shuffled_masks[batch_start:batch_end]

            # 处理 Hidden States
            # Hidden states 存储在 list 中且可能为 None，难以完全向量化，
            # 但这部分数据量小，列表推导式足够快
            current_starts = shuffled_start_indices[batch_start:batch_end]
            # 返回每个chunk起始位置对应的hidden state
            batch_init_hidden = [self.hidden_states[i] for i in current_starts]
            # next() 会返回迭代器中的第一个元素，如果没找到则返回 None
            template_hidden = next((h for h in batch_init_hidden if h is not None), None)
            if template_hidden is not None:
                # valid_hiddens = [h for h in batch_init_hidden if h is not None]
                init_zeros = torch.zeros_like(template_hidden)
                processed_hiddens = [
                        h if h is not None else init_zeros 
                        for h in batch_init_hidden
                    ]
                init_hidden_batch = torch.cat(processed_hiddens, dim=1)
            else:
                # 如果当前 batch 全是 None (TemporalTransformer)，则返回 None
                init_hidden_batch = None

            yield (batch_img_seqs, batch_V_states, batch_pi_states, batch_actions,
                   batch_log_probs, batch_returns, batch_advantages, batch_values,
                   batch_aux_pos, batch_aux_rot, batch_aux_vel, batch_aux_ang_vel,
                   batch_masks, init_hidden_batch)

    def finish_path(self, last_value, done):
        """
        计算GAE：
            参考: Schulman et al. (2016) "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
            这个方法仅计算当前 episode的 GAE，支持多个 episode 累积在同一个 buffer 中
            只计算当前 episode的 GAE，不覆盖之前的 episode
        使用 last_episode_end 追踪当前 episode 的起始位置，每次调用后更新 last_episode_end，为下个 episode 做准备
        调用时机：每当一个 episode 结束时调用（done=True）
        """
        # 获取当前episode的范围
        episode_start = self.last_episode_end
        episode_end = len(self.rewards) # 用rewards长度代表buffer长度，作为当前episode结束位置
        episode_length = episode_end - episode_start # 当前 episode 的长度
        
        # 记录这个 episode 的边界（用于序列采样）
        self.episode_boundaries.append((episode_start, episode_end))
        
        # 只处理当前episode的数据
        episode_rewards = self.rewards[episode_start:episode_end]
        episode_values = self.values[episode_start:episode_end]
        episode_dones = self.dones[episode_start:episode_end]
        # 初始化advantage数组
        advantages = np.zeros(episode_length, dtype=np.float32)
        last_gae_lam = 0
        '''
        计算 TD error，需要下一步的价值 V(s_{t+1})
        # δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        # A_t = δ_t + (γλ) * δ_{t+1} + (γλ)² * δ_{t+2} + ...
        '''
        values = np.array(episode_values + [last_value], dtype=np.float32)
        rewards = np.array(episode_rewards, dtype=np.float32)
        dones = np.array(episode_dones + [done], dtype=np.float32)

        # 从后往前逆序计算GAE；GAE 是递归定义的，要计算 A_t，必须先知道 A_{t+1}
        '''
        range:生成一个[start, length-1]的整数序列
        reversed:将该序列反转，变成[length-1, start]
        '''
        for t in reversed(range(episode_length)):
            next_value = values[t + 1]
            
            ''' 
            TD error: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
            δ > 0：Critic 低估了这个状态的价值 → 应该增强这个动作
            δ < 0：Critic 高估了这个状态的价值 → 应该削弱这个动作
            (1.0 - dones[t + 1])用于截断未来价值，done=False时才保留未来价值
            '''
            delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t + 1]) - values[t]
            
            # GAE: A_t = δ_t + (γλ)*δ_{t+1} + (γλ)²*δ_{t+2} + ...
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * (1.0 - dones[t + 1]) * last_gae_lam
        '''
        Returns = Advantages + Values
        Returns: 从当前时刻开始到 episode 结束，能获得的【实际】累积折扣奖励(Q值)
        Values: V(s_t),当前时刻的【估计】能得到的状态价值，用于计算critic loss
        critic loss = (Gt - V(s_t))^2
        At = Q(s_t, a_t) - V(s_t) = Gt-Vt,表示当前动作比平均水平好了多少
        Gt = At + V(s_t) = Q(s_t, a_t)
        '''
        returns = advantages + values[:-1]

        # 计算当前 episode 的 returns 和 advantages
        episode_returns = returns.tolist()
        episode_advantages = advantages.tolist()
        
        # 追加到总的 returns 和 advantages
        self.returns.extend(episode_returns)
        self.advantages.extend(episode_advantages)
        
        # 更新 episode 结束位置，为下一个 episode 做准备
        self.last_episode_end = len(self.rewards)