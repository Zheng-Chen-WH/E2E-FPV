import random
import numpy as np
import os
import pickle
from collections import deque

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