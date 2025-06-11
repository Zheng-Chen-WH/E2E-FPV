import random
import numpy as np
import os
import pickle

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, pi_img, pi_state, q_state, action, reward, next_pi_img, next_pi_state, next_q_state, done, goal, info):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None) #没有充满时先用none创造出self.position
        self.buffer[self.position] = (pi_img.squeeze(0), pi_state, q_state, action, reward, next_pi_img.squeeze(0), next_pi_state, next_q_state, done, goal)
        # 把两个视频张量的多余维度挤掉，输出stack的时候才能叠回正常batchsize
        self.position = (self.position + 1) % self.capacity #超出记忆池容量后从第一个开始改写以维持容量不超标

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        pi_img, pi_state, q_state, action, reward, next_pi_img, next_pi_state, next_q_state, done, goal = map(np.stack, zip(*batch))
        '''zip()函数用于将多个可迭代对象（例如列表、元组等）的对应元素打包成一个元组，返回一个迭代器。
        map()函数将一个函数应用于迭代器中的每个元素，并返回一个新的迭代器。
        这里将np.stack()函数应用于zip(*batch)的每个元素。np.stack()函数用于沿着新的轴堆叠数组序列，返回一个新的数组。
        map(np.stack, zip(*batch))将返回一个包含多个新数组的迭代器，其中每个新数组由批次中对应元素的堆叠组成。
        最后将上一步得到的迭代器中的新数组分别赋值给state、action、reward、next_state和done这五个变量。这意味着每个变量都是一个包含了批次中对应元素的堆叠数组。
        【迭代器】：允许按需逐个访问集合中的元素，而不是一次性获取整个集合；range()函数、zip()函数和字典的items()方法都返回迭代器，
        还可以使用关键字yield来定义生成器函数，生成器函数返回的对象也是迭代器'''
        return pi_img, pi_state, q_state, action, reward, next_pi_img, next_pi_state, next_q_state, done, goal

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

class HERMemory: # 暂定使用的
    def __init__(self, capacity, method):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.temporary_buffer=[]
        self.HERmethod=method
        self.target_distance=0.5

    def reward_cal(self, next_state, goal, new_goal):
        done=False
        if np.linalg.norm(goal+next_state[9:12]-new_goal)<self.target_distance:
            done=True
        if not done:
            reward=(np.linalg.norm(self.initial_pos-new_goal)-np.linalg.norm(goal+next_state[9:12]-new_goal))/5
        if done:
            reward=200
        return reward, done

    def push(self, pi_img, pi_state, q_state, action, reward, next_pi_img, next_pi_state, next_q_state, done, goal, info):
        self.temporary_buffer.append((pi_img, pi_state, q_state, action, reward, next_pi_img, next_pi_state, next_q_state, done, goal))
        if done:
                if not info: #先把没有成功时的最后一组单独塞进去，然后把缓存切片
                    if len(self.buffer) < self.capacity:
                        self.buffer.append(None)
                    self.buffer[self.position] = self.temporary_buffer[-1]
                    self.position = (self.position + 1) % self.capacity
                    self.temporary_buffer=self.temporary_buffer[:-1]
                    self.initial_pos=np.linalg.norm(goal+self.temporary_buffer[0][2][9:12])
                if self.HERmethod=='final':
                    new_goal=goal+self.temporary_buffer[-1][2][9:12]
                    for ii in self.temporary_buffer:
                        if len(self.buffer) < self.capacity:
                            self.buffer.append(None) #没有充满时先用none创造出self.position
                        new_reward, new_done=self.reward_cal(next_q_state, goal, new_goal)
                        if len(self.buffer) < self.capacity:
                            self.buffer.append(None) #没有充满时先用none创造出self.position
                        self.buffer[self.position]=(ii[0], ii[1], ii[2], ii[3], ii[4], new_reward, ii[6], ii[7], ii[8], new_done, new_goal)
                        self.position = (self.position + 1) % self.capacity
                    self.temporary_buffer=[]
                if self.HERmethod=='future':
                    for index,ii in enumerate(self.temporary_buffer): #用来同时获取索引和值
                        if len(self.buffer) < self.capacity:
                            self.buffer.append(None) #没有充满时先用none创造出self.position
                        self.buffer[self.position] = ii #把原目标下的参数塞进hermemory
                        self.position = (self.position + 1) % self.capacity
                        if len(self.temporary_buffer)>index+1:
                            new_goal_no=np.random.choice(np.arange(index+1, len(self.temporary_buffer)), size=1, replace=False) #replace是确认是否整数重复
                            new_goal=self.temporary_buffer[new_goal_no.item()][2][9:12]
                        else:
                            new_goal=self.temporary_buffer[-1][2][9:12]
                        new_reward, new_done=self.reward_cal(next_q_state, goal, new_goal)
                        if len(self.buffer) < self.capacity:
                            self.buffer.append(None) #没有充满时先用none创造出self.position
                        self.buffer[self.position]=(ii[0], ii[1], ii[2], ii[3], ii[4], new_reward, ii[6], ii[7], ii[8], new_done, new_goal)
                        self.position = (self.position + 1) % self.capacity
                    self.temporary_buffer=[]

    def sample(self, batch_size):
        #np.random.seed(42)
        batch = random.sample(self.buffer, batch_size)
        pi_img, pi_state, q_state, action, reward, next_pi_img, next_pi_state, next_q_state, done, goal = map(np.stack, zip(*batch))
        '''zip()函数用于将多个可迭代对象（例如列表、元组等）的对应元素打包成一个元组，返回一个迭代器。
        map()函数将一个函数应用于迭代器中的每个元素，并返回一个新的迭代器。
        这里将np.stack()函数应用于zip(*batch)的每个元素。np.stack()函数用于沿着新的轴堆叠数组序列，返回一个新的数组。
        map(np.stack, zip(*batch))将返回一个包含多个新数组的迭代器，其中每个新数组由批次中对应元素的堆叠组成。
        最后将上一步得到的迭代器中的新数组分别赋值给state、action、reward、next_state和done这五个变量。这意味着每个变量都是一个包含了批次中对应元素的堆叠数组。
        【迭代器】：允许按需逐个访问集合中的元素，而不是一次性获取整个集合；range()函数、zip()函数和字典的items()方法都返回迭代器，
        还可以使用关键字yield来定义生成器函数，生成器函数返回的对象也是迭代器'''
        return pi_img, pi_state, q_state, action, reward, next_pi_img, next_pi_state, next_q_state, done, goal

    def __len__(self):
        return len(self.buffer)
        # 双下划线（__）用于表示特殊方法或特殊属性。这些特殊方法和属性具有预定义的名称，它们在对象上具有特殊的行为。
        # __len__() 是一个特殊方法，用于返回对象的长度

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity