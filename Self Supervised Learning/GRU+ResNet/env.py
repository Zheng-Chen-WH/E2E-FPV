import numpy as np
import airsim
from datetime import datetime
import time
import math
from gymnasium import spaces
import matplotlib.pyplot as plt
import os
from scipy.optimize import fsolve
from sac import SAC
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torch

'''每次都会忘的vsc快捷键：
    打开设置脚本：ctrl+shift+P
    多行注释：ctrl+/
    关闭多行注释：选中之后再来一次ctrl+/
    多行缩进：tab
    关闭多行缩进：选中多行shift+tab'''

class env:
    def __init__(self, args): # 门框的名称（确保与UE4中的名称一致）
        # 连接到AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        # 获取当前实际时间并设置为仿真时间
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.client.simSetTimeOfDay(True, start_datetime=current_time_str, celestial_clock_speed=1)

        self.DT = args['DT'] # 每个step无人机在airsim里自由飞的时长
        self.img_time = args['img_time'] # 两帧之间的间隔
        self.door_frames = args['door_frames']
        self.initial_pose = None # Will be set during reset for door orientation
        self.pass_threshold = args['pass_threshold_y']
        self.max_action = args['control_max']
        self.min_action = args['control_min']

        # 门框正弦运动运动参数
        self.door_param =  args['door_param']
        self.start_time = 0 # To be set at the beginning of each episode
        
        # 图像类型
        self.image_type = airsim.ImageType.Scene  # 图像类型（RGB）
        
        # 初始化迭代参数
        self.target_distance = args['POS_TOLERANCE']
        self.i=0
        self.info=0 # 完成情况flag
        self.phase_idx = 0

        # 正态分布噪声
        # self.sigma=np.degrees(0.001)
        # self.mu=math.degrees(0.005)

        # TimeSformer拍照帧数
        self.frames= args['frames']


        # 图像预处理器
        # 将图像转换为模型可接受的格式，同时调整尺寸
        self.transform = transforms.Compose([
            transforms.Resize((256, 144)),  # 调整图像尺寸为模型输入的尺寸
            transforms.ToTensor(),          # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])

    def _move_door(self, door_frame_name, position): 
        """将门移动到指定x,y,z位置的辅助函数, 保持初始姿态, 名字前加_使其只能在类内部被调用"""
        if self.initial_pose is None: # 如果没有被设定，则按照第一个门的姿态设定
            self.initial_pose = self.client.simGetObjectPose(door_frame_name)

        new_door_pos_vector = airsim.Vector3r(position[0], position[1], position[2])
        new_airsim_pose = airsim.Pose(new_door_pos_vector, self.initial_pose.orientation)
        self.client.simSetObjectPose(door_frame_name, new_airsim_pose, True)

    def _update_door_positions(self, elapsed_time):
        """基于已经过时间更新门位置"""
        for i, door_name in enumerate(self.door_frames):
            # 计算门的新x坐标
            new_x = self.door_param["initial_x_pos"][i] + \
                      self.door_param["amplitude"] * math.sin(
                          2 * math.pi * self.door_param["frequency"] * elapsed_time + self.door_param["deviation"][i])
            # 计算门的x速度
            self.door_x_velocities[i] = 2 * math.pi * self.door_param["frequency"] * \
                                       self.door_param["amplitude"] * math.cos(
                                           2 * math.pi * self.door_param["frequency"] * elapsed_time + self.door_param["deviation"][i])
            self.door_current_x_positions[i] = new_x
            # 门i的y位置是self.waypoints_y[i+1]
            self._move_door(door_name, np.array([new_x, self.waypoints_y[i+1], self.door_z_positions[i]]))
    
    # 航路点管理
    def _get_current_waypoint_index(self, current_y_pos, waypoints_y_list, threshold):
        # 航路点：[start_y, door1_y, door2_y, final_target_y]
        # index 0: target is door1 (at waypoints_y_list[1])
        # index 1: target is door2 (at waypoints_y_list[2])
        # index 2: target is final_target (at waypoints_y_list[3])
        if current_y_pos < waypoints_y_list[1] + threshold: # 靠近第一个门
            return 0
        elif current_y_pos < waypoints_y_list[2] + threshold: # elif确保已经越过了第一个门
            return 1
        else: # else确保越过了第二个门
            return 2

    def get_drone_state(self): # 给MPC的13维向量和给Q网络的25维状态向量
        # 获取无人机状态
        fpv_state_raw = self.client.getMultirotorState()

        '''生成MPC用的状态'''
        # 获取位置
        position = fpv_state_raw.kinematics_estimated.position
        fpv_pos = np.array([position.x_val, position.y_val, position.z_val])

        # 获取线速度
        linear_velocity = fpv_state_raw.kinematics_estimated.linear_velocity
        fpv_vel = np.array([linear_velocity.x_val, linear_velocity.y_val, linear_velocity.z_val])

        # 获取姿态角 (俯仰pitch, 滚转roll, 偏航yaw, 欧拉角表示, 弧度制)
        orientation_q = fpv_state_raw.kinematics_estimated.orientation
        fpv_attitude = np.array([orientation_q.w_val, orientation_q.x_val, orientation_q.y_val, orientation_q.z_val]) # 四元数表示
        # pitch, roll, yaw = airsim.to_eularian_angles(orientation_q) # # 将四元数转换为欧拉角 (radians)
        # fpv_attitude = np.array([pitch, roll, yaw])

        # 获取角速度
        angular_velocity = fpv_state_raw.kinematics_estimated.angular_velocity
        fpv_angular_vel = np.array([angular_velocity.x_val, angular_velocity.y_val, angular_velocity.z_val])

        '''生成Q网络用的状态'''
        # 获取无人机相对最终目标的位置
        relative_pos_target=np.array([position.x_val-self.final_target_state[0],
                                     position.y_val-self.final_target_state[1],
                                     position.z_val-self.final_target_state[2]])

        # 获取无人机相对两个门的位置、速度
        relative_pos_door_one=np.array([self.door_current_x_positions[0] - position.x_val,
                                      self.waypoints_y[1] - position.y_val,
                                      - position.z_val])
        relative_vel_door_one=np.array([self.door_x_velocities[0] - linear_velocity.x_val,
                                        - linear_velocity.y_val,
                                        - linear_velocity.z_val])
        relative_pos_door_two=np.array([self.door_current_x_positions[1] - position.x_val,
                                      self.waypoints_y[2] - position.y_val,
                                      - position.z_val])
        relative_vel_door_two=np.array([self.door_x_velocities[1] - linear_velocity.x_val,
                                        - linear_velocity.y_val,
                                        - linear_velocity.z_val])

        return np.concatenate((fpv_pos, fpv_vel, fpv_attitude, fpv_angular_vel)), \
            np.concatenate((fpv_vel, fpv_attitude, fpv_angular_vel, 
                            relative_pos_door_one, relative_vel_door_one,  
                            relative_pos_door_two, relative_vel_door_two, relative_pos_target))
    
    def get_img_sequence(self):
        img_vector = []
        for i in range(self.frames): # 在这里需要计算拍照时间，可能考虑异步编程
            # 计算当前时间
            # deviation=0.0 # 不同门框错开
            # elapsed_time = time.time() - self.start_time
            self._update_door_positions(self.elapsed_time) # 更新门位置
            self.elapsed_time = self.elapsed_time + self.img_time
            self.client.simPause(True)
            # 拍照
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, pixels_as_float=False, compress=False)
            ])
            if responses:
                # 将 AirSim 的字节流转换为 NumPy 数组
                img_data = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                # 重塑为正确的图像形状 (Height x Width x 3 Channels)
                img_rgb = img_data.reshape(responses[0].height, responses[0].width, 3)
                # 转换 BGR 到 RGB（AirSim 默认返回 BGR 格式）
                img_rgb = img_rgb[..., ::-1]  # BGR → RGB

                # 使用预处理将图像转换为张量
                img_tensor = self.transform(Image.fromarray(img_rgb))
                img_vector.append(img_tensor)  # 添加到序列
            self.client.simPause(False)
            # 等待下一帧
            time.sleep(self.img_time)

        # 将序列堆叠为 Tensor (num_frames, channels, height, width)
        img_vector = torch.stack(img_vector, dim=0)  # 输出维度: (4, 3, 256, 144)

        # 添加 Batch 维度 (batch_size, num_frames, channels, height, width)
        img_vector = img_vector.unsqueeze(0)  # 输出维度: (1, 4, 3, 256, 144)

        return img_vector

    def reset(self):
        # AirSim状态重置与初始化
        self.client.simPause(False) # 解除暂停
        for attempt in range(10):
            # print(f"Attempting to reset and initialize drone (Attempt {attempt + 1}/{10})...")
            try:
                self.client.reset()
                time.sleep(0.5) # 短暂等待AirSim完成重置，根据需要调整
                self.client.enableApiControl(True)
                self.client.armDisarm(True)
                time.sleep(0.5)

                # 验证状态
                if not self.client.isApiControlEnabled():
                    print("Failed to enable API control after reset.")
                    continue
                # print(f"Drone reset and initialized successfully (Attempt {attempt + 1}).")
                break
            except Exception as e:
                print(f"Error during drone initialization (Attempt {attempt + 1}): {e}")
                try:
                    self.client.confirmConnection()
                except Exception as conn_err:
                    print(f"Failed to re-confirm connection: {conn_err}")
                    break
                time.sleep(1)
        else: # If loop completes without break
            raise RuntimeError("Failed to reset and initialize drone after multiple attempts.")

        # 定义无人机的初始位置和方向
        # FPV_position=np.array([np.random.uniform(-3,3), np.random.uniform(-5,5), -1.0])
        # initial_drone_position = airsim.Vector3r(FPV_position[0],FPV_position[1],FPV_position[2])  # 定义位置 (x=10, y=20, z=-0.5)
        # yaw = math.radians(90)  # 90 度（朝向正 y 轴）
        # # 创建 Pose 对象
        # initial_drone_pose = airsim.Pose(initial_drone_position, airsim.to_quaternion(0.0, 0.0, yaw))
        # # 设置无人机初始位置
        # self.client.simSetVehiclePose(initial_drone_pose, ignore_collision=True)
        
        # 航路点与门初始化
        self.waypoints_y = [0.0] # 起点Y位置、各扇门Y位置、终点Y位置
        # self.way_points_y.append(FPV_position[1])
        self.door_initial_x_positions = []
        self.door_current_x_positions = [] # 存储门的当前位置
        self.door_z_positions = []
        self.door_x_velocities = np.zeros(len(self.door_frames)) #存储门的速度
        self.door_param["deviation"] = np.random.uniform(0, 10, size=len(self.door_frames))

        self.initial_pose = None # 在第一次执行movedoor的时候将设为第一个门的姿态

        for i, door_name in enumerate(self.door_frames):
            try:
                # 获取initial pose
                current_door_pose_raw = self.client.simGetObjectPose(door_name)
                if self.initial_pose is None: # 储存第一个门的朝向
                    self.initial_pose = current_door_pose_raw   
                initial_door_z = current_door_pose_raw.position.z_val # 保留z坐标

                # 随机生成门初始位置
                new_x = 0 + np.random.uniform(-1, 1)
                new_y = (i + 1) * 15 + np.random.uniform(-2, 2)
                
                self._move_door(door_name, np.array([new_x, new_y, initial_door_z]))
                
                self.door_initial_x_positions.append(new_x)
                self.door_current_x_positions.append(new_x) 
                self.door_z_positions.append(initial_door_z)
                self.waypoints_y.append(new_y)

            except Exception as e:
                print(f"Error processing door '{door_name}': {e}")
                print(f"请确保场景中存在名为 '{door_name}' 的对象。")
                raise

        self.door_param["initial_x_pos"] = self.door_initial_x_positions

        # 最终目标状态初始化
        self.final_target_state = np.array([
            np.random.uniform(-1, 1),    # 目标位置x
            np.random.uniform(43, 47),   # 目标位置y
            np.random.uniform(-3, -2),   # 目标位置z
            0.0, 0.0, 0.0,               # 目标速度x, y, z
            0.707, 0.0, 0.0, 0.707,              # 目标姿态四元数
            0.0, 0.0, 0.0                # 目标角速度x, y, z
        ])
        self.waypoints_y.append(self.final_target_state[1])

        # 设置目标点视觉标记物（橙球）
        target_ball_pos = airsim.Vector3r(self.final_target_state[0], self.final_target_state[1], self.final_target_state[2])
        ball_initial_pose = self.client.simGetObjectPose("OrangeBall_Blueprint")
        self.client.simSetObjectPose("OrangeBall_Blueprint", airsim.Pose(target_ball_pos, ball_initial_pose.orientation), True)

        self.client.takeoffAsync().join()
        time.sleep(0.5)

        self.start_time = time.time()
        self._update_door_positions(0.0)
        self.door_param["start_time"] = self.start_time

        self.phase_idx = 0
        self.elapsed_time = time.time() - self.start_time
        img_tensor = self.get_img_sequence() # 获取图像编码张量
        current_drone_state, Q_state = self.get_drone_state()
        self.start_time_step = time.time()

        collision_info = self.client.simGetCollisionInfo()
        self.first_collide_time = collision_info.time_stamp / 1e9

        self.info = 0
        self.done = False
        self.past_actions = np.array([0.5, 0.5, 0.5, 0.5,
                                0.5, 0.5, 0.5, 0.5,
                                0.5, 0.5, 0.5, 0.5])
        self.final_pi_target = self.final_target_state[0:3] # 只看位置

        return current_drone_state, self.final_target_state, self.waypoints_y,\
                self.door_z_positions, self.door_param, img_tensor, self.past_actions, Q_state, self.final_pi_target, self.elapsed_time

    def step(self, control_signal):
        # 发送油门指令
        # end_time=time.time()
        # print("calculation time consumed:", end_time-self.start_time_step)
        self.client.simPause(False)
        self.client.moveByMotorPWMsAsync(float(control_signal[0]),float(control_signal[1]),float(control_signal[2]),float(control_signal[3]), self.DT*2)
        time.sleep(self.DT-4*self.img_time) # 仿真持续步长
        
        self.elapsed_time = self.elapsed_time + self.DT - 4 * self.img_time
        # 理论上这里要过4个self.img_time
        img_tensor = self.get_img_sequence()
        self._update_door_positions(self.elapsed_time) # 更新门位置

        # 往期动作也需要缩放
        self.past_actions = np.concatenate((self.past_actions[4:],\
                                             2 * (control_signal - self.min_action) / (self.max_action - self.min_action) - 1))
        self.client.simPause(True)
        # self.start_time_step=time.time()

        current_drone_state, Q_state = self.get_drone_state()
        self.phase_idx = self._get_current_waypoint_index(current_drone_state[1], self.waypoints_y, self.pass_threshold)
        # print(f"airsim仿真环境, {current_drone_state[0:3]},速度,{current_drone_state[3:6]},姿态四元数{current_drone_state[6:10]},角速度{current_drone_state[10:13]}")
        # print("————————————————————————————————————")
        collision_info = self.client.simGetCollisionInfo()
        
        collided = False
        # 碰撞时间需要大于一个小阈值，避免起飞碰撞被判定为碰撞
        if collision_info.has_collided and (collision_info.time_stamp / 1e9 > self.first_collide_time + 0.5) :
            collided = True

        # 计算进度奖励 (R_progress)
        R_approach = - np.linalg.norm(Q_state[22:25]) # 距离最终目标的距离

        R_centering = 0
        R_velocity_align = 0
        if self.phase_idx != 2:
            R_centering = - np.linalg.norm(Q_state[10 + self.phase_idx * 6 : 13 + self.phase_idx * 6]) # 相对下一个门的距离
            R_velocity_align = - np.linalg.norm(Q_state[13 + self.phase_idx * 6]) # 相对下一个门x方向的速度

        R_progress = 0.2 * R_approach + 1.0 * R_centering + 5.0 * R_velocity_align # 进度奖励加权

        # 计算成本惩罚 (R_cost)
        R_action_cost = - 1.0 * np.linalg.norm(control_signal) # 控制代价
        R_stability_cost = - 5.0 * np.linalg.norm(current_drone_state[10:]) # 无人机角速度尽可能小
        R_time_cost = - 1.0 # 耗时代价
        R_cost = R_action_cost + R_stability_cost + R_time_cost

        # 计算事件奖励 (R_event)
        R_event = self.phase_idx * 100
        if collided:
            R_event -= 200
            self.done=True
            self.info=0
            self.i+=1
        elif np.linalg.norm(Q_state[22:25]) < self.target_distance:
            R_event += 300
            self.done = True
            self.info = 1
            self.i += 1

        # 5. 计算总奖励
        reward = R_progress + R_event + R_cost
        
        return current_drone_state, img_tensor, self.past_actions, Q_state, reward, self.done, self.phase_idx, self.info, self.elapsed_time