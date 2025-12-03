import numpy as np
import airsim
from datetime import datetime
import time
import math
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

        # 固定仿真时间
        self.client.simSetTimeOfDay(True, start_datetime="2025-07-08 17:00:00", celestial_clock_speed=0)

        self.DT = args['DT'] # 每个step无人机在airsim里自由飞的时长
        self.img_time = args['img_time'] # 两帧之间的间隔
        self.door_frames = args['door_frames']
        self.initial_pose = None # Will be set during reset for door orientation
        self.pass_threshold = args['pass_threshold_y']
        self.max_action = args['control_max']
        self.min_action = args['control_min']
        self.reward_weight = args['reward_weight']

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
        self.last_phase_idx = 0

        self.last_y_pos = 0.0
        self.last_potential = 0

        # 正态分布噪声
        # self.sigma=np.degrees(0.001)
        # self.mu=math.degrees(0.005)

        self.frames= args['frames']
        self.elapsed_time = 0.0

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
    # def _get_current_waypoint_index(self, current_y_pos, waypoints_y_list, threshold):
    #     # 航路点：[start_y, door1_y, door2_y, final_target_y]
    #     # index 0: target is door1 (at waypoints_y_list[1])
    #     # index 1: target is door2 (at waypoints_y_list[2])
    #     # index 2: target is final_target (at waypoints_y_list[3])
    #     if current_y_pos < waypoints_y_list[1] + threshold: # 靠近第一个门
    #         return 0
    #     elif current_y_pos < waypoints_y_list[2] + threshold: # elif确保已经越过了第一个门
    #         return 1
    #     else: # else确保越过了第二个门
    #         return 2
    
    def check_gate_passage(self, current_drone_state, waypoints_y, relative_target_position, door_params):
        
        """
        检查无人机是否成功穿越了当前阶段的目标门，并返回新的phase_idx和事件奖励。
        Args:
            current_drone_state (np.array): 无人机完整状态，包含x,y,z位置。
            waypoints_y (list): [start_y, door1_y, door2_y, final_target_y]
            door_params (dict): 包含门的位置和尺寸信息，例如门中心的X和Z，以及门的宽度和高度。
        Returns:
            passage_successful, event_reward
        """
        current_pos = current_drone_state[0:3]
        current_y = current_pos[1]
        passage_successful = False

        new_phase_idx = self.phase_idx
        event_reward = 0
        
        # 检查是否从阶段0（目标门1）切换到阶段1
        if self.phase_idx == 0:
            target_door_y = waypoints_y[1]
            
            # 检测“穿越平面”事件：从Y值小的一侧移动到Y值大的一侧
            if self.last_y_pos < target_door_y+ self.pass_threshold and current_y >= target_door_y+ self.pass_threshold:
                # print("穿越门1平面")
                
                # 在穿越瞬间，验证X和Z坐标是否在门框内
                door_width = door_params["width"]   # 您需要在door_params中添加门的尺寸
                door_height = door_params["height"]

                # 检查X坐标
                is_x_valid = abs(relative_target_position[0]) < (door_width / 2.0)
                # 检查Z坐标
                # print("z relative distance:", relative_target_position[2])
                is_z_valid = abs(relative_target_position[2]) < (door_height / 2.0)
                passage_successful = True
                
                if is_x_valid and is_z_valid:
                    # print("成功穿越门1！")
                    # new_phase_idx = 1  # 切换到下一阶段
                    event_reward = self.reward_weight['GATE_PASS_BONUS']  # 给予一次性的里程碑奖励
                else:
                    # print("从门1旁边绕过或撞门框！")
                    # 如果没有成功穿越，可以给予一个小的负奖励，或者什么都不做
                    event_reward = self.reward_weight['GATE_BYPASS_PENALTY']
        
        # 检查是否需要从阶段1（目标门2）切换到阶段2
        elif self.phase_idx == 1:
            target_door_y = waypoints_y[2]
            
            if self.last_y_pos < target_door_y and current_y >= target_door_y:
                # print("穿越门2平面")

                door_width = door_params["width"]
                door_height = door_params["height"]

                is_x_valid = abs(relative_target_position[0]) < (door_width / 2.0)
                # print("z relative distance:", relative_target_position[2])
                is_z_valid = abs(relative_target_position[2]) < (door_height / 2.0)
                passage_successful = True

                if is_x_valid and is_z_valid:
                    # print("成功穿越门2！")
                    # new_phase_idx = 2  # 切换到最终目标阶段
                    event_reward = 5  # 再次给予奖励
                else:
                    # print("从门2旁边绕过或撞门框！")
                    event_reward = -3
        
        # 更新上一步的Y坐标，为下一次检测做准备
        self.last_y_pos = current_y
        
        return passage_successful, event_reward # new_phase_idx,
    
    # 将四元数转成旋转矩阵
    def quaternions_to_rotation_matrices(self, quaternions):
        """
        将四元数转换为旋转矩阵。
        参数: quaternions: 一个形状为 (..., 4) 的NumPy数组，最后一个维度是 (w, x, y, z)。
        返回: 一个形状为 (..., 3, 3) 的旋转矩阵NumPy数组。
        """
        # 归一化
        q_norm = np.linalg.norm(quaternions, axis=-1, keepdims=True)
        quaternions = quaternions / q_norm
        
        w, x, y, z = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
        
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        
        rot_mats = np.zeros(quaternions.shape[:-1] + (3, 3), dtype=np.float32)
        
        rot_mats[..., 0, 0] = 1 - 2 * (yy + zz)
        rot_mats[..., 0, 1] = 2 * (xy - wz)
        rot_mats[..., 0, 2] = 2 * (xz + wy)
        
        rot_mats[..., 1, 0] = 2 * (xy + wz)
        rot_mats[..., 1, 1] = 1 - 2 * (xx + zz)
        rot_mats[..., 1, 2] = 2 * (yz - wx)
        
        rot_mats[..., 2, 0] = 2 * (xz - wy)
        rot_mats[..., 2, 1] = 2 * (yz + wx)
        rot_mats[..., 2, 2] = 1 - 2 * (xx + yy)
        
        return rot_mats
    
    def quat_rotate_vector(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        使用四元数 q (w, x, y, z) 旋转一个三维向量 v.
        """
        # 将向量 v 转换为纯四元数 (0, vx, vy, vz)
        v_quat = np.array([0, v[0], v[1], v[2]])
        
        # 计算四元数共轭 q* = (w, -x, -y, -z)
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        
        # 四元数乘法: q1 * q2
        def quat_multiply(q1, q2):
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            return np.array([w, x, y, z])

        # 旋转公式: v' = q * v * q*
        rotated_v_quat = quat_multiply(quat_multiply(q, v_quat), q_conj)
        
        # 返回旋转后向量的部分
        return rotated_v_quat[1:]

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
        attitude_9d = self.quaternions_to_rotation_matrices(fpv_attitude)
        attitude_6d = np.concatenate((attitude_9d[0], attitude_9d[1]))
        # pitch, roll, yaw = airsim.to_eularian_angles(orientation_q) # # 将四元数转换为欧拉角 (radians)
        # fpv_attitude = np.array([pitch, roll, yaw])

        # 获取角速度
        angular_velocity = fpv_state_raw.kinematics_estimated.angular_velocity
        fpv_angular_vel = np.array([angular_velocity.x_val, angular_velocity.y_val, angular_velocity.z_val])

        '''生成Q网络用的状态'''
        # 获取无人机相对最终目标的位置
        relative_pos_target=np.array([self.final_target_state[0]-position.x_val,
                                     self.final_target_state[1]-position.y_val,
                                     self.final_target_state[2]-position.z_val])

        # 获取无人机相对两个门的位置、速度
        relative_pos_door_one=np.array([self.door_current_x_positions[0] - position.x_val,
                                      self.waypoints_y[1] - position.y_val,
                                      self.door_param["center"] + position.z_val])
        relative_vel_door_one=np.array([self.door_x_velocities[0] - linear_velocity.x_val,
                                        - linear_velocity.y_val,
                                        - linear_velocity.z_val])
        relative_pos_door_two=np.array([self.door_current_x_positions[1] - position.x_val,
                                      self.waypoints_y[2] - position.y_val,
                                       self.door_param["center"] + position.z_val])
        relative_vel_door_two=np.array([self.door_x_velocities[1] - linear_velocity.x_val,
                                        - linear_velocity.y_val,
                                        - linear_velocity.z_val])
        # print(position.z_val)
        # 获取无人机相对阶段目标的位置、速度
        if self.phase_idx == 0:
            relative_next_target_pos = relative_pos_door_one
            relative_next_target_vel = relative_vel_door_one
        elif self.phase_idx == 1:
            relative_next_target_pos = relative_pos_door_two
            relative_next_target_vel = relative_vel_door_two
        else:
            relative_next_target_pos = relative_pos_target
            relative_next_target_vel = - fpv_vel
        
        # 给辅助头的标签要/10
        return np.concatenate((fpv_pos, fpv_vel, fpv_attitude, fpv_angular_vel)), \
            np.concatenate((fpv_vel/10.0, attitude_6d, fpv_angular_vel, 
                            relative_next_target_pos/10.0, relative_next_target_vel/10.0, relative_pos_target/10.0)), \
                            relative_next_target_pos/10.0, attitude_9d, relative_next_target_vel/10.0, fpv_angular_vel
    
    def get_img_sequence(self):
        img_vector = []
        relative_next_target_pos, attitude_9d, relative_next_target_vel, fpv_angular_vel = [], [], [], []
        for i in range(self.frames): # 在这里需要计算拍照时间，可能考虑异步编程
            # 计算当前时间
            # deviation=0.0 # 不同门框错开
            # elapsed_time = time.time() - self.start_time
            # 等待下一帧
            '''time.sleep(self.img_time)'''
            self.client.simContinueForTime(self.img_time)
            self.elapsed_time += self.img_time
            self._update_door_positions(self.elapsed_time) # 更新门位置
            '''self.client.simPause(True)'''
            _, _, pos, atti, vel, angular = self.get_drone_state()
            relative_next_target_pos.append(pos)
            attitude_9d.append(atti)
            relative_next_target_vel.append(vel)
            fpv_angular_vel.append(angular)
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
            '''self.client.simPause(False)'''
        
        relative_next_target_pos = np.stack(relative_next_target_pos)
        attitude_9d = np.stack(attitude_9d)
        relative_next_target_vel = np.stack(relative_next_target_vel)
        fpv_angular_vel = np.stack(fpv_angular_vel)

        # 将序列堆叠为 Tensor (num_frames, channels, height, width)
        img_vector = torch.stack(img_vector, dim=0)  # 输出维度: (4, 3, 256, 144)

        # 添加 Batch 维度 (batch_size, num_frames, channels, height, width)
        img_vector = img_vector.unsqueeze(0)  # 输出维度: (1, 4, 3, 256, 144)

        return img_vector, relative_next_target_pos, attitude_9d, relative_next_target_vel, fpv_angular_vel

    def reset(self, seed=None):
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
            # random.seed(seed) # 如果使用了random库
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            # random.seed(seed) # 如果使用了random库

        # AirSim状态重置与初始化
        self.client.simPause(False) # 解除暂停
        for attempt in range(10):
            # print(f"Attempting to reset and initialize drone (Attempt {attempt + 1}/{10})...")
            try:
                self.client.reset()
                '''time.sleep(0.5)''' # 短暂等待AirSim完成重置，根据需要调整
                '''self.client.simPause(True)''' 

                self.client.enableApiControl(True)
                self.client.armDisarm(True)
                '''time.sleep(0.5)'''

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
        
        # 设置初始位置 (模拟起飞后的状态，高度 -1.5m，设置初始 Yaw 为 90 度 (math.pi / 2))
        # 使用 seed 控制的随机数生成器产生微小扰动（如果需要完全固定，可设为0）
        '''self.client.simPause(True)'''
        # start_x = 0.0 #np.random.uniform(-0.1, 0.1)
        # start_y = 0.0 #np.random.uniform(-0.1, 0.1)
        # start_z = -1.8 
        # start_pos = airsim.Vector3r(start_x, start_y, start_z)
        # start_pose = airsim.Pose(start_pos, airsim.to_quaternion(0, 0, np.pi/2))
        # self.client.simSetVehiclePose(start_pose, ignore_collision=True)
        # # 稳定机身：发送悬停指令(速度0)并推进物理时间
        # '''self.client.cancelLastTask()'''
        # # 这一步确保无人机在开始episode前处于动力学稳定状态，且过程是确定性的
        # self.client.moveByVelocityAsync(0, 0, 0, duration=10) 
        # self.client.simContinueForTime(1.0) # 推进1秒仿真时间让无人机稳定

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
        
        # print(f"seed:{seed},door_position:{self.door_current_x_positions}, {self.waypoints_y}, {self.door_z_positions}")
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
        '''time.sleep(0.5)'''
        self.client.simContinueForTime(0.5)
        self.client.simPause(True)

        self.start_time = time.time()
        self._update_door_positions(0.0)
        self.door_param["start_time"] = self.start_time

        self.phase_idx = 0
        self.last_phase_idx = 0

        '''self.elapsed_time = time.time() - self.start_time'''
        self.elapsed_time = 0.0

        img_tensor, relative_next_target_pos, attitude_9d, relative_next_target_vel, fpv_angular_vel = self.get_img_sequence() # 获取图像编码张量
        current_drone_state, Q_state, _, _, _, _ = self.get_drone_state()
        self.last_y_pos = current_drone_state[1]
        dist_next_target = np.linalg.norm(Q_state[12:15])
        vel_to_next_target = np.linalg.norm(Q_state[15:18])
        dist_to_final_target = np.linalg.norm(Q_state[18:21])
        self.last_potential = (
            - self.reward_weight['W_POS_PROG'] * dist_next_target
            - self.reward_weight['W_VEL_ALIGN'] * vel_to_next_target
            - self.reward_weight['W_FINAL_PULL'] * dist_to_final_target
        )

        self.start_time_step = time.time()

        collision_info = self.client.simGetCollisionInfo()
        self.first_collide_time = collision_info.time_stamp / 1e9

        self.info = 0
        self.done = False
        # self.past_actions = np.array([0.6, 0.6, 0.6, 0.6,
        #                         0.6, 0.6, 0.6, 0.6,
        #                         0.6, 0.6, 0.6, 0.6])
        self.final_pi_target = np.array([self.final_target_state[0], self.final_target_state[1]/47.0, self.final_target_state[2]/(-3.0)]) # 只看位置

        return current_drone_state, self.final_target_state, self.waypoints_y,\
                self.door_z_positions, self.door_param, img_tensor, Q_state, self.final_pi_target, self.elapsed_time,\
                relative_next_target_pos, attitude_9d, relative_next_target_vel, fpv_angular_vel

    def step(self, control_signal):
        # 发送油门指令
        # end_time=time.time()
        # print("calculation time consumed:", end_time-self.start_time_step)
        '''self.client.simPause(False)'''
        # print(f"control signal received by env:{control_signal}")
        self.client.moveByMotorPWMsAsync(float(control_signal[0]),float(control_signal[1]),float(control_signal[2]),float(control_signal[3]), self.DT*10)
        # time.sleep(self.DT-4*self.img_time) # 仿真持续步长
        
        # self.elapsed_time = self.elapsed_time + self.DT - 4 * self.img_time
        # 这里要过4个self.img_time
        img_tensor, relative_next_target_pos, attitude_9d, relative_next_target_vel, fpv_angular_vel = self.get_img_sequence()

        # 往期动作也需要缩放
        # self.past_actions = np.concatenate((self.past_actions[4:], control_signal))
        '''self.client.simPause(True)'''
        # self.start_time_step=time.time()

        # 基于旧的phase_idx检测是否穿门，再用检测后的Phase_idx计算状态，避免出现phase_idx更新与状态更新异步进而导致势跳变
        temp_drone_state, _, temp_relative_pos, _, _, _ = self.get_drone_state()
        phase_switched, pass_reward = self.check_gate_passage(
            temp_drone_state, self.waypoints_y, temp_relative_pos * 10, self.door_param
        )
        if phase_switched:
            self.phase_idx += 1

        current_drone_state, Q_state, relative_pos, _, _, _ = self.get_drone_state()
        # self.phase_idx = self._get_current_waypoint_index(current_drone_state[1], self.waypoints_y, self.pass_threshold)
        # self.phase_idx, pass_reward = self.check_gate_passage(current_drone_state, self.waypoints_y, relative_pos*10, self.door_param)
        # print(f"airsim仿真环境, {current_drone_state[0:3]},速度,{current_drone_state[3:6]},姿态四元数{current_drone_state[6:10]},角速度{current_drone_state[10:13]}")
        # print("————————————————————————————————————")
        collision_info = self.client.simGetCollisionInfo()
        
        collided = False
        # 碰撞时间需要大于一个小阈值，避免起飞碰撞被判定为碰撞
        if collision_info.has_collided and (collision_info.time_stamp / 1e9 > self.first_collide_time + 0.5) :
            collided = True

        # Q state
        # np.concatenate((fpv_vel/10.0, attitude_6d, fpv_angular_vel, 
        # relative_next_target_pos/10.0, relative_next_target_vel/10.0, relative_pos_target/10.0))

        # 从Q状态中提取信息
        # 注意：这些值都是基于已经 /10 的状态向量
        dist_next_target = np.linalg.norm(Q_state[12:15])
        vel_to_next_target = np.linalg.norm(Q_state[15:18])
        dist_to_final_target = np.linalg.norm(Q_state[18:21])

        # 定义并计算当前步势能函数
        # 势能基于“下一个阶段性目标”
        current_potential = (
            - self.reward_weight['W_POS_PROG'] * dist_next_target          # 接近下一个目标点
            - self.reward_weight['W_VEL_ALIGN'] * vel_to_next_target       # 匹配下一个目标的速度
            - self.reward_weight['W_FINAL_PULL'] * dist_to_final_target    # 最终目标
        )

        # 计算进度奖励 (R_progress)
        # 奖励 = gamma * 当前势能 - 上一步势能
        # 乘以gamma是理论上保证策略不变性的做法，简化版可以省略gamma
        # 加入了穿门时刻不计算势能、只给出穿门奖励的代码
        if phase_switched:
            self.last_potential = current_potential
            R_progress = 0
        else:
            R_progress = current_potential - self.last_potential

        # 成本惩罚 (R_cost)
        R_action_magnitude = -self.reward_weight['W_ACTION_MAG'] * np.linalg.norm(control_signal)
        R_body_rate = -self.reward_weight['W_BODY_RATE'] * np.linalg.norm(current_drone_state[10:])
        R_time_cost = -self.reward_weight['W_TIME_COST']

        # 计算机头对准下一门中心的角度对齐惩罚
        if dist_next_target > 1e-6: # 避免除以零
            target_direction_vec = Q_state[12:15] / dist_next_target
        # print(Q_state[12:15])
        body_forward_vec = np.array([1.0, 0.0, 0.0])
        world_forward_vec = self.quat_rotate_vector(current_drone_state[6:10], body_forward_vec)
        alignment_dot_product = np.dot(world_forward_vec, target_direction_vec)
        # print(world_forward_vec, target_direction_vec)
        R_alignment = self.reward_weight['W_ALIGNMENT'] * ((alignment_dot_product + 1.0) / 2.0) # [0,2]

        # 计算速度矢量指向目标中心的奖励
        current_vel = current_drone_state[3:6]
        vel_norm = np.linalg.norm(current_vel)
        R_vel_dir_align = 0.0
        if vel_norm > 0.1: # 速度太小时方向不稳定，不计算奖励
            vel_dir = current_vel / vel_norm
            vel_align_dot = np.dot(vel_dir, target_direction_vec)
            # 奖励范围 [0, 1] * 权重
            R_vel_dir_align = self.reward_weight['W_VEL_DIR_ALIGN'] * ((vel_align_dot + 1.0) / 2.0)
        # print(f"R_vel_dir_align:{R_vel_dir_align:3f},R_alignment:{R_alignment:3f}, R_action_magnitude:{R_action_magnitude:3f}, R_body_rate:{R_body_rate:3f}, R_time_cost:{R_time_cost:3f}")
        R_cost = R_action_magnitude + R_body_rate + R_time_cost + R_alignment + R_vel_dir_align

        # 事件奖励 (R_event)：终止事件
        R_event = 0
        if collided:
            R_event = self.reward_weight['CRASH_PENALTY']
            self.done = True
            self.info=0
            self.i+=1
        elif dist_to_final_target * 10 < self.target_distance:
            R_event = self.reward_weight['SUCCESS_BONUS']
            self.done = True
            self.info=1
            self.i+=1


        # 最终总奖励
        # print(f"位置进度：{- self.reward_weight['W_POS_PROG'] * dist_next_target:3f}, 速度匹配：{- self.reward_weight['W_VEL_ALIGN'] * vel_to_next_target:3f}, 最终进度：{- self.reward_weight['W_FINAL_PULL'] * dist_to_final_target:3f}")
        # print(f"动作幅度：{R_action_magnitude:3f}, 角速度：{R_body_rate:3f}, 时间：{R_time_cost:3f}, 对准目标:{R_alignment:3f}")
        # print(f"last potential:{self.last_potential:3f}, current_potential:{current_potential:3f}")
        # print(f"进度：{R_progress:3f}, 代价：{R_cost:3f}, 事件：{R_event:3f}, 穿门：{pass_reward:3f}, 目标：{self.phase_idx}")
        reward = (R_progress + R_cost + R_event + pass_reward) / self.reward_weight['REWARD_NORMALIZATION']
        # print(f"reward:{reward:3f}, R_progress:{R_progress:3f}, R_cost:{R_cost:3f}, R_event:{R_event:3f}, pass_reward:{pass_reward:3f}")

        # 更新势能值
        if not phase_switched:
            self.last_potential = current_potential
                
        return current_drone_state, img_tensor, Q_state, reward, self.done, self.phase_idx, self.info, self.elapsed_time,\
                relative_next_target_pos, attitude_9d, relative_next_target_vel, fpv_angular_vel