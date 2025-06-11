import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
import os
import math
from scipy.optimize import fsolve
from sac import SAC
import torch.nn as nn
import airsim
import time
from datetime import datetime
import os
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
    def __init__(self,args):
        # 连接到AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.save_dir = "/media/zheng/A214861F1485F697/Dataset"  # 图像保存路径
        self.image_type = airsim.ImageType.Scene  # 图像类型（RGB）
        # 获取当前实际时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 设置仿真时间为实际时间
        self.client.simSetTimeOfDay(True, start_datetime=current_time, celestial_clock_speed=1)
        # 状态空间设置为无限连续状态空间，虽然不知道相比设成离散空间有什么影响
        self.action_space=spaces.Box(low=np.array([-5,0.0,-0.5]), high=np.array([5.0,5.0,0.5]), shape=(3,), dtype=np.float32)  # 动作为速度指令
        self.pi_observation_space=spaces.Box(-np.inf,np.inf,shape=(12,),dtype=np.float32) # 过去三次动作+目标
        self.q_observation_space=spaces.Box(-np.inf,np.inf,shape=(15,),dtype=np.float32) # 无人机角速度，相对下一个门的位置、速度、相对目标的位置,目标位置
        self.target_distance=1
        self.i=0
        self.info=0
        self.gamma=0.5
        self.sigma=np.degrees(0.001)
        self.mu=math.degrees(0.005)
        # 门框的名称（确保与UE4中的名称一致）
        self.door_frames=args['door_frames']
        self.frames=args['num_frames']
        # 图像预处理
        # 将图像转换为模型可接受的格式，同时调整尺寸
        self.transform = transforms.Compose([
            transforms.Resize((256, 144)),  # 调整图像尺寸为模型输入的尺寸
            transforms.ToTensor(),          # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])
        self.time_sleep=0.0025
        # 门框正弦运动运动参数
        self.amplitude = 3  # 运动幅度（米）
        self.frequency = 0.1  # 运动频率（Hz）
        # self.duration = 15.0  # 运动总时间（秒）
    
    def N_dis(self,mean):
        sample = np.random.normal(loc=mean, scale=self.sigma, size=1)
        return sample.item()+math.degrees(self.mu)

    def movedoor(self, door_frame, x, y, z):
        new_pose = airsim.Vector3r(x, y, z)
        self.client.simSetObjectPose(door_frame, airsim.Pose(new_pose, self.initial_pose.orientation), True)
    
    def state_generate(self):
        # 获取无人机状态
        state = self.client.getMultirotorState()
        # 获取位置
        position = state.kinematics_estimated.position
        relative_pos_target=np.array([position.x_val-self.target_pos[0],
                                     position.y_val-self.target_pos[1],
                                     position.z_val-self.target_pos[2]])
        # 获取线速度
        linear_velocity = state.kinematics_estimated.linear_velocity
        # 获取角速度
        angular_velocity = state.kinematics_estimated.angular_velocity
        ww=np.array([angular_velocity.x_val,angular_velocity.y_val,angular_velocity.z_val])
        relative_pos=None
        for i,value in enumerate(self.initial_door_y_positions):
            if position.y_val>value:
                door_position=self.client.simGetObjectPose(self.door_frames[i]).position #下一个门
                relative_pos=np.array([position.x_val-door_position.x_val,
                                      position.y_val-door_position.y_val,
                                      position.z_val-door_position.z_val])
                relative_speed=np.array([linear_velocity.x_val-self.door_x_velocity[i],
                                        linear_velocity.y_val,
                                        linear_velocity.z_val])
        if relative_pos is None:
            door_position=self.client.simGetObjectPose(self.door_frames[0]).position #没有穿越任意一个门时用第一个门算
            relative_pos=np.array([position.x_val-door_position.x_val,
                                    position.y_val-door_position.y_val,
                                    position.z_val-door_position.z_val])
            relative_speed=np.array([linear_velocity.x_val-self.door_x_velocity[0],
                                    linear_velocity.y_val,
                                    linear_velocity.z_val])
        return np.concatenate((ww, relative_pos, relative_speed, relative_pos_target))

        

    def reset(self,seed=None): 
        self.client.reset()
        self.client.enableApiControl(True)       # 获取控制权

        if seed is None:
            np.random.seed(self.i)
        else:
            np.random.seed(seed)

        self.client.armDisarm(True)              # 无人机解锁

        # 定义无人机的初始位置和方向
        initial_drone_position = airsim.Vector3r(np.random.uniform(-3,3), np.random.uniform(-5,5), -0.5)  # 定义位置 (x=10, y=20, z=-0.5)
        yaw = math.radians(90)  # 90 度（朝向正 y 轴）

        # 创建 Pose 对象
        initial_drone_pose = airsim.Pose(initial_drone_position, airsim.to_quaternion(0, 0, yaw))
        # 设置无人机初始位置
        self.client.simSetVehiclePose(initial_drone_pose, ignore_collision=True)
        
        self.client.moveByVelocityAsync(0, 0, -0.2, 0.5).join()        # 起飞
        collision_info = self.client.simGetCollisionInfo()
        self.takeoff_time=time.time()

        # 获取门框的初始位姿并随机修改
        self.initial_door_x_positions=[]
        self.initial_door_y_positions=[]
        self.door_x_positions=[]
        self.door_x_velocity=np.array([0,0,0,0])
        for ii, i in enumerate(self.door_frames):
            try:
                self.initial_pose = self.client.simGetObjectPose(i)
                self.initial_door_position = self.initial_pose.position
                # print(f"Initial Position: X={initial_position.x_val}, Y={initial_position.y_val}, Z={initial_position.z_val}")
                new_x=0+np.random.uniform(-1,1)
                new_y=(ii+1)*10+np.random.uniform(-2,2)
                self.movedoor(i,new_x,new_y,self.initial_door_position.z_val)
                self.initial_door_x_positions.append(new_x)
                self.door_x_positions.append(new_x)
                self.initial_door_y_positions.append(new_y)
            except Exception as e:
                print(f"Error: {e}") 
                print(f"请确保场景中存在名为 '{i}' 的对象。")
                exit()
        self.target_pos=np.array([np.random.uniform(-5,5),
                                 np.random.uniform(45,55),
                                 np.random.uniform(-2,-0.5)])
        new_ball_pose = airsim.Vector3r(self.target_pos[0], self.target_pos[1], self.target_pos[2])
        self.client.simSetObjectPose("OrangeBall_Blueprint", airsim.Pose(new_ball_pose, self.client.simGetObjectPose("OrangeBall_Blueprint")), True)
        
        # 开始时间
        self.start_time = time.time()
        input_sequence = []
        for k in range(self.frames):
            # 计算当前时间
            t = time.time() - self.start_time
            # deviation=0.0 # 不同门框错开
            for ii,value in enumerate(self.door_frames):
                pose = self.client.simGetObjectPose(value) #相当于把门的状态存在ue4里，每次用的时候直接读取
                position = pose.position
                # 计算门框的新位置
                new_x = self.initial_door_x_positions[ii] + self.amplitude * math.sin(2 * math.pi * self.frequency * t)# + deviation)
                # print(self.initial_door_x_positions[ii])
                self.door_x_velocity[ii]=(new_x-self.door_x_positions[ii])/self.time_sleep
                self.door_x_positions[ii]=new_x

                new_y = position.y_val
                new_z = position.z_val

                # 设置门框的新位置
                self.movedoor(value,new_x,new_y,new_z)
                # deviation+=1.0 # 不同门相位不同

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
                input_sequence.append(img_tensor)  # 添加到序列

            # 等待下一帧
            time.sleep(self.time_sleep)

        # 将序列堆叠为 Tensor (num_frames, channels, height, width)
        input_sequence = torch.stack(input_sequence, dim=0)  # 输出维度: (4, 3, 256, 144)

        # 添加 Batch 维度 (batch_size, num_frames, channels, height, width)
        input_sequence = input_sequence.unsqueeze(0)  # 输出维度: (1, 4, 3, 256, 144)

        self.info=0
        self.collision_count=0
        self.done=False
        q_state=self.state_generate()
        self.initial_distance_target=np.linalg.norm(q_state[9:12])
        self.pi_state=np.array([0,0,-0.2,0,0,-0.2,0,0,-0.2])
        pi_img=input_sequence
        
        return pi_img, self.pi_state, q_state, self.target_pos

    def step(self, action):#,red_action

        # 发送速度指令
        self.client.moveByVelocityAsync(float(action[0]), float(action[1]), float(action[2]), 1.0) 
        input_sequence = []
        for k in range(self.frames):
            # 开始时间
            # 计算当前时间
            t = time.time() - self.start_time
            deviation=0.0 # 不同门框错开
            for ii,value in enumerate(self.door_frames):
                pose = self.client.simGetObjectPose(value) #相当于把门的状态存在ue4里，每次用的时候直接读取
                position = pose.position
                # 计算门框的新位置
                new_x = self.initial_door_x_positions[ii] + self.amplitude * math.sin(2 * math.pi * self.frequency * t + deviation)
                self.door_x_velocity[ii]=(new_x-self.door_x_positions[ii])/self.time_sleep
                self.door_x_positions[ii]=new_x

                new_y = position.y_val
                new_z = position.z_val

                # 设置门框的新位置
                self.movedoor(value,new_x,new_y,new_z)
                deviation+=2.0 # 不同门相位不同

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
                input_sequence.append(img_tensor)  # 添加到序列

            # 等待下一帧
            time.sleep(self.time_sleep)

        # 将序列堆叠为 Tensor (num_frames, channels, height, width)
        input_sequence = torch.stack(input_sequence, dim=0)  # 输出维度: (4, 3, 256, 144)

        # 添加 Batch 维度 (batch_size, num_frames, channels, height, width)
        input_sequence = input_sequence.unsqueeze(0)  # 输出维度: (1, 4, 3, 256, 144)


        q_state=self.state_generate()
        self.pi_state=np.concatenate((self.pi_state[3:],np.array(action)))
        pi_img=input_sequence

        self.distance_nextdoor=np.linalg.norm(q_state[3:6])
        collision_info = self.client.simGetCollisionInfo()
        if  collision_info.time_stamp/1e9 >self.takeoff_time or np.linalg.norm(q_state[9:12])<self.target_distance:
            self.done=True
        
        if not self.done:
            reward= (self.initial_distance_target-np.linalg.norm(q_state[9:12]))/5 -1 # - self.distance_nextdoor/10 #与下个门中心的距离、与目标的距离

        if self.done:
            if collision_info.time_stamp/1e9 >self.takeoff_time:
                self.info=0 #用来区分碰撞和到达目标
                reward=0
            elif np.linalg.norm(q_state[9:12])<self.target_distance and not collision_info.time_stamp/1e9 >self.takeoff_time:
                self.info=1
                # print(self.red_pos-self.blue_pos,self.red_vel-self.blue_vel)
                reward=200
            self.i+=1

        return pi_img, self.pi_state, q_state, reward, self.done, self.info

    # def plot(self, args, data_x, data_y, data_z=None):
    #     font = {'family': 'serif',
    #      'serif': 'Times New Roman',
    #      'weight': 'normal',
    #      'size': 15,
    #      }
    #     plt.rc('font', **font)
    #     plt.style.use('seaborn-whitegrid')
    #     if data_z!=None and args['plot_type']=="3D-1line":
    #         fig = plt.figure()
    #         ax = fig.gca(projection='3d') #Axes对象是图形的绘图区域，可以在其上进行各种绘图操作。
    #                                     #通过gca()函数可以获取当前图形的Axes对象，如果该对象不存在，则会创建一个新的。
    #                                     #projection='3d' 参数指定了Axes对象的投影方式为3D，即创建一个三维坐标系。
    #         plt.plot(0,0,0,'r*') #画一个位于原点的星形
    #         plt.plot(data_x,data_y,data_z,'b',linewidth=1) #画三维图
    #         ax.set_xlabel('x/km', fontsize=15)
    #         ax.set_ylabel('y/km', fontsize=15)
    #         ax.set_zlabel('z/km', fontsize=15)
    #         ax.set_xlim(np.min(data_x),np.max(data_x))
    #         ax.set_ylim(np.min(data_y),np.max(data_y))
    #         ax.set_zlim(np.min(data_z),np.max(data_z))
    #         # ax.set_xlim(np.min([np.min(data_x),np.min(data_y),np.min(data_z)]),np.max([np.max(data_x),np.max(data_y),np.max(data_z)]))
    #         # ax.set_ylim(np.min([np.min(data_x),np.min(data_y),np.min(data_z)]),np.max([np.max(data_x),np.max(data_y),np.max(data_z)]))
    #         # ax.set_zlim(np.min([np.min(data_x),np.min(data_y),np.min(data_z)]),np.max([np.max(data_x),np.max(data_y),np.max(data_z)]))
    #         plt.tight_layout()# 调整布局使得图像不溢出
    #         plt.savefig('svg.svg', format='svg', bbox_inches='tight')# 'logs/{}epoch-{}steps.png'.format(epoch,steps))
    #         plt.show()
    #     elif data_z!=None and args['plot_type']=="2D-2line":
    #         fig = plt.figure()
    #         ax = fig.gca() #Axes对象是图形的绘图区域，可以在其上进行各种绘图操作。通过gca()函数可以获取当前图形的Axes对象，如果该对象不存在，则会创建一个新的。
    #         plt.plot(data_x,data_y,'b',linewidth=0.5)
    #         plt.plot(data_x,data_z,'g',linewidth=1)
    #         ax.set_xlabel('x', fontsize=15)
    #         ax.set_ylabel('y', fontsize=15)
    #         ax.set_xlim(np.min(data_x),np.max(data_x))
    #         ax.set_ylim(np.min([np.min(data_y),np.min(data_z)]),np.max([np.max(data_y),np.max(data_z)]))
    #         plt.tight_layout()# 调整布局使得图像不溢出
    #         plt.savefig(args['plot_title'], format='svg', bbox_inches='tight')# 'logs/{}epoch-{}steps.png'.format(epoch,steps))
    
    # def plotstep(self,blue_action):
    #     state=[]
    #     o1,o2=self.angle_observe(self.blue_pos,self.blue_vel,self.red_pos,self.red_vel,-self.interval)
    #     state.append(o1)
    #     state.append(o2)
    #     o5,o6=self.angle_observe(self.blue_pos,self.blue_vel,self.red_pos,self.red_vel,0)
    #     state.append(o5)
    #     state.append(o6)
    #     state.append(blue_action[0])
    #     state.append(blue_action[1])
    #     state.append(blue_action[2])
    #     self.blue_vel=self.blue_vel+blue_action[0:3]
    #     o9,o10=self.angle_observe(self.blue_pos,self.blue_vel,self.red_pos,self.red_vel,self.interval)
    #     state.append(o9)
    #     state.append(o10)
    #     middle_red_pos,middle_red_vel=CW_Prop(self.red_pos,self.red_vel,self.omega,self.interval) #下一时刻前300s机动，另一种随机给机动是本时刻后300s给机动，这个不行就试另一种
    #     #if self.i%4==0:
    #     # red_impulse=np.array([np.random.uniform(-0.5,0.5),np.random.uniform(-0.05,0.05),np.random.uniform(-0.5,0.5)])
    #     # if self.i%4==1:
    #     red_impulse=self.opt_escape(self.red_pos,self.red_vel,self.blue_pos,self.blue_vel,blue_action[3]-self.interval)
    #     # if self.i%4==2:
    #     # red_impulse=self.gamma/np.linalg.norm(self.blue_vel)*self.blue_vel
    #     # if self.i%4==3:
    #     # red_impulse=self.pos_escape(self.red_pos,self.blue_pos)
    #     alpha=np.random.normal(0,2*math.pi)
    #     beta=np.random.normal(0,2*math.pi)
    #     # red_impulse=np.array([self.gamma*math.cos(alpha)*math.cos(beta),self.gamma*math.sin(beta),self.gamma*math.sin(alpha)*math.cos(beta)]) #乱跑
    #     self.blue_pos,self.blue_vel=CW_Prop(self.blue_pos,self.blue_vel,self.omega,blue_action[3])
    #     self.red_pos,self.red_vel=CW_Prop(middle_red_pos,middle_red_vel+red_impulse,self.omega,blue_action[3]-self.interval)
    #     # self.blue_pos_list.append(next_blue_pos)
    #     self.fuel=np.array([self.fuel[0]-np.linalg.norm(blue_action[0:3])])
    #     state.append(self.fuel[0])
    #     state.append(blue_action[3]/1e3)
    #     state=np.array(state)
    #     # self.appr_distance=np.linalg.norm(self.blue_pos-appr_red_pos) #不考虑目标航天器-300s机动时的目标终端位置
    #     self.distance=np.linalg.norm(self.red_pos-self.blue_pos)

    #     if self.fuel[0]<0 or self.distance<=self.target_distance:
    #         self.done=True

    #     if self.done:
    #         if self.fuel[0]<0:
    #             self.info=0 #用来区分越界和到达目标
    #         elif self.distance<=self.target_distance and self.fuel[0]>=0:
    #             self.info=1
    #         self.i+=1

    #     return state, self.done, self.info, self.red_pos, self.blue_pos, self.red_vel, self.blue_vel




    