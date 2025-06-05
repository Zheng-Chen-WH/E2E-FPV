import numpy as np
import airsim
from datetime import datetime
import time
import math
import config as cfg

class AirSimEnv:
    def __init__(self, cfg): # 门框的名称（确保与UE4中的名称一致）
        # 连接到AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        # 获取当前实际时间并设置为仿真时间
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.client.simSetTimeOfDay(True, start_datetime=current_time_str, celestial_clock_speed=1)

        self.DT = cfg.DT
        self.door_frames = cfg.door_frames_names
        self.initial_pose = None # Will be set during reset for door orientation

        # 门框正弦运动运动参数
        self.door_param =  cfg.door_param
        self.start_time = 0 # To be set at the beginning of each episode

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

    def get_drone_state(self):
        # 获取无人机状态
        fpv_state_raw = self.client.getMultirotorState()

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
        
        # roll_deg = math.degrees(roll)
        # pitch_deg = math.degrees(pitch)
        # yaw_deg = math.degrees(yaw)
        # fpv_attitude = np.array([pitch_deg, roll_deg, yaw_deg])

        # 获取角速度
        angular_velocity = fpv_state_raw.kinematics_estimated.angular_velocity
        fpv_angular_vel = np.array([angular_velocity.x_val, angular_velocity.y_val, angular_velocity.z_val])

        return np.concatenate((fpv_pos, fpv_vel, fpv_attitude, fpv_angular_vel))

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
            np.random.uniform(48, 52),   # 目标位置y
            np.random.uniform(-3, -2),   # 目标位置z
            0.0, 0.0, 0.0,               # 目标速度x, y, z
            0.707, 0.0, 0.0, 0.707,              # 目标姿态四元数
            0.0, 0.0, 0.0                # 目标角速度x, y, z
        ])
        self.waypoints_y.append(self.final_target_state[1])

        # 设置目标点视觉标记物（橙球）
        target_ball_pos = airsim.Vector3r(self.final_target_state[0], self.final_target_state[1], self.final_target_state[2])
        try:
            ball_initial_pose = self.client.simGetObjectPose("OrangeBall_Blueprint")
            self.client.simSetObjectPose("OrangeBall_Blueprint", airsim.Pose(target_ball_pos, ball_initial_pose.orientation), True)
        except Exception as e:
            print(f"Warning: Could not set pose for OrangeBall_Blueprint: {e}")

        self.client.takeoffAsync().join()
        time.sleep(0.5)

        self.start_time = time.time()
        self._update_door_positions(0.0)
        self.door_param["start_time"] = self.start_time

        current_drone_state = self.get_drone_state()
        self.start_time_step=time.time()

        collision_info = self.client.simGetCollisionInfo()
        self.first_collide_time=collision_info.time_stamp / 1e9
        return (current_drone_state, self.final_target_state, self.waypoints_y,
                self.door_z_positions, np.array(self.door_current_x_positions), self.door_x_velocities,
                self.start_time, self.door_param)

    def step(self, control_signal):
        # 发送速度指令
        # self.client.moveByVelocityAsync(
        #     float(control_action_xyz_velocity[0]),
        #     float(control_action_xyz_velocity[1]),
        #     float(control_action_xyz_velocity[2]),
        #     duration= 2 * self.DT # duration改到2*DT
        # )
        # 发送油门指令
        end_time=time.time()
        print("calculation time consumed:", end_time-self.start_time_step)
        self.client.simPause(False)
        self.client.moveByMotorPWMsAsync(float(control_signal[0]),float(control_signal[1]),float(control_signal[2]),float(control_signal[3]), self.DT*2)
        time.sleep(self.DT) # 仿真持续步长

        elapsed_time = time.time() - self.start_time
        self._update_door_positions(elapsed_time) # 更新门位置
        self.client.simPause(True)
        self.start_time_step=time.time()

        current_drone_state = self.get_drone_state()
        # print(f"airsim仿真环境, {current_drone_state[0:3]},速度,{current_drone_state[3:6]},姿态四元数{current_drone_state[6:10]},角速度{current_drone_state[10:13]}")
        # print("————————————————————————————————————")
        collision_info = self.client.simGetCollisionInfo()
        
        collided = False
        # 碰撞时间需要大于一个小阈值，避免起飞碰撞被判定为碰撞
        if collision_info.has_collided and (collision_info.time_stamp / 1e9 > self.first_collide_time + 0.5) :
            collided = True
        return current_drone_state, np.array(self.door_current_x_positions), self.door_x_velocities, collided