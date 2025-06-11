import numpy as np
import airsim
from datetime import datetime
import time
import math
from pymavlink import mavutil

'''启动px4:PX4-Autopilot文件夹下执行make px4_sitl_default none'''
"""用mavlink向px4发布指令"""

class AirSimPx4Env:
    def __init__(self, cfg): # 门框的名称（确保与UE4中的名称一致）
        # 连接到AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        # 获取当前实际时间并设置为仿真时间
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.client.simSetTimeOfDay(True, start_datetime=current_time_str, celestial_clock_speed=1)

        self.mavlink_conn_str = cfg.mavlink_connection_string
        self.target_system = cfg.mavlink_target_system
        self.target_component = cfg.mavlink_target_component
        self.drone = None
        self._connect_mavlink()

        self.DT = cfg.DT
        self.door_frames = cfg.door_frames_names
        self.initial_pose = None # Will be set during reset for door orientation

        # 门框正弦运动运动参数
        self.door_param =  cfg.door_param

        self.start_time = 0 # To be set at the beginning of each episode
        self.takeoff_altitude_px4 = cfg.takeoff_altitude
        self.control_mode_px4 = cfg.control_mode_px4.upper()
        self.px4_vehicle_name_in_airsim = cfg.px4_vehicle_name_in_airsim

    def _connect_mavlink(self):
        print(f"Connecting to MAVLink via: {self.mavlink_conn_str}")
        try:
            self.drone = mavutil.mavlink_connection(self.mavlink_conn_str,
                            source_system=255,  # 指定当前这个脚本的MAVLink系统ID。
                            autoreconnect=True, # Pymavlink库会在连接意外断开时尝试自动重新连接。
                            source_component=mavutil.mavlink.MAV_COMP_ID_MISSIONPLANNER) # 指定当前脚本的组件ID。一个MAVLink系统可以有多个组件。
                                                                                         # MAV_COMP_ID_MISSIONPLANNER (值为190) 是一个预定义的常量，表示这个脚本扮演的角色类似于一个任务规划器/地面站软件。
            # 等待从连接的MAVLink设备（例如PX4飞控）接收心跳消息
            # 心跳消息是MAVLink设备周期性发送的，用来表明其在线并报告其基本状态。
            self.drone.wait_heartbeat(timeout=10)
            print(f"MAVLink Heartbeat received from system {self.drone.target_system}, component {self.drone.target_component}")
            # self.drone.target_system 和 self.drone.target_component 会被自动填充为
            # 发送心跳的那个MAVLink设备（即无人机飞控）的系统ID和组件ID。
            self.target_system = self.drone.target_system # 从cfg默认值更新
            self.target_component = self.drone.target_component
        except Exception as e:
            print(f"MAVLink connection failed: {e}")
            self.drone = None
            raise ConnectionError(f"MAVLink connection failed: {e}")
    
    def _send_mavlink_command_long(self, command, params, confirmation=0, timeout=3):

        """   
        command: 要发送的MAVLink命令的ID (例如 mavutil.mavlink.MAV_CMD_NAV_TAKEOFF)。
        params: 一个包含7个浮点数参数的列表或元组，这些参数的具体含义取决于 'command' 的值。
                MAV_CMD_COMMAND_LONG 消息固定有7个参数字段。
        confirmation: COMMAND_LONG消息中的'confirmation'字段。通常第一次发送时为0。
                      飞控可能会用这个字段来处理重发或序列化命令。
        timeout: 等待命令应答 (COMMAND_ACK) 的最长时间（秒）。"""

        if not self.drone:
            print("MAVLink drone not connected or connection lost.")
            return False
        try:
            # Pymavlink 的 mav 对象 (self.drone.mav) 提供了便捷的方法来构造和发送各种MAVLink消息。
            # command_long_send 会自动打包这些参数到一个MAVLink COMMAND_LONG消息包中并通过已建立的连接发送出去。
            self.drone.mav.command_long_send(
                self.target_system, # 目标系统的ID (通常是飞控的System ID, 例如 1)
                self.target_component, # 目标组件的ID (通常是飞控主组件的Component ID, 例如 1)
                command, # 要发送的具体命令ID
                confirmation, # 命令的确认号
                params[0], params[1], params[2], params[3], params[4], params[5], params[6] # MAV_CMD_COMMAND_LONG 消息的固定七个参数字段
            )
            ack_msg_type = 'COMMAND_ACK' # 期望收到的应答消息类型是 COMMAND_ACK

            # self.drone.recv_match() 是一个强大的Pymavlink函数，用于接收符合特定条件的消息。
            # 如果在timeout时间内收到了匹配的COMMAND_ACK消息，ack变量会是该消息对象；否则为None。
            ack = self.drone.recv_match(type=ack_msg_type, # 只匹配类型为 'COMMAND_ACK' 的消息。
                                        blocking=True, # 设置为True，表示这个函数会阻塞（暂停）程序的执行，直到匹配到消息或超时。
                                        timeout=timeout) # 使用函数参数中指定的超时时间。
            
            # 为了打印更友好的日志，获取命令的名称
            # mavutil.mavlink.enums 包含了MAVLink协议中定义的各种枚举值（包括命令ID和结果代码）的名称。
            # 如果命令ID在枚举中，就取其名称，否则直接用数字ID。
            cmd_name = mavutil.mavlink.enums['MAV_CMD'][command].name if command in mavutil.mavlink.enums['MAV_CMD'] else str(command)
            if ack: # 如果ack不是None
                if ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    # COMMAND_ACK消息中有一个'result'字段，表示命令执行的结果。
                    # MAV_RESULT_ACCEPTED (值为0) 表示命令已被飞控接受。
                    # print(f"MAVLink Command {cmd_name} accepted.")
                    return True
                else:
                    # 如果结果不是ACCEPTED，说明命令可能被拒绝或执行失败。
                    # 获取结果代码的名称，用于打印日志。
                    res_name = mavutil.mavlink.enums['MAV_RESULT'][ack.result].name if ack.result in mavutil.mavlink.enums['MAV_RESULT'] else str(ack.result)
                    print(f"MAVLink Command {cmd_name} rejected with result: {res_name}")
                    return False
            print(f"MAVLink Command {cmd_name} ACK not received.") # 如果ack是None，意味着等待超时，没有收到预期的COMMAND_ACK。
            return False
        except Exception as e:
            print(f"Error sending MAVLink command_long for {mavutil.mavlink.enums['MAV_CMD'][command].name if command in mavutil.mavlink.enums['MAV_CMD'] else command}: {e}")
            return False

    def _set_px4_mode(self, mode_string): # mode_string: 一个字符串，表示期望设置的飞行模式的名称（例如 "POSCTL", "OFFBOARD"）。
        if not self.drone:
            print("MAVLink drone not connected or connection lost.")
            return False

        # 定义PX4特定模式名称到其MAVLink自定义模式数值的映射
        # PX4使用MAVLink的“自定义模式”字段来表示其具体的飞行模式。
        # 这个字典将用户友好的模式字符串映射到PX4内部使用的数字。
        px4_custom_mode_map = {
            'MANUAL': 1, # 手动模式
            'ALTCTL': 2, # 高度控制模式
            'POSCTL': 3, # 位置控制模式 (常用)
            'AUTO.MISSION': 4, # 自动任务模式
            'AUTO.LOITER': 5, # 自动悬停/留待模式
            'AUTO.RTL': 6, # 自动返航模式
            'ACRO': 7, # 特技模式
            'OFFBOARD': 8, # 离线控制模式 (用于外部计算机控制)
            'STABILIZED': 9, # 增稳模式
            'AUTO.TAKEOFF': 10, # 自动起飞模式
            'AUTO.LAND': 11, # 自动降落模式
        }
        mode_str_upper = mode_string.upper() # 将输入的模式字符串转换为大写，以便不区分大小写地匹配。

        if mode_str_upper in px4_custom_mode_map:
            custom_mode = px4_custom_mode_map[mode_str_upper]
            # 构造 MAV_CMD_DO_SET_MODE 命令所需的 base_mode 参数
            # base_mode 是一个位掩码，包含了一些基本的模式标志。
            # 对于PX4的自定义模式，MAV_MODE_FLAG_CUSTOM_MODE_ENABLED 必须设置。
            # MAV_MODE_FLAG_STABILIZE_ENABLED 通常也作为基础，表示期望有姿态稳定。
            # 使用 '|' (按位或) 操作符来组合这些标志。
            """按位或操作符 | 是一个二元操作符，它对两个整数的每一个对应的二进制位进行操作。
            如果两个位中至少有一个是1，那么结果位就是1；否则，结果位是0。
            在这些场景中的作用：
            开启/设置多个选项：将多个独立的标志位（每个通常只有一个bit为1）合并到一个整数中。
            创建复合条件：每个被 | 连接的常量都贡献了它所代表的那个特定的位。
            高效紧凑：用一个整数就能传递丰富的信息，这对于带宽有限的无人机通信来说很重要。
            mavlink中大量状态是以某一位为1、其他位为0的二进制数表示的，通过按位或可以实现表示多种状态"""
            base_mode = mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED | mavutil.mavlink.MAV_MODE_FLAG_STABILIZE_ENABLED
            if self.drone.motors_armed():
                # 如果已解锁，也应在base_mode中包含 MAV_MODE_FLAG_SAFETY_ARMED 标志。
                # 这表明请求模式切换的源头知道飞机是解锁的。
                base_mode |= mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED

            print(f"Attempting to set PX4 mode: {mode_str_upper} (CustomMode: {custom_mode}, BaseMode: {base_mode})")
            
            # 发送 MAV_CMD_DO_SET_MODE 命令
            # 这是一个 COMMAND_LONG 类型的命令。
            self.drone.mav.command_long_send(
                self.target_system, self.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE, # 命令ID：设置模式
                0, # confirmation (通常为0) 
                base_mode, # 参数1: 计算得到的 base_mode 
                custom_mode, # 参数2: PX4的自定义模式数值 
                0,0,0,0,0
            )
            
            ack = self.drone.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
            if ack and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                print(f"Mode change to {mode_str_upper} acknowledged by COMMAND_ACK.")

                # (重要步骤) 通过心跳包进一步验证模式是否真的改变
                # 有时飞控可能会接受命令，但由于某些条件不满足，实际模式并未改变或很快恢复。
                # HEARTBEAT消息会包含飞控当前的实际飞行模式。
                verify_start_time = time.time()
                while time.time() - verify_start_time < 3: # 在3秒内尝试验证
                    msg = self.drone.recv_match(type='HEARTBEAT', blocking=False, timeout=0.1) # 非阻塞地尝试获取心跳包
                    if msg: # 如果收到心跳包：
                        current_flight_mode_str = "" # 用于存储从心跳包解析出的当前模式字符串

                        # 尝试从心跳包的 custom_mode 和 base_mode 反向解析出模式字符串
                        for k,v in px4_custom_mode_map.items():
                            # PX4在心跳包中通过 msg.custom_mode 报告其模式，
                            # 并且 msg.base_mode 中应包含 MAV_MODE_FLAG_CUSTOM_MODE_ENABLED 标志。
                            if v == msg.custom_mode and (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED):
                                current_flight_mode_str = k
                                break
                        if current_flight_mode_str == mode_str_upper:
                            print(f"Mode successfully changed to {mode_str_upper} (verified by HEARTBEAT: {current_flight_mode_str}).")
                            return True
                    time.sleep(0.1)
                print(f"Mode change to {mode_str_upper} ACKed, but HEARTBEAT did not confirm specific custom mode string in time. Current HB flightmode: {self.drone.flightmode}")
                return True
            elif ack:
                print(f"Mode change to {mode_str_upper} rejected with result: {ack.result}")
                return False
            else:
                print(f"Mode change to {mode_str_upper} ACK not received.")
                return False
        else:
            print(f"Mode '{mode_string}' not in defined PX4 custom mode map.")
            return False

    def _move_door(self, door_frame_name, position): 
        """将门移动到指定x,y,z位置的辅助函数, 保持初始姿态，名字前加_使其只能在类内部被调用"""
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

    def get_drone_state_mavlink(self):
        if not self.drone:
            return np.zeros(12)
        # NaN (Not a Number) 用于标记数据尚未获取。
        pos_ned = np.array([np.nan]*3) # 位置 (北, 东, 下（NED坐标系）)
        vel_ned = np.array([np.nan]*3) # 速度 (北向, 东向, 下向)
        attitude_rad = np.array([np.nan]*3) # 姿态 (俯仰, 横滚, 偏航 - 弧度)
        angular_vel_rad_s = np.array([np.nan]*3) # 角速度 (俯仰速率, 横滚速率, 偏航速率 - 弧度/秒)
        
        # 尝试接收 LOCAL_POSITION_NED 消息
        # 这个消息包含无人机在本地NED坐标系下的位置和速度信息。
        # blocking=False: 非阻塞接收，即如果没有立即收到消息，不会一直等待。
        # timeout=0.02: 等待消息的最短时间（20毫秒）。
        msg_pos = self.drone.recv_match(type='LOCAL_POSITION_NED', blocking=False, timeout=0.02)
        if msg_pos:
            pos_ned = np.array([msg_pos.x, msg_pos.y, msg_pos.z])
            vel_ned = np.array([msg_pos.vx, msg_pos.vy, msg_pos.vz])

        # 尝试接收 ATTITUDE 消息
        # 这个消息包含无人机的姿态（欧拉角）和角速度信息。
        msg_att = self.drone.recv_match(type='ATTITUDE', blocking=False, timeout=0.02)
        if msg_att:
            attitude_rad = np.array([msg_att.pitch, msg_att.roll, msg_att.yaw])
            angular_vel_rad_s = np.array([msg_att.pitchspeed, msg_att.rollspeed, msg_att.yawspeed])
        
        # 如果没有接收到对应信息，np.nan_to_num() 会将数组中的NaN替换为0.0。
        pos_ned = np.nan_to_num(pos_ned)
        vel_ned = np.nan_to_num(vel_ned)
        attitude_rad = np.nan_to_num(attitude_rad)
        angular_vel_rad_s = np.nan_to_num(angular_vel_rad_s)
        return np.concatenate((pos_ned, vel_ned, attitude_rad, angular_vel_rad_s))
    
    def _send_acceleration_command(self, acceleration_xyz_body):
        """ 这是一个辅助函数，用于在OFFBOARD模式下发送加速度指令。 """
        if not self.drone or self.control_mode_px4 != "OFFBOARD":
            # 必须满足条件：
            # a. MAVLink已连接 (self.drone is not None)
            # b. 飞控当前的控制模式必须是 "OFFBOARD" (self.control_mode_px4 == "OFFBOARD")，OFFBOARD模式允许外部计算机发送设定点。
            if self.control_mode_px4 != "OFFBOARD":
                print("Cannot send acceleration: Not in OFFBOARD mode.")
            else:
                print("Cannot send acceleration: MAVLink drone not connected.")
            return False
        
        # 构建 type_mask (类型掩码)
        # type_mask 是一个非常重要的位掩码，它告诉飞控 SET_POSITION_TARGET_LOCAL_NED 消息中的哪些字段是有效的，哪些应该被忽略。
        ax, ay, az = acceleration_xyz_body[0], acceleration_xyz_body[1], acceleration_xyz_body[2]
        type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |   # 忽略X位置设定点
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |   # 忽略Y位置设定点
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |   # 忽略Z位置设定点
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE |  # 忽略VX速度设定点
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |  # 忽略VY速度设定点
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |  # 忽略VZ速度设定点
            # AX, AY, AZ 是被控制的 (即它们对应的 IGNORE 标志位没有被设置)
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_ANGLE_IGNORE | # 忽略偏航角设定点
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE # 忽略偏航角速率设定点
        )
        try:
            self.drone.mav.set_position_target_local_ned_send(
                0, # time_boot_ms: 消息发送时的启动时间毫秒数 (通常设为0，由飞控填充或用于流式设定点)
                self.target_system, self.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED, # 坐标系: MAV_FRAME_BODY_OFFSET_NED表示提供的加速度 (ax, ay, az) 是在无人机的机体坐标系下（前-右-下）
                type_mask, # 上面定义的类型掩码
                0, 0, 0,    # x, y, z 位置设定点 (因为type_mask中已忽略，这些值无效)
                0, 0, 0,    # vx, vy, vz velocity (ignored)
                float(ax), float(ay), float(az), # afx, afy, afz 加速度设定点 (单位: m/s^2)
                0,          # yaw angle (ignored)
                0           # yaw_rate (ignored)
            )
            return True # 假设指令已成功发送。SET_POSITION_TARGET 通常不直接产生COMMAND_ACK。
        except Exception as e:
            print(f"Error sending MAVLink set_position_target_local_ned (for acceleration): {e}")
            return False 

    def reset(self):
        # 状态重置与初始化
        try:
            self.client.reset()
            time.sleep(0.2)
            self.client.enableApiControl(True, vehicle_name=self.px4_vehicle_name_in_airsim)
            print("AirSim simulation reset. API control enabled for AirSim client.")
        except Exception as e:
            raise RuntimeError(f"Failed to reset AirSim client: {e}")

        if not self.drone:
            self._connect_mavlink()
        if not self.drone:
             raise ConnectionError("MAVLink not connected after reset. Cannot proceed.")
        while self.drone.recv_match(blocking=False): pass

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
            np.random.uniform(-5, 5),    # 目标位置x
            np.random.uniform(38, 42),   # 目标位置y
            np.random.uniform(-2, -1),   # 目标位置z
            4.0, 0.0, 0.0,               # 目标速度x, y, z
            0.0, 0.0, 0.0,               # 目标姿态pitch, roll, yaw
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

        print("Preparing PX4 drone via MAVLink...")
        if not self._set_px4_mode("POSCTL"):
            print("Warning: Failed to set PX4 to POSCTL mode before arming.")
        time.sleep(0.5)

        armed = self._send_mavlink_command_long(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            params=[1, 0, 0, 0, 0, 0, 0], timeout=5
        )
        if not armed:
            raise RuntimeError("Failed to arm drone via MAVLink.")
        print("Drone armed via MAVLink.")
        time.sleep(1)

        print(f"Commanding MAVLink takeoff to {self.takeoff_altitude_px4}m...")
        takeoff_params = [0, 0, 0, float('nan'), float('nan'), float('nan'), self.takeoff_altitude_px4]
        took_off = self._send_mavlink_command_long(
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, params=takeoff_params, timeout=5
        )
        if not took_off:
            print("MAVLink Takeoff command failed or was rejected.")
        else:
            print("MAVLink Takeoff command accepted. Monitoring altitude (simplified wait)...")
            time.sleep(max(5.0, self.takeoff_altitude_px4 * 1.5)) # Adjusted wait time
            print("Assumed takeoff complete.")

        if self.control_mode_px4 == "OFFBOARD":
            if not self._set_px4_mode("OFFBOARD"):
                 raise RuntimeError(f"Critical: Failed to set OFFBOARD mode.")
            print(f"Drone mode set to '{self.control_mode_px4}'.")
            print("OFFBOARD mode active. Sending initial neutral acceleration setpoint.")
            if not self._send_acceleration_command(np.array([0.0, 0.0, 0.0])): # Send zero acceleration
                print("Warning: Failed to send initial zero acceleration command in OFFBOARD mode.")

        self.start_time = time.time()
        self._update_door_positions(0.0)
        self.door_param["start_time"] = self.start_time

        current_drone_state = self.get_drone_state_mavlink()
        
        return (current_drone_state, self.final_target_state, self.waypoints_y,
                self.door_z_positions, np.array(self.door_current_x_positions), self.door_x_velocities,
                self.start_time, self.door_param)

    def step(self, control_action_xyz_acceleration_body):
        if not self.drone:
            print("MAVLink drone not connected. Cannot send control action.")
            dummy_state = self.get_drone_state_mavlink()
            return dummy_state, np.array(self.door_current_x_positions), self.door_x_velocities, False

        if self.control_mode_px4 == "OFFBOARD":
            if not self._send_acceleration_command(control_action_xyz_acceleration_body):
                print("Warning: Failed to send acceleration command during step.")
        else:
            print(f"Warning: Drone not in OFFBOARD mode (current: {self.drone.flightmode if self.drone else 'N/A'}). Acceleration commands may not be effective.")

        time.sleep(self.DT)

        elapsed_time = time.time() - self.start_time
        self._update_door_positions(elapsed_time) # 更新门位置

        current_drone_state = self.get_drone_state_mavlink()
        collision_info = self.client.simGetCollisionInfo()
        
        collided = False
        # 碰撞时间需要大于一个小阈值，避免起飞碰撞被判定为碰撞
        if collision_info.has_collided and (collision_info.time_stamp / 1e9 > self.start_time + 0.5) :
            collided = True

        return current_drone_state, np.array(self.door_current_x_positions), self.door_x_velocities, collided
    
    def close(self):
        print("Closing AirSimPx4Env...")
        if self.drone:
            print("Disarming drone via MAVLink...")
            self._send_mavlink_command_long(
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                params=[0, 0, 0, 0, 0, 0, 0], timeout=3
            )
            self.drone.close()
            print("MAVLink connection closed.")
        try:
            self.client.reset()
            self.client.enableApiControl(False, vehicle_name=self.px4_vehicle_name_in_airsim)
            print("AirSim client reset and API control disabled.")
        except Exception as e:
            print(f"Error during AirSim client cleanup: {e}")