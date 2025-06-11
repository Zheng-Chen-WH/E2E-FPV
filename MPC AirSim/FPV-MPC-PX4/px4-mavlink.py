'''启动px4:PX4-Autopilot文件夹下执行make px4_sitl_default none'''
"""用mavlink向px4发布指令"""

from pymavlink import mavutil
import time
import airsim
import numpy as np

target_system=0
while target_system==0:
    # 连接到 PX4 SITL
    master = mavutil.mavlink_connection('udp:127.0.0.1:14550') #px4用14550接收指令

    # 等待心跳消息
    print("Waiting for heartbeat...")
    master.wait_heartbeat()
    target_system=master.target_system
    print("Heartbeat received from system (system_id=%u component_id=%u)" % (
        master.target_system, master.target_component))

airsim_env = airsim.MultirotorClient()
airsim_env.confirmConnection()
airsim_env.reset()
time.sleep(5)
# 设置飞行模式为 Offboard
master.set_mode("OFFBOARD")

# ARM 无人机
master.arducopter_arm()
print("Armed!")

# 持续发送位置指令，让 PX4 保持在 Offboard 模式
def send_position():
    master.mav.set_position_target_local_ned_send( #NED坐标系下给出位置目标
        0,  # time_boot_ms (not used)
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b111111111000,  # 控制掩码，每位代表启用不同值，1为不启用，0为启用，0b表示二进制编码，12位分别表示航向角速率，航向角，加速度表示加速度/力，ax,ay,az,vx,vy,vz,x,y,z
        0, 0, -3,  # x, y, z positions (meters)
        0, 0, 0,  # x, y, z velocity in m/s (not used)
        0, 0, 0,  # x, y, z acceleration (not used)
        0, 0  # yaw, yaw_rate (not used)
    )

def send_velocity():
    """
    发送速度指令到PX4
    :param vx: 东向速度（m/s）
    :param vy: 北向速度（m/s）
    :param vz: 垂直速度（m/s），负值表示向下，正值表示向上
    """
    master.mav.set_position_target_local_ned_send(
        0,  # 时间戳（可以设置为0）
        master.target_system,  # 系统ID
        master.target_component,  # 组件ID
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # 使用LOCAL_NED坐标系
        0b111111000111,  # type_mask（忽略位置x, y, z，只启用速度vx, vy, vz）
        0, 0, 0,  # x, y, z位置（忽略）
        -1, 2, 0,  # x, y, z速度
        0, 0, 0,  # x, y, z加速度（忽略）
        0, 0  # yaw和yaw_rate（忽略）
    )

def get_drone_state():
        # 获取无人机状态
        fpv_state_raw = airsim_env.getMultirotorState()

        # 获取位置
        position = fpv_state_raw.kinematics_estimated.position
        fpv_pos = np.array([position.x_val, position.y_val, position.z_val])

        # 获取线速度
        linear_velocity = fpv_state_raw.kinematics_estimated.linear_velocity
        fpv_vel = np.array([linear_velocity.x_val, linear_velocity.y_val, linear_velocity.z_val])

        # 获取姿态角 (俯仰pitch, 滚转roll, 偏航yaw, 欧拉角表示, 弧度制)
        orientation_q = fpv_state_raw.kinematics_estimated.orientation
        pitch, roll, yaw = airsim.to_eularian_angles(orientation_q) # # 将四元数转换为欧拉角 (radians)
        fpv_attitude = np.array([roll, pitch, yaw])
        # roll_deg = math.degrees(roll)
        # pitch_deg = math.degrees(pitch)
        # yaw_deg = math.degrees(yaw)
        # fpv_attitude = np.array([pitch_deg, roll_deg, yaw_deg])

        # 获取角速度
        angular_velocity = fpv_state_raw.kinematics_estimated.angular_velocity
        fpv_angular_vel = np.array([angular_velocity.x_val, angular_velocity.y_val, angular_velocity.z_val])

        return np.concatenate((fpv_pos, fpv_vel, fpv_attitude, fpv_angular_vel))

def send_acceleration():
    master.mav.set_position_target_local_ned_send(
        # 注意除了AFX AFY AFZ 设置为 0 表示不忽略以外，还需要将 FORCE 置为 0。否则，PX4 将误认为 AFX AFY AFZ 是推力矢量。
        0,  # 时间戳（可以设置为0）
        master.target_system,  # 系统ID
        master.target_component,  # 组件ID
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # 使用LOCAL_NED坐标系
        0b110000111111,  # type_mask（忽略位置x, y, z，只启用速度vx, vy, vz）
        0, 0, 0,  # x, y, z位置（忽略）
        0, 0, 0,  # x, y, z速度
        0, 2, -0.5,  # x, y, z加速度（忽略）
        0, 0  # yaw和yaw_rate（忽略）
    )
   

# 持续发送指令
state=get_drone_state()
print(state)
for i in range(50): #先起飞
    send_position()
    print(state)
    time.sleep(0.1)
for i in range(50):
    send_acceleration()
    time.sleep(0.1)