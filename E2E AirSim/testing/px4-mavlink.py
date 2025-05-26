'''启动px4:PX4-Autopilot文件夹下执行make px4_sitl_default none'''
"""用mavlink向px4发布指令"""

from pymavlink import mavutil
import time

# 连接到 PX4 SITL
master = mavutil.mavlink_connection('udp:127.0.0.1:14550') #px4用14550接收指令

# 等待心跳消息
print("Waiting for heartbeat...")
master.wait_heartbeat()
print("Heartbeat received from system (system_id=%u component_id=%u)" % (
    master.target_system, master.target_component))

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
        0, 0, -10,  # x, y, z positions (meters)
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
        1, 2, 0.5,  # x, y, z加速度（忽略）
        0, 0  # yaw和yaw_rate（忽略）
    )
   

# 持续发送指令
for i in range(100): #先起飞
    send_position()
    time.sleep(0.1)
while True:
    send_acceleration()
    time.sleep(0.1)