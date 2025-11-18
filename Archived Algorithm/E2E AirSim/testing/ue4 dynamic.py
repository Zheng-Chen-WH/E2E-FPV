import airsim
import time
import math
from datetime import datetime
import os
from PIL import Image
import numpy as np

# 连接到AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)       # 获取控制权
client.armDisarm(True)              # 解锁
client.takeoffAsync().join()        # 起飞
client.moveToZAsync(-1, 1).join()   # 第二阶段：上升到1米高度
save_dir = "/media/zheng/A214861F1485F697/Dataset"  # 图像保存路径
image_type = airsim.ImageType.Scene  # 图像类型（RGB）

# 获取当前实际时间
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# 设置仿真时间为实际时间
client.simSetTimeOfDay(True, start_datetime=current_time, celestial_clock_speed=1)

# 门框的名称（确保与UE4中的名称一致）
door_frames=["BP_Doorframe","BP_Doorframe2","BP_Doorframe3","BP_Doorframe4"]
initial_positions=[]

# 获取门框的初始位姿
for i in door_frames:
    try:
        initial_pose = client.simGetObjectPose(i)
        initial_position = initial_pose.position
        print(f"Initial Position: X={initial_position.x_val}, Y={initial_position.y_val}, Z={initial_position.z_val}")
        initial_positions.append(initial_position.x_val)
    except Exception as e:
        print(f"Error: {e}") 
        print(f"请确保场景中存在名为 '{i}' 的对象。")
        exit()

# 门框正弦运动运动参数
amplitude = 2  # 运动幅度（米）
frequency = 1  # 运动频率（Hz）
duration = 20.0  # 运动总时间（秒）

# 开始时间
start_time = time.time()

# 控制门框运动
while time.time() - start_time < duration:
    # 计算当前时间
    t = time.time() - start_time
    deviation=0.0 # 不同门框错开
    for ii in range(len(door_frames)):
        pose = client.simGetObjectPose(door_frames[ii])
        position = pose.position
        # 计算门框的新位置
        new_x = initial_positions[ii] + amplitude * math.sin(2 * math.pi * frequency * t + deviation)
        new_y = position.y_val
        new_z = position.z_val

        # 设置门框的新位置
        new_position = airsim.Vector3r(new_x, new_y, new_z)
        client.simSetObjectPose(door_frames[ii], airsim.Pose(new_position, initial_pose.orientation), True)
        deviation+=2.0 # 不同门相位不同
    
    #无人机运动
    client.moveToPositionAsync(0.0,49.0,-1.0,5.0)

    # 拍照
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, pixels_as_float=False, compress=False)])
    if responses:
        # 将 AirSim 的字节流转换为 NumPy 数组
        img_data = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        # 重塑为正确的图像形状 (Height x Width x 3 Channels)
        img_rgb = img_data.reshape(responses[0].height, responses[0].width, 3)
        # 转换 BGR 到 RGB（AirSim 默认返回 BGR 格式）
        img_rgb = img_rgb[..., ::-1]  # BGR → RGB
        # 使用 Pillow 保存为 PNG
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(save_dir, f"RGB_{timestamp}.png")
        
        Image.fromarray(img_rgb).save(filename)

    # 等待下一帧
    time.sleep(0.03)

client.landAsync().join()#降落无人机，60s不落地直接执行下一行
client.armDisarm(False)#锁上旋翼
client.enableApiControl(False)#交出控制权

print("Door Frame Movement Completed.")

client.reset()
time.sleep(1)
for ii in range(len(door_frames)):
        pose = client.simGetObjectPose(door_frames[ii])
        position = pose.position
        # 计算门框的新位置
        new_x = initial_positions[ii]
        new_y = position.y_val
        new_z = position.z_val

        # 设置门框的新位置
        new_position = airsim.Vector3r(new_x, new_y, new_z)
        client.simSetObjectPose(door_frames[ii], airsim.Pose(new_position, initial_pose.orientation), True)


    