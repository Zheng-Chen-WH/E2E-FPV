"""
 airsim simpleflight速度控制四旋翼飞圆形
"""
import airsim
import numpy as np
import math
import time

client = airsim.MultirotorClient()  # connect to the AirSim simulator
client.enableApiControl(True)       # 获取控制权
client.armDisarm(True)              # 解锁
client.takeoffAsync().join()        # 起飞
client.moveToZAsync(-3, 1).join()   # 第二阶段：上升到2米高度

center = np.array([[0], [5]])    # 圆心设置
speed = 1                        # 速度设置
radius = 5                       # 半径设置
clock_wise = True                # 顺时针或逆时针设置

pos_reserve = np.array([[0.], [0.], [-3.]])

# 速度控制
for i in range(5000):
    # 获取无人机当前真实位置
    state = client.simGetGroundTruthKinematics()
    pos = np.array([[state.position.x_val], [state.position.y_val], [state.position.z_val]])
    # 计算径向速度的方向向量
    dp = pos[0:2] - center #二维向量，自然指向圆心
    if np.linalg.norm(dp) - radius > 0.1:
        vel_dir_1 = -dp
    elif np.linalg.norm(dp) - radius < 0.1:
        vel_dir_1 = dp
    # 计算切向速度的方向向量
    theta = math.atan2(dp[1, 0], dp[0, 0]) #就是[1][0]的简写表达
    if clock_wise:
        theta += math.pi / 2
    else:
        theta -= math.pi / 2
    v_dir_2 = np.array([[math.cos(theta)], [math.sin(theta)]])
    # 计算最终速度的方向向量
    v_dir = 0.02 * vel_dir_1 + v_dir_2 #加权使轨迹更圆润，径向不缩小一点轨迹会变成锯齿
    # 计算最终速度指令
    v_cmd = speed * v_dir/np.linalg.norm(v_dir)
    # 速度控制
    client.moveByVelocityZAsync(v_cmd[0, 0], v_cmd[1, 0], -3, 1) # z方向高度为3
    # 画图
    point_reserve = [airsim.Vector3r(pos_reserve[0, 0], pos_reserve[1, 0], pos_reserve[2, 0])]
    point = [airsim.Vector3r(pos[0, 0], pos[1, 0], pos[2, 0])] # 位置点
    point_end = pos + np.vstack((v_cmd, np.array([[0]]))) # 下一时刻位置微分
    point_end = [airsim.Vector3r(point_end[0, 0], point_end[1, 0], point_end[2, 0])] # 需要转成Vector3r格式
    client.simPlotArrows(point, point_end, arrow_size=8.0, color_rgba=[0.0, 0.0, 1.0, 1.0]) # 速度矢量画箭头
    client.simPlotLineList(point_reserve+point, color_rgba=[1.0, 0.0, 0.0, 1.0], is_persistent=True) # 画出实际轨迹（上一位置到本时刻位置；is persistent不会消失
    # 循环
    pos_reserve = pos
    time.sleep(0.02)