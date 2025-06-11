# python=3.8 虚拟环境不可安装jupyter 单位是米
# 大地系x北y东 机体系x前y右
# 默认的simpleflight飞控，需要给出航路点/速度
# 通过异步不join实现简单的避障（垂直往上飞）与实时状态输出

import airsim
import os
import pprint
import time
import numpy as np
import datetime

# 建立与 AirSim 模拟器的连接
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True) #允许通过代码控制
client.armDisarm(True) #无人机解锁
# 获取当前实际时间
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# 设置仿真时间为实际时间
client.simSetTimeOfDay(True, start_datetime=current_time, celestial_clock_speed=1)


def flyto(fpv,target,velo=10.0):
    fpv.moveToPositionAsync(target[0], target[1], target[2], velo)

def breakpoint(fpv,target,takeoff_time,threshold=1.0):
    last_collision_time=takeoff_time
    state=fpv.getMultirotorState()
    state=state.kinematics_estimated.position
    position=np.array([state.x_val,state.y_val,state.z_val])
    print(position)
    collide=False
    collision_info = fpv.simGetCollisionInfo()
    if collision_info.has_collided and collision_info.time_stamp > last_collision_time:
        last_collision_time = collision_info.time_stamp
        collide=True
    if np.linalg.norm(target-position)<threshold and not collide:
        return 1
    if np.linalg.norm(target-position)>threshold and collide:
        fpv.moveToZAsync(-20,1).join()
        flyto(fpv,target)

# 以 5 m/s 的速度起飞并将无人机移动到坐标(-10, 10, -10)
client.takeoffAsync().join() # 起飞无人机，join表示任务执行完后执行下一个任务，否则立即执行下一行代码

takeoff_time = client.simGetCollisionInfo().time_stamp #记录碰撞时间戳，避免把起飞时刻碰撞认为发生了碰撞

timestep=1.0

client.moveToZAsync(-10,1).join()#以1m/s的速度移动到Z轴-3的位置，z轴指向地下
position_target=np.array([-200.0,100.0,-2.0])
flyto(client, position_target)
# 以10m/s的速度移动到(-20,10,-2)的位置，z轴指向地下
# lookahead 和 adaptive_lookahead 设置当四旋翼飞轨迹的时候的朝向
# drivetrain控制偏航角模式（是否总朝向前或总朝向速度方向
while True:
    if breakpoint(client,position_target,takeoff_time=takeoff_time): #碰撞或者到达目标时暂停
        break
    else:
        time.sleep(timestep)

client.moveByVelocityAsync(10,0,0,15).join() #x轴10m/s的速度飞15s，由于非线性系统，速度控制对位置有误差
client.hoverAsync().join() #悬停模式
time.sleep(2) # 时间暂停多久就悬停多久


client.landAsync().join()#降落无人机，60s不落地直接执行下一行
client.armDisarm(False)#锁上旋翼
client.enableApiControl(False)#交出控制权