import numpy as np
import torch
import time
import math
import config as cfg
from airsim_env import AirSimEnv
from analytical_model import SimpleFlightDynamics

# 环境初始化
# 初始化
airsim_env = AirSimEnv(cfg)
(current_true_state, final_target_state, waypoints_y,
    door_z_positions, door_x_positions, door_x_velocities,
    episode_start_time, door_parameters_dict) = airsim_env.reset()

for step_idx in range(5):
    # AirSim执行指令
    next_true_state, _, _, collided = airsim_env.step([0,0,0])
    if collided:
        break