import config as cfg
from env import env
from CEM_MPC import CEM_MPC
import numpy as np
import itertools
from sac import SAC
import time
from utils import map_value
import math
import os

# 超参数字典
agent_args = {'device': cfg.device, # device
            'critic_param': cfg.critic_param, # Critic (Q)网络构建参数
            'actor_param': cfg.actor_param, # Actor (Pi)网络构建参数
            'SAC_param': cfg.SAC_param, # SAC算法训练参数
            }

args = { # 本页面中经常修改的参数，改完可以直接在本页面右键运行
    'task':'Train', # 测试或训练，Train,Test
    'eval':True, # 训练中是否进行测试 (default: True)
    # 频率相关参数
    'updates_interval': 1, # total_num_step达到value步后进行一组训练（类似PPO效果）
    'updates_per_episode': 1, # 每步对参数更新的次数
    'evaluate_freq': 25, # 训练过程中value个episode之后进行测试
    'evaluate_episode': 5, # 训练过程中插入测试的次数
    'expert_freq': 5, # 训练过程中value个episode进行一次纯MPC示范飞行
    'roll_back': False, # 是否一段时间后开始自动回滚
    'LOAD PARA': False, #是否读取参数
    'load_file': 'master_60_82.17_0.1509_39.12_model', # 需要加载的模型，不管是train还是test都在这改
    'max_episodes':1e6, #测试算法（eval=False）情况下的总步数
    'max_steps': 500, # 每个episode最大步数
    'max_episode': 10000, # 最大训练episode数
    'logs': True, #是否留存训练参数供tensorboard分析
    'logs_folder': './runs/'
    }

# CEM超参数
cem_hyperparams = cfg.CEM_param
# MPC参数
mpc_params = cfg.mpc_params
# env参数
env_params = cfg.env_params

# 初始化
airsim_environment = env(env_params)
# Agent
agent = SAC(agent_args)
MPC_agent = CEM_MPC(cem_hyperparams, mpc_params)
time_start=time.time()
'''Tensorboard使用
显示图像：用cmd（不是vscode的终端） cd到具体存放日志的文件夹（runs），然后tensorboard --logdir=./ --samples_per_plugin scalars=999999999
或者直接在import的地方点那个启动会话
如果还是不行的话用netstat -ano | findstr "6006" 在cmd里查一下6006端口有没有占用，用taskkill全杀了之后再tensorboard一下'''

if args['task']=='Train':
    if args['logs']==True:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args['logs_folder'])
    # Training Loop
    updates = 0
    best_avg_reward = - np.inf
    k = 0
    total_num_steps = 0
    roll_back = args['roll_back']
    if args['LOAD PARA']==True:
        agent.load_model(args['load_file'], evaluate=False)
        # memory.load_buffer("master")
        
    for i_episode in itertools.count(0): #itertools.count(1)用于创建一个无限迭代器。它会生成一个连续的整数序列，从0开始，每次递增1。
        success = False
        episode_reward = 0
        done = False
        episode_steps = 0
        phase_idx = 0
        current_drone_state, final_target_state, waypoints_y, door_z_positions, door_param,\
                 img_tensor, Q_state, final_pi_target, elapsed_time, relative_next_target_pos, attitude_9d, relative_next_target_vel, fpv_angular_vel = airsim_environment.reset()
        MPC_agent.reset(current_drone_state,final_target_state, waypoints_y, door_z_positions, door_param)
        agent.reset()
        while episode_steps <= args['max_steps']:

            # 生成动作
            NN_action = agent.select_action(img_tensor, final_pi_target)  # 输出actor网络动作
            MPC_action = MPC_agent.step(current_drone_state, phase_idx, elapsed_time)

            # 把MPC动作映射到（0，10）
            scaled_MPC_action = map_value(MPC_action, mpc_params['control_min'], mpc_params['control_max'], 
                                          agent_args['actor_param']['scaled_min_action'], agent_args['actor_param']['scaled_max_action'])
            print(f"expert action:{np.round(scaled_MPC_action,4)}, NN action:{np.round(NN_action,4)}")

            # 把NN动作映射回 (0,1)
            rescaled_NN_action = map_value(NN_action, agent_args['actor_param']['scaled_min_action'], agent_args['actor_param']['scaled_max_action'], 
                                           mpc_params['control_min'], mpc_params['control_max'])
            
            # episode数达到freq时进行mpc示范飞行
            if i_episode % args['expert_freq'] == 0:
                next_drone_state, next_img_tensor, next_Q_state,\
                    reward, done, phase_idx, info, elapsed_time,\
                    relative_next_target_pos, attitude_9d, relative_next_target_vel, fpv_angular_vel = airsim_environment.step(MPC_action)
            else: # 进行神经网络控制的DAgger飞行
                next_drone_state, next_img_tensor, next_Q_state,\
                    reward, done, phase_idx, info, elapsed_time,\
                    relative_next_target_pos, attitude_9d, relative_next_target_vel, fpv_angular_vel = airsim_environment.step(rescaled_NN_action)
            
            episode_steps += 1
            episode_reward += reward
            total_num_steps += 1

            # 存储数据
            if math.fabs(scaled_MPC_action[0]) < math.fabs(agent_args['actor_param']['scaled_max_action']) and \
                scaled_MPC_action[0] > agent_args['actor_param']['scaled_min_action']:
                '''expert_memory.push(img_tensor, Q_state, scaled_MPC_action, reward, next_img_tensor, next_Q_state, done, final_pi_target,
                            relative_next_target_pos, attitude_9d, relative_next_target_vel, fpv_angular_vel)'''
                agent.push_data("expert", (img_tensor, Q_state, scaled_MPC_action, NN_action, next_img_tensor, next_Q_state, reward,
            done, final_pi_target, relative_next_target_pos, attitude_9d, relative_next_target_vel, fpv_angular_vel))
                agent.push_data("dagger", (img_tensor, Q_state, scaled_MPC_action, NN_action, next_img_tensor, next_Q_state, reward,
            done, final_pi_target, relative_next_target_pos, attitude_9d, relative_next_target_vel, fpv_angular_vel))
            
            # 这些变量更新会影响数据存入
            current_drone_state = next_drone_state
            img_tensor = next_img_tensor
            Q_state = next_Q_state

            # 训练
            if total_num_steps % args['updates_interval'] == 0:
                for i in range(args['updates_per_episode']):
                    # Update parameters of all the networks
                    '''policy_loss, rl_loss, il_loss, ent_loss, alpha = agent.update_parameters(expert_memory, dagger_memory, exploration_memory, recent_memory, args['batch_size'], updates)'''
                    loss_results = agent.update(updates)
                    policy_loss, qf_loss, rl_loss, il_loss, aux_loss = loss_results
                    if args['logs'] == True and loss_results is not None:
                        writer.add_scalar('loss/policy', policy_loss, updates)
                        writer.add_scalar('loss/qf_loss', qf_loss, updates)
                        writer.add_scalar('loss/rl_loss', rl_loss, updates)
                        writer.add_scalar('loss/il_loss', il_loss, updates)
                        writer.add_scalar('loss/aux_loss', aux_loss, updates)
                    updates += 1
            
            if done:
                # print(f"收集到数据量：{len(memory)}")
                if info:
                    success=True
                break
            
        if args['logs']==True:
            writer.add_scalar('reward/train', episode_reward, i_episode)
        
        if i_episode % args['expert_freq'] == 0:
            print(f"----------------------Episode: {i_episode}, [Expert], steps: {episode_steps}, reward: {round(episode_reward, 2)}, succeed: {success}, updates:{updates}----------------------") #, loss{policy_loss}")
        else: # DAgger
            print(f"----------------------Episode: {i_episode}, [DAgger], steps: {episode_steps}, reward: {round(episode_reward, 2)}, succeed: {success}, updates:{updates}----------------------") #, loss{policy_loss}")

        # 满足条件时进行若干轮测试
        if i_episode % args['evaluate_freq'] == 0 and args['eval'] is True and i_episode > 0: # 测试飞行
            avg_reward = 0.
            episodes = args['evaluate_episode']
            done_num=0
            avg_step = 0
            for _ in range(episodes):
                episode_reward = 0
                done=False
                episode_steps = 0
                success=False
                phase_idx = 0
                current_drone_state, final_target_state, waypoints_y,\
                        door_z_positions, door_param, img_tensor, Q_state, final_pi_target, elapsed_time,\
                             relative_next_target_pos, attitude_9d, relative_next_target_vel, fpv_angular_vel = airsim_environment.reset()
                agent.reset()
                while True:
                    NN_action = agent.select_action(img_tensor, final_pi_target, evaluate=True)  # 开始输出actor网络动作
                    rescaled_NN_action = map_value(NN_action, agent_args['actor_param']['scaled_min_action'], agent_args['actor_param']['scaled_max_action'], 
                                           mpc_params['control_min'], mpc_params['control_max'])
                    # print(f"action:{rescaled_NN_action}")
                    next_drone_state, next_img_tensor, next_Q_state,\
                        reward, done, phase_idx, info, elapsed_time,\
                        relative_next_target_pos, attitude_9d, relative_next_target_vel, fpv_angular_vel = airsim_environment.step(rescaled_NN_action)  # Step
                    episode_reward += reward 
                    
                    current_drone_state = next_drone_state
                    img_tensor = next_img_tensor
                    Q_state = next_Q_state
                    avg_step += 1
                    if info:
                        done_num+=1
                    if done or episode_steps>200:
                        break
                avg_reward += episode_reward
            avg_reward /= episodes
            avg_step /= episodes
            if args['logs']==True:
                writer.add_scalar('avg_reward/test', avg_reward, i_episode)
            if avg_reward >= best_avg_reward and avg_reward >= 0.0:
                best_avg_reward = avg_reward
                agent.save_model("best_master")
            model_name = f'master_{k}_{round(avg_reward,2)}_{round(policy_loss,4)}_{round(avg_step,2)}'
            agent.save_model(model_name)
            k += 1
            print("----------------------------------------")
            print(f"Test Episodes: {episodes}, Avg. Reward: {round(avg_reward, 2)}, success num：{done_num}")
            print("----------------------------------------")
        
        if roll_back:
            if i_episode > 100 and (i_episode % (args['evaluate_freq'] * 20) == 0) and os.path.isfile("best_master_model.pt"): # 大于100轮之后，每20个模型重新加载一次
                agent.load_model("best_master", evaluate=False)

        '''if i_episode == args['max_episode']:
        # if len(memory) == args['replay_size']: # 生成数据集
            # memory.save_buffer("master")
            print("训练结束，{}次仍未完成训练".format(args['max_episode']))
            # if args['logs']==True:
            #     writer.close()
            break'''

if args['task']=='Test':
    name = args['load_file']
    agent.load_model(name.replace('_model', ''))
    time_start = time.time()
    episodes = 100
    done_num = 0
    avg_reward = 0
    for iii in range(episodes):
        episode_reward = 0
        done=False
        episode_steps = 0
        success=False
        phase_idx = 0
        current_drone_state, final_target_state, waypoints_y,\
                        door_z_positions, door_param, img_tensor, Q_state, final_pi_target, elapsed_time,\
                             relative_next_target_pos, attitude_9d, relative_next_target_vel, fpv_angular_vel = airsim_environment.reset()
        agent.reset()
        while True:
            # print(f"true distance:", relative_next_target_pos)
            # print(f"true attitude:", attitude_9d)
            # print(f"true velocity:", relative_next_target_vel)
            # print(f"true angular:", fpv_angular_vel)
            NN_action = agent.select_action(img_tensor, final_pi_target)  # 开始输出actor网络动作
            rescaled_NN_action = map_value(NN_action, args['min_action'], args['max_action'], mpc_params['control_min'], mpc_params['control_max'])
            next_drone_state, next_img_tensor, next_Q_state,\
                reward, done, phase_idx, info, elapsed_time,\
                relative_next_target_pos, attitude_9d, relative_next_target_vel, fpv_angular_vel = airsim_environment.step(rescaled_NN_action)
            episode_reward += reward
            current_drone_state = next_drone_state
            img_tensor = next_img_tensor
            # past_actions = next_past_actions
            Q_state = next_Q_state
            episode_steps += 1
            avg_reward+=reward
            if info:
                success=True
                done_num+=1
            if done or episode_steps>200:
                # if episode_steps >= 50:
                #     model_name = f'master_{k}_{round(episode_reward,2)}_{round(policy_loss,4)}_{episode_steps}'
                #     agent.save_model(model_name)
                #     k += 1
                break
        # print(f"Episode: {iii+1}, reward: {round(episode_reward, 2)}, succeed: {info}")
    avg_reward = avg_reward / episodes
    #writer.add_scalar('avg_reward/test', avg_reward, i_episode)
    time_end=time.time()
    print("----------------------------------------")
    print(f"Model:{name}, Test Episodes: {episodes}, Avg. Reward: {round(avg_reward, 4)},done num:{done_num}")
    print("----------------------------------------")
