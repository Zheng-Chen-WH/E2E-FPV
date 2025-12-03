import config as cfg
from env import env
from CEM_MPC import CEM_MPC
import numpy as np
import itertools
from sac import SAC
from ppo import PPO
import time
from utils import map_value
import math
import os

# 超参数字典
agent_args = {'device': cfg.device, # device
            'critic_param': cfg.critic_param, # Critic (Q)网络构建参数
            'actor_param': cfg.actor_param, # Actor (Pi)网络构建参数
            'SAC_param': cfg.SAC_param, # SAC算法训练参数
            'PPO_param': cfg.PPO_param, # PPO算法训练参数（主要是为了用到clip参数）
            }

args = { # 本页面中经常修改的参数，改完可以直接在本页面右键运行
    'rl_algorithm':'SAC', # 强化学习算法选择，SAC or PPO
    'task':'Train', # 测试或训练，Train,Test
    'eval':True, # 训练中是否进行测试 (default: True)
    # 频率相关参数
    'updates_interval': 10, # total_num_step达到value步后进行一组训练（类似PPO效果）
    'updates_per_episode': 10, # 每次训练对参数更新的次数
    'evaluate_freq': 25, # 训练过程中value个episode之后进行测试;PPO需要大得多的值
    'evaluate_episode': 5, # 训练过程中插入测试的次数
    'expert_freq': 5, # 训练过程中value个episode进行一次纯MPC示范飞行
    'roll_back': False, # 是否一段时间后开始自动回滚
    'LOAD PARA': False, #是否读取参数
    'load_file': 'master_721_13.17_17.165_42.0_model', # 需要加载的模型，不管是train还是test都在这改
    'max_episodes':1e6, #测试算法（eval=False）情况下的总步数
    'max_steps': 500, # 每个episode最大步数
    'max_episode': 10000, # 最大训练episode数
    'logs': True, #是否留存训练参数供tensorboard分析
    'logs_folder': './runs/',
    'test_episode': 200, # Test模式下回合数
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
if args['rl_algorithm']=='PPO':
    agent = PPO(agent_args)
else:
    agent = SAC(agent_args)
MPC_agent = CEM_MPC(cem_hyperparams, mpc_params)
time_start=time.time()
'''Tensorboard使用
显示图像：用cmd（不是vscode的终端） cd到具体存放日志的文件夹（runs），conda activate FPV，
然后tensorboard --logdir=./ --samples_per_plugin scalars=999999999
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
    # PPO需要累计步数来实现达到n_steps后更新
    accumulated_steps = 0
    roll_back = args['roll_back']
    if args['LOAD PARA']==True:
        agent.load_model(args['load_file'], evaluate=False)
        # memory.load_buffer("master")
        
    for i_episode in itertools.count(0):
        success = False
        episode_reward = 0
        done = False
        episode_steps = 0
        phase_idx = 0
        current_drone_state, final_target_state, waypoints_y, door_z_positions, door_param,\
                 img_tensor, critic_state, final_pi_target, elapsed_time, relative_next_target_pos, attitude_9d, relative_next_target_vel, fpv_angular_vel = airsim_environment.reset(seed=i_episode)
        MPC_agent.reset(current_drone_state,final_target_state, waypoints_y, door_z_positions, door_param)
        agent.reset()
        while episode_steps <= args['max_steps']:

            # 生成动作
            NN_action, log_prob, value, hidden_state = agent.select_action(img_tensor, final_pi_target, critic_state)  # 输出actor网络动作
            MPC_action = MPC_agent.step(current_drone_state, phase_idx, elapsed_time)
            # print(hidden_state.shape if hidden_state is not None else None)
            # 把MPC动作映射到神经网络动作空间
            scaled_MPC_action = map_value(MPC_action, mpc_params['control_min'], mpc_params['control_max'], 
                                          agent_args['actor_param']['scaled_min_action'], agent_args['actor_param']['scaled_max_action'])
            # print(f"expert action:{np.round(scaled_MPC_action,4)}, NN action:{np.round(NN_action,4)}")

            # 把NN动作映射回MPC动作空间
            rescaled_NN_action = map_value(NN_action, agent_args['actor_param']['scaled_min_action'], agent_args['actor_param']['scaled_max_action'], 
                                           mpc_params['control_min'], mpc_params['control_max'])
            
            # episode数达到freq时进行mpc示范飞行
            if i_episode % args['expert_freq'] == 0:
                next_drone_state, next_img_tensor, next_critic_state,\
                    reward, done, phase_idx, info, elapsed_time,\
                    relative_next_target_pos, attitude_9d, relative_next_target_vel, fpv_angular_vel = airsim_environment.step(MPC_action)
                # print(attitude_9d.shape)
            else: # 进行神经网络控制的DAgger飞行
                next_drone_state, next_img_tensor, next_critic_state,\
                    reward, done, phase_idx, info, elapsed_time,\
                    relative_next_target_pos, attitude_9d, relative_next_target_vel, fpv_angular_vel = airsim_environment.step(rescaled_NN_action)
            
            episode_steps += 1
            episode_reward += reward
            total_num_steps += 1

            # 存储数据
            if math.fabs(scaled_MPC_action[0]) < math.fabs(agent_args['actor_param']['scaled_max_action']) and \
                scaled_MPC_action[0] > agent_args['actor_param']['scaled_min_action']:
                
                # SAC与PPO均需存入expert和dagger buffer
                agent.push_data("expert", (img_tensor, critic_state, scaled_MPC_action, NN_action, next_img_tensor, next_critic_state, reward,
                    done, final_pi_target, relative_next_target_pos, attitude_9d, relative_next_target_vel, fpv_angular_vel))
                agent.push_data("dagger", (img_tensor, critic_state, scaled_MPC_action, NN_action, next_img_tensor, next_critic_state, reward,
                    done, final_pi_target, relative_next_target_pos, attitude_9d, relative_next_target_vel, fpv_angular_vel))
                # print("expert_buffer",type(img_tensor), img_tensor.shape)
                # PPO将当前步数据存入rollout buffer
                if args['rl_algorithm'] == 'PPO' and log_prob is not None:
                    # rollout buffer需要: (img_seq, V_state, state, action, reward, done, log_prob, value, 
                    #                      pos, rot, vel, ang, hidden_state)
                    # print("rollout_buffer",type(img_tensor), img_tensor.shape)
                    agent.push_data("rollout", (img_tensor, critic_state, final_pi_target, NN_action, reward, done, 
                                                 log_prob, value, relative_next_target_pos, attitude_9d, 
                                                 relative_next_target_vel, fpv_angular_vel, hidden_state))  # hidden_state暂时用None占位
            
            # 这些变量更新会影响数据存入
            current_drone_state = next_drone_state
            img_tensor = next_img_tensor
            critic_state = next_critic_state

            if args['rl_algorithm'] == 'PPO':
                accumulated_steps += 1

            # SAC训练: 每隔固定步数就更新
            if args['rl_algorithm'] == 'SAC' and total_num_steps % args['updates_interval'] == 0:
                for i in range(args['updates_per_episode']):
                    policy_loss, qf_loss, rl_loss, il_loss, aux_loss = agent.update(updates)
                    if args['logs'] == True and policy_loss is not None:
                        writer.add_scalar('loss/policy', policy_loss, updates)
                        writer.add_scalar('loss/qf_loss', qf_loss, updates)
                        writer.add_scalar('loss/rl_loss', rl_loss, updates)
                        writer.add_scalar('loss/il_loss', il_loss, updates)
                        writer.add_scalar('loss/aux_loss', aux_loss, updates)
                    updates += 1
            
            # PPO训练: 在episode结束后，计算returns和advantages，然后更新
            update_condition = (accumulated_steps >= agent_args['PPO_param']['n_steps'] and done)
            if args['rl_algorithm'] == 'PPO' and update_condition:
                accumulated_steps = 0
                # 更新网络
                policy_loss, value_loss, rl_loss, il_loss, aux_loss, n_updates = agent.update(updates)
                
                # 记录日志
                if args['logs'] == True:
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/value', value_loss, updates)
                    writer.add_scalar('loss/rl_loss', rl_loss, updates)
                    writer.add_scalar('loss/il_loss', il_loss, updates)
                    writer.add_scalar('loss/aux_loss', aux_loss, updates)
                updates += n_updates

            if done: 
                # 检查是否成功
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
            test_seeds = [1000000 + i for i in range(episodes)] # 固定测试种子
            done_num=0
            avg_step = 0
            for j, seed in enumerate(test_seeds):
                episode_reward = 0
                done=False
                episode_steps = 0
                success=False
                phase_idx = 0
                current_drone_state, final_target_state, waypoints_y,\
                        door_z_positions, door_param, img_tensor, critic_state, final_pi_target, elapsed_time,\
                             relative_next_target_pos, attitude_9d, relative_next_target_vel, fpv_angular_vel = airsim_environment.reset(seed=seed)
                agent.reset()
                while True:
                    NN_action, _, _, _ = agent.select_action(img_tensor, final_pi_target, critic_state, evaluate=True)  # 开始输出actor网络动作
                    rescaled_NN_action = map_value(NN_action, agent_args['actor_param']['scaled_min_action'], agent_args['actor_param']['scaled_max_action'], 
                                           mpc_params['control_min'], mpc_params['control_max'])
                    # print(f"action:{rescaled_NN_action}")
                    next_drone_state, next_img_tensor, next_critic_state,\
                        reward, done, phase_idx, info, elapsed_time,\
                        relative_next_target_pos, attitude_9d, relative_next_target_vel, fpv_angular_vel = airsim_environment.step(rescaled_NN_action)  # Step
                    episode_reward += reward 
                    
                    current_drone_state = next_drone_state
                    img_tensor = next_img_tensor
                    critic_state = next_critic_state
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
            if updates > 10:
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
    test_seeds = [1000000 + i for i in range(args['test_episode'])] # 固定测试种子
    done_num = 0
    avg_reward = 0
    for iii, seed in enumerate(test_seeds):
        episode_reward = 0
        done=False
        episode_steps = 0
        success=False
        phase_idx = 0
        current_drone_state, final_target_state, waypoints_y,\
                        door_z_positions, door_param, img_tensor, critic_state, final_pi_target, elapsed_time,\
                             relative_next_target_pos, attitude_9d, relative_next_target_vel, fpv_angular_vel = airsim_environment.reset(seed=seed)
        agent.reset()
        while True:
            # print(f"true distance:", relative_next_target_pos)
            # print(f"true attitude:", attitude_9d)
            # print(f"true velocity:", relative_next_target_vel)
            # print(f"true angular:", fpv_angular_vel)
            
            # SAC和PPO兼容处理
            NN_action, _, _, _ = agent.select_action(img_tensor, final_pi_target, critic_state, evaluate=True)  # 开始输出actor网络动作
            
            rescaled_NN_action = map_value(NN_action, agent_args['actor_param']['scaled_min_action'], agent_args['actor_param']['scaled_max_action'], 
                                           mpc_params['control_min'], mpc_params['control_max'])
            next_drone_state, next_img_tensor, next_critic_state,\
                reward, done, phase_idx, info, elapsed_time,\
                relative_next_target_pos, attitude_9d, relative_next_target_vel, fpv_angular_vel = airsim_environment.step(rescaled_NN_action)
            episode_reward += reward
            current_drone_state = next_drone_state
            img_tensor = next_img_tensor
            # past_actions = next_past_actions
            critic_state = next_critic_state
            episode_steps += 1
            avg_reward+=reward
            if done or episode_steps>200:
                if info:
                    success=True
                    done_num+=1
                print(f"Episode: {iii+1} succeed: {success} reward:{round(episode_reward, 2)} success rate: {done_num/(iii+1)}")
                break
        # print(f"Episode: {iii+1}, reward: {round(episode_reward, 2)}, succeed: {info}")
    avg_reward = avg_reward / args['test_episode']
    #writer.add_scalar('avg_reward/test', avg_reward, i_episode)
    time_end=time.time()
    print("----------------------------------------")
    print(f"Model:{name}, Test Episodes: {args['test_episode']}, Avg. Reward: {round(avg_reward, 4)},done num:{done_num}")
    print("----------------------------------------")
