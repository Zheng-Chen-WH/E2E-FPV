import matplotlib.pyplot as plt
import torch.optim as optim
import config as cfg
from env import env
from CEM_MPC import CEM_MPC
import datetime
import numpy as np
import itertools
import torch.nn as nn
from sac import SAC
from replay_memory import HERMemory,ReplayMemory
import time
import matplotlib.pyplot as plt
import torch

# 超参数字典
args={'eval':True, # Evaluates a policy a policy every 10 episode (default: True)
        'gamma':0.99, # discount factor for reward (default: 0.99)
        'tau':0.2, # target smoothing coefficient(τ) (default: 0.005) 参数tau定义了目标网络软更新时的平滑系数，
                     # 它控制了新目标值在更新目标网络时与旧目标值的融合程度。
                     # 较小的tau值会导致目标网络变化较慢，从而增加了训练的稳定性，但也可能降低学习速度。
        'lr':cfg.LEARNING_RATE, # learning rate (default: 0.0003)
        'alpha':0.2, # Temperature parameter α determines the relative importance of the entropy\term against the reward (default: 0.2)
        'automatic_entropy_tuning':False, # Automaically adjust α (default: False)
        'batch_size':cfg.BATCH_SIZE, # batch size (default: 256)
        'max_steps':cfg.MAX_STEP_PER_EPISODE, # 每个episode最大步数
        'hidden_sizes':cfg.NN_HIDDEN_SIZE, # 隐藏层大小，带有激活函数的隐藏层层数等于这一列表大小
        'updates_per_episode':cfg.NN_TRAIN_EPOCHS_PER_STEP, # model updates per simulator step (default: 1) 每步对参数更新的次数
        'start_steps':1000, # Steps sampling random actions (default: 10000) 在开始训练之前完全随机地进行动作以收集数据
        'target_update_interval':10, # Value target update per no. of updates per step (default: 1) 目标网络更新的间隔
        'replay_size':cfg.BUFFER_SIZE, # size of replay buffer (default: 10000000)
        'cuda':True, # run on CUDA (default: False)
        'LOAD PARA':False, #是否读取参数
        'task':'Train', # 测试或训练或画图，Train,Test,Plot
        'activation':nn.ReLU, #激活函数类型
        'plot_type':'2D-2line', #'3D-1line'为三维图，一条曲线；'2D-2line'为二维图，两条曲线
        'plot_title':'reward-steps.svg',
        'max_episodes':1e6, #测试算法（eval=False）情况下的总步数
        'evaluate_freq':50, #训练过程中每多少个epoch之后进行测试
        'seed':20000323, #网络初始化的时候用的随机数种子  
        'max_epoch':10000,
        'logs':True, #是否留存训练参数供tensorboard分析 
        'embedding_dim':128,
        'num_frames':4,
        'door_frames':cfg.door_frames_names,
        'max_action':cfg.SCALED_CONTROL_MAX,
        'min_action':cfg.SCALED_CONTROL_MIN,
        'warm_up':cfg.WARM_UP}

cem_hyperparams = {
    'prediction_horizon': cfg.PREDICTION_HORIZON,
    'n_samples': cfg.N_SAMPLES_CEM,
    'n_elites': cfg.N_ELITES_CEM,
    'n_iter': cfg.N_ITER_CEM,
    'initial_std': cfg.INITIAL_STD_CEM,
    'min_std': cfg.MIN_STD_CEM,
    'alpha': cfg.ALPHA_CEM,
}

mpc_params = {
    'waypoint_pass_threshold_y': cfg.WAYPOINT_PASS_THRESHOLD_Y,
    'dt': cfg.DT,
    'control_max': cfg.CONTROL_MAX,
    'control_min': cfg.CONTROL_MIN,
    'q_state_matrix_gpu':cfg.Q_STATE_COST_MATRIX_GPU,
    'r_control_matrix_gpu':cfg.R_CONTROL_COST_MATRIX_GPU,
    'q_terminal_matrix_gpu':cfg.Q_TERMINAL_COST_MATRIX_GPU,
    'q_state_matrix_gpu_two':cfg.Q_STATE_COST_MATRIX_GPU_TWO,
    'r_control_matrix_gpu_two':cfg.R_CONTROL_COST_MATRIX_GPU,
    'q_terminal_matrix_gpu_two':cfg.Q_TERMINAL_COST_MATRIX_GPU_TWO,
    'static_q_state_matrix_gpu':cfg.STATIC_Q_STATE_COST_MATRIX_GPU,
    'static_r_control_matrix_gpu':cfg.STATIC_R_CONTROL_COST_MATRIX_GPU,
    'static_q_terminal_matrix_gpu':cfg.STATIC_Q_TERMINAL_COST_MATRIX_GPU,
    'action_dim':cfg.ACTION_DIM,
    'state_dim':cfg.MPC_STATE_DIM,
    'device':cfg.device,
    'pos_tolerence':cfg.POS_TOLERANCE
}

env_params={'DT':cfg.DT,
            'img_time':cfg.img_time,
            'door_frames':cfg.door_frames_names,
            'door_param':cfg.door_param,
            'POS_TOLERANCE':cfg.POS_TOLERANCE,
            'frames':cfg.NUM_TRANSFORMER_FRAMES,
            'pass_threshold_y': cfg.WAYPOINT_PASS_THRESHOLD_Y,
            'control_max': cfg.CONTROL_MAX,
            'control_min': cfg.CONTROL_MIN}

# 初始化
airsim_environment = env(env_params)
# Agent
agent = SAC(cfg.PI_STATE_DIM, cfg.Q_STATE_DIM, cfg.ACTION_DIM, args)
MPC_agent = CEM_MPC(cem_hyperparams, mpc_params)
time_start=time.time()
#Tensorboard
#显示图像：用cmd（不是vscode的终端） cd到具体存放日志的文件夹（runs），然后tensorboard --logdir=./ --samples_per_plugin scalars=999999999
#或者直接在import的地方点那个启动会话
#如果还是不行的话用netstat -ano | findstr "6006" 在cmd里查一下6006端口有没有占用，用taskkill全杀了之后再tensorboard一下

# Memory
memory = ReplayMemory(args['replay_size'])
# memory = HERMemory(args['replay_size'],"Final") # HER没修好

# 记录列表
# all_episode_steps = []
# all_episode_target_reached_flags = []
# all_episode_avg_losses = []
# last_episode_detailed_data = None # For detailed logging of the final episode if needed

if args['task']=='Train':
    if args['logs']==True:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter('./runs/')
    # Training Loop
    updates = 0
    best_avg_reward=0
    total_numsteps=0
    steps_list=[]
    episode_reward_list=[]
    avg_reward_list=[]
    k=0
    if args['LOAD PARA']==True:
        agent.load_model("master", evaluate=False)
        memory.load_buffer("master")
        
    for i_episode in itertools.count(1): #itertools.count(1)用于创建一个无限迭代器。它会生成一个连续的整数序列，从1开始，每次递增1。
        success=False
        episode_reward = 0
        done=False
        episode_steps = 0
        phase_idx = 0
        current_drone_state, final_target_state, waypoints_y, door_z_positions, door_param,\
                 img_tensor, past_actions, Q_state, final_pi_target, elapsed_time = airsim_environment.reset()
        MPC_agent.reset(current_drone_state,final_target_state, waypoints_y, door_z_positions, door_param)
        while episode_steps <= args['max_steps']:
            # NN_action = agent.select_action(img_tensor, np.concatenate((past_actions, final_pi_target)))  # 开始输出actor网络动作
            MPC_action = MPC_agent.step(current_drone_state, phase_idx, elapsed_time)
            next_drone_state, next_img_tensor, next_past_actions, next_Q_state,\
                  reward, done, phase_idx, info, elapsed_time = airsim_environment.step(MPC_action)  # Step
            # print("real state:", next_drone_state)
            episode_steps += 1
            episode_reward += reward #没用gamma是因为在sac里求q的时候用了
            total_numsteps += 1
            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            
            scaled_MPC_action = 2 * (MPC_action - mpc_params['control_min']) / (mpc_params['control_max'] - mpc_params['control_min']) - 1
            memory.push(img_tensor, past_actions, Q_state, scaled_MPC_action, \
                        reward, next_img_tensor, next_past_actions, next_Q_state, done, final_pi_target, info)
            
            current_drone_state = next_drone_state
            img_tensor = next_img_tensor
            past_actions = next_past_actions
            Q_state = next_Q_state

            if done:
                if len(memory) > args['batch_size']:
                    # Number of updates per step in environment 每次交互之后可以进行多次训练...
                    for i in range(args['updates_per_episode']):
                        # Update parameters of all the networks
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args['batch_size'], updates)
                        # 清理显存
                        torch.cuda.empty_cache()
                        if args['logs']==True:
                            writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                            writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                            writer.add_scalar('loss/policy', policy_loss, updates)
                            # print(policy_loss)
                            writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                            writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                        updates += 1
                steps_list.append(i_episode)
                episode_reward_list.append(episode_reward)
                if len(episode_reward_list)>=500:
                    avg_reward_list.append(sum(episode_reward_list[-500:])/500)
                else:
                    avg_reward_list.append(sum(episode_reward_list)/len(episode_reward_list))
                if info:
                    success=True
                break
        if args['logs']==True:
            writer.add_scalar('reward/train', episode_reward, i_episode)
        if len(memory) > args['batch_size']:
            print(f"Episode: {i_episode}, steps: {episode_steps}, reward: {round(episode_reward, 2)}, succeed: {success}, loss{policy_loss}")
        # round(episode_reward,2) 对episode_reward进行四舍五入，并保留两位小数

        '''保留测试环节'''
        if i_episode % args['evaluate_freq'] == 0 and args['eval'] is True: #评价上一个训练过程
            avg_reward = 0.
            episodes = 5
            done_num=0
            for _ in range(episodes):
                episode_reward = 0
                done=False
                episode_steps = 0
                success=False
                phase_idx = 0
                current_drone_state, final_target_state, waypoints_y,\
                        door_z_positions, door_param, img_tensor, past_actions, Q_state, final_pi_target, elapsed_time = airsim_environment.reset()
                while True:
                    NN_action = agent.select_action(img_tensor, np.concatenate((past_actions, final_pi_target)))  # 开始输出actor网络动作

                    scaled_NN_action = (NN_action + 1) / 2 * (mpc_params['control_max'] - mpc_params['control_min']) + mpc_params['control_min']
                    next_drone_state, next_img_tensor, next_past_actions, next_Q_state,\
                        reward, done, phase_idx, info, elapsed_time = airsim_environment.step(MPC_action)  # Step
                    episode_reward += reward 

                    memory.push(img_tensor, past_actions, Q_state, NN_action, \
                                reward, next_img_tensor, next_past_actions, next_Q_state, done, final_pi_target, info)
                    
                    current_drone_state = next_drone_state
                    img_tensor = next_img_tensor
                    past_actions = next_past_actions
                    Q_state = next_Q_state
                    if info:
                        done_num+=1
                    if done or episode_steps>200:
                        break
                avg_reward += episode_reward
            avg_reward /= episodes
            if args['logs']==True:
                writer.add_scalar('avg_reward/test', avg_reward, i_episode)
            print("----------------------------------------")
            print(f"Test Episodes: {episodes}, Avg. Reward: {round(avg_reward, 2)}, success num：{done_num}")
            print("----------------------------------------")
            # if i_episode<=100000:
            #     if avg_reward>best_avg_reward and avg_fuel_left>best_avg_fuel:
            #         best_avg_reward=avg_reward
            #         best_avg_fuel=avg_fuel_left
            #         model_name='sofarsogood_{}_success_{}_fuel_{}.pt'.format(k,done_num,round(best_avg_fuel,4))
            #         agent.save_checkpoint(model_name)
            #         memory.save_buffer("AON")
            #         k=k+1
            # if avg_reward > best_avg_reward: #在能成功的基础上，只考虑燃料最优；
            if done_num == episodes and avg_reward > best_avg_reward: #继续训练而已
                model_name = f'master_{k}'
                agent.save_model(model_name)
                best_avg_reward = avg_reward
                memory.save_buffer(model_name)
                k=k+1
                # env.plot(args, steps_list, episode_reward_list, avg_reward_list)
                # break

        if i_episode==args['max_epoch']:
            print("训练结束，{}次仍未完成训练".format(args['max_epoch']))
            # env.plot(args, steps_list, episode_reward_list, avg_reward_list)
            if args['logs']==True:
                writer.close()
            break

if args['task']=='Test':
    agent.load_checkpoint('xxxxxxxxxxxxxxxxxxxxx')
    time_start=time.time()
    episodes = 1000
    done_num=0
    for _ in range(episodes):
        pi_img, pi_state, q_state, goal = env.reset()
        episode_reward = 0
        done = False
        steps=0
        while not done:
            action = agent.select_action(pi_img, np.concatenate((pi_state, goal)), evaluate=True) #evaluate为True时为确定性网络，直接输出mean
            next_pi_img, next_pi_state, next_q_state, reward, done, info = env.step(action)
            memory.push(pi_img, pi_state, q_state, action, reward, next_pi_img, next_pi_state, next_q_state, done, goal, info)
            episode_reward += reward
            pi_img=next_pi_img
            pi_state = next_pi_state
            q_state=next_q_state
            steps+=1
            if info:
                done_num+=1
            if done:
                break
        if info:
            avg_steps+=steps
            avg_reward += episode_reward
    avg_reward /= done_num
    #writer.add_scalar('avg_reward/test', avg_reward, i_episode)
    time_end=time.time()
    print("----------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {},done num:{}".format(episodes, round(avg_reward, 4),done_num))
    print("----------------------------------------")
    print((time_end-time_start)/1000)
