import datetime
import numpy as np
import itertools
import torch.nn as nn
from sac import SAC
from replay_memory import HERMemory,ReplayMemory
import env
import time
import matplotlib.pyplot as plt
import numpy as np
import torch


# 字典形式存储全部参数
args={'eval':True, # Evaluates a policy a policy every 10 episode (default: True)
        'gamma':0.99, # discount factor for reward (default: 0.99)
        'tau':0.2, # target smoothing coefficient(τ) (default: 0.005) 参数tau定义了目标网络软更新时的平滑系数，
                     # 它控制了新目标值在更新目标网络时与旧目标值的融合程度。
                     # 较小的tau值会导致目标网络变化较慢，从而增加了训练的稳定性，但也可能降低学习速度。
        'lr':0.0003, # learning rate (default: 0.0003)
        'alpha':0.2, # Temperature parameter α determines the relative importance of the entropy\term against the reward (default: 0.2)
        'automatic_entropy_tuning':False, # Automaically adjust α (default: False)
        'batch_size':32, # batch size (default: 256)
        'num_steps':1000, # maximum number of steps (default: 1000000)
        'hidden_sizes':[128,128,64], # 隐藏层大小，带有激活函数的隐藏层层数等于这一列表大小
        'updates_per_episode':2, # model updates per simulator step (default: 1) 每步对参数更新的次数
        'start_steps':1000, # Steps sampling random actions (default: 10000) 在开始训练之前完全随机地进行动作以收集数据
        'target_update_interval':10, # Value target update per no. of updates per step (default: 1) 目标网络更新的间隔
        'replay_size':10000000, # size of replay buffer (default: 10000000)
        'cuda':True, # run on CUDA (default: False)
        'LOAD PARA':False, #是否读取参数
        'task':'Train', # 测试或训练或画图，Train,Test,Plot
        'activation':nn.ReLU, #激活函数类型
        'plot_type':'2D-2line', #'3D-1line'为三维图，一条曲线；'2D-2line'为二维图，两条曲线
        'plot_title':'reward-steps.svg',
        'max_episodes':1e6, #测试算法（eval=False）情况下的总步数
        'evaluate_freq':50, #训练过程中每多少个epoch之后进行测试
        'seed':20000323, #网络初始化的时候用的随机数种子  
        'max_epoch':250000,
        'logs':True, #是否留存训练参数供tensorboard分析 
        'embedding_dim':128,
        'num_frames':4,
        'door_frames':["men_Blueprint","men2_Blueprint","men3_Blueprint","men4_Blueprint"]} 

# Environment
env = env.env(args)
# Agent
agent = SAC(env.pi_observation_space.shape[0], env.q_observation_space.shape[0], env.action_space, args) #discrete不能用shape，要用n提取维度数量
time_start=time.time()
#Tensorboard
#显示图像：用cmd（不是vscode的终端） cd到具体存放日志的文件夹（runs），然后tensorboard --logdir=./ --samples_per_plugin scalars=999999999
#或者直接在import的地方点那个启动会话
#如果还是不行的话用netstat -ano | findstr "6006" 在cmd里查一下6006端口有没有占用，用taskkill全杀了之后再tensorboard一下

# Memory
memory = HERMemory(args['replay_size'],"Final") # HER没修好
memory = ReplayMemory(args['replay_size']) 

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
        agent.load_checkpoint("xxxxxxxxxxxxx")
        best_avg_reward=0
    for i_episode in itertools.count(1): #itertools.count(1)用于创建一个无限迭代器。它会生成一个连续的整数序列，从1开始，每次递增1。
        success=False
        episode_reward = 0
        done=False
        episode_steps = 0
        pi_img, pi_state, q_state, goal = env.reset()
        while True:
            action = agent.select_action(pi_img, np.concatenate((pi_state, goal)))  # 开始输出actor网络动作
            next_pi_img, next_pi_state, next_q_state, reward, done, info= env.step(action) # Step
            episode_steps += 1
            episode_reward += reward #没用gamma是因为在sac里求q的时候用了
            total_numsteps+=1
            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            memory.push(pi_img, pi_state, q_state, action, reward, next_pi_img, next_pi_state, next_q_state, done, goal, info) # Append transition to memory
            pi_img=next_pi_img
            pi_state = next_pi_state
            q_state=next_q_state

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
        if not success:
            print("Episode: {}, steps: {}, reward: {}, succeed: {}".format(i_episode, episode_steps, round(episode_reward, 2), success))
        elif success:
            print("Episode: {}, steps: {}, reward: {}, ❤ ❤ ❤ ❤ ❤ ❤ ❤ ❤ ❤ ❤ ❤ ❤ ❤ ❤ ❤succeed: {}❤ ❤ ❤ ❤ ❤ ❤ ❤ ❤ ❤ ❤ ❤ ❤ ❤ ❤ ❤".format(i_episode, episode_steps, round(episode_reward, 2), success))
        # round(episode_reward,2) 对episode_reward进行四舍五入，并保留两位小数

        if i_episode % args['evaluate_freq'] == 0 and args['eval'] is True: #评价上一个训练过程
            avg_reward = 0.
            episodes = 5
            done_num=0
            for _ in range(episodes):
                episode_reward = 0
                done=False
                episode_steps = 0
                pi_img, pi_state, q_state, goal = env.reset()
                print("reset了")
                while True:
                    action = agent.select_action(pi_img, np.concatenate((pi_state, goal)), evaluate=True) #evaluate为True时为确定性网络，直接输出mean
                    next_pi_img, next_pi_state, next_q_state, reward, done, info = env.step(action)
                    memory.push(pi_img, pi_state, q_state, action, reward, next_pi_img, next_pi_state, next_q_state, done, goal, info)
                    episode_reward += reward
                    pi_img=next_pi_img
                    pi_state = next_pi_state
                    q_state=next_q_state
                    episode_steps+=1
                    if info:
                        done_num+=1
                    if done or episode_steps>200:
                        break
                avg_reward += episode_reward
            avg_reward /= episodes
            if args['logs']==True:
                writer.add_scalar('avg_reward/test', avg_reward, i_episode)
            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}, success num：{}".format(episodes, round(avg_reward, 2), done_num))
            print("----------------------------------------")
            # if i_episode<=100000:
            #     if avg_reward>best_avg_reward and avg_fuel_left>best_avg_fuel:
            #         best_avg_reward=avg_reward
            #         best_avg_fuel=avg_fuel_left
            #         model_name='sofarsogood_{}_success_{}_fuel_{}.pt'.format(k,done_num,round(best_avg_fuel,4))
            #         agent.save_checkpoint(model_name)
            #         memory.save_buffer("AON")
            #         k=k+1
            if avg_reward>best_avg_reward: #在能成功的基础上，只考虑燃料最优；
            # if done_num==episodes: #继续训练而已
                model_name='sofarsogood_{}.pt'.format(k)
                agent.save_checkpoint(model_name)
                best_avg_reward=avg_reward
                # memory.save_buffer("AON-1")
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

if args['task']=='Plot':
    agent.load_checkpoint('sofarsogood_39_success_20_fuel_9.7859.pt')
    state = env.reset(99)
    done = False
    Xt=[]
    Xc=[]
    Vt=[]
    Vc=[]
    time_sequence=[]
    t=0
    while not done:
        action = agent.select_action(state, evaluate=True) #evaluate为True时为确定性网络，直接输出mean
        t=t+action[3]
        time_sequence.append(t)
        next_state, done, info, red_pos, blue_pos, red_vel, blue_vel = env.plotstep(action)
        Xt.append(red_pos)
        Xc.append(blue_pos)
        Vt.append(red_vel)
        Vc.append(blue_vel)
        state = next_state
    print("----------------------------------------")
    print("done:{}".format(info))
    print("----------------------------------------")
    pos_x_target = np.array([array[0] for array in Xt])
    pos_y_target = np.array([array[1] for array in Xt])
    pos_z_target = np.array([array[2] for array in Xt])
    pos_x_chaser = np.array([array[0] for array in Xc])
    pos_y_chaser = np.array([array[1] for array in Xc])
    pos_z_chaser = np.array([array[2] for array in Xc])
    velo_x_target = np.array([array[0] for array in Vt])
    velo_y_target = np.array([array[1] for array in Vt])
    velo_z_target = np.array([array[2] for array in Vt])
    velo_x_chaser = np.array([array[0] for array in Vc])
    velo_y_chaser = np.array([array[1] for array in Vc])
    velo_z_chaser = np.array([array[2] for array in Vc])
    
    font = {'family': 'serif',
    'serif': 'Times New Roman',
    'weight': 'normal',
    'size': 23,
    }
    plt.rc('font', **font)
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['font.family'] = 'Times New Roman'
    fig = plt.figure(figsize=(17,6))
    ax1 = fig.add_subplot(131, projection='3d') #, projection='3d' #生成3d子图
    ax2 = fig.add_subplot(132) #, projection='3d'
    ax3 = fig.add_subplot(133)
    #Axes对象是图形的绘图区域，可以在其上进行各种绘图操作。
    #通过gca()函数可以获取当前图形的Axes对象，如果该对象不存在，则会创建一个新的。
    #projection='3d' 参数指定了Axes对象的投影方式为3D，即创建一个三维坐标系。
    ax1.plot(pos_x_target/1e3,pos_y_target/1e3,pos_z_target/1e3,'r',linewidth=2, label="Target")
    ax1.plot(pos_x_chaser/1e3,pos_y_chaser/1e3,pos_z_chaser/1e3,'g',linewidth=2, label="Pursuer")
    ax2.plot(time_sequence,(pos_x_target-pos_x_chaser)/1e3,'r',linewidth=2, label="x distance") 
    ax2.plot(time_sequence,(pos_y_target-pos_y_chaser)/1e3,'b',linewidth=2, label="y distance") 
    ax2.plot(time_sequence,(pos_z_target-pos_z_chaser)/1e3,'g',linewidth=2, label="z distance")
    ax3.plot(time_sequence,velo_x_target-velo_x_chaser,'r',linewidth=2, label="x velocity") 
    ax3.plot(time_sequence,velo_y_target-velo_y_chaser,'b',linewidth=2, label="y velocity") 
    ax3.plot(time_sequence,velo_z_target-velo_z_chaser,'g',linewidth=2, label="z velocity")
    #ax.scatter(data_x,data_y,data_z,'b',s=1)
    ax1.set_xlabel('x/km', fontsize=25)
    ax1.set_ylabel('y/km', fontsize=25)
    ax1.set_zlabel('z/km', fontsize=25)
    ax1.xaxis.labelpad = 20  # 负值将标签向图像外部移动
    ax1.yaxis.labelpad = 20  # 负值将标签向图像外部移动
    ax1.zaxis.labelpad = 13  # 负值将标签向图像外部移动
    # 获取当前的x轴和y轴标签对象
    zlabel = ax1.get_zlabel()
    # 设置z轴标签的方向，使其与z轴平行
    ax1.zaxis.label.set_rotation(90)  # 0度表示与X轴平行
    ax2.set_xlabel('t/s', fontsize=25)
    ax2.set_ylabel('distance/km', fontsize=25)
    # 获取当前的x轴和y轴
    ax2_xaxis = ax2.get_xaxis()
    ax2_yaxis = ax2.get_yaxis()
    # 调整x轴标签的位置到x轴的最右端
    ax2_xaxis.label.set_horizontalalignment('right')
    ax2_yaxis.label.set_position((1, 0))
    # 调整y轴标签的位置到y轴的最上端
    ax2_yaxis.label.set_verticalalignment('top')
    y_axis_pos = ax2.get_ylim()[1]  # 获取y轴的最上端位置
    ax2_yaxis.label.set_position((0, 1))
    # 将x轴标签移动到x轴的末端,调节y坐标控制其上下移动
    ax2.xaxis.set_label_coords(1.05, -0.05)
    # 将y轴标签移动到y轴的末端,调节x坐标控制其左右移动
    ax2.yaxis.set_label_coords(-0.18, 0.9)
    ax3.set_xlabel('t/s', fontsize=25)
    ax3.set_ylabel('velocity/(m/s)', fontsize=25)
    ax1.legend(fontsize=20,loc='upper right')
    ax2.legend(fontsize=20)
    ax3.legend(fontsize=20)
    # ax.set_xlim(np.min([np.min(data_x),np.min(data_y),np.min(data_z)]),np.max([np.max(data_x),np.max(data_y),np.max(data_z)]))
    # ax.set_ylim(np.min([np.min(data_x),np.min(data_y),np.min(data_z)]),np.max([np.max(data_x),np.max(data_y),np.max(data_z)]))
    # ax.set_zlim(np.min([np.min(data_x),np.min(data_y),np.min(data_z)]),np.max([np.max(data_x),np.max(data_y),np.max(data_z)]))
    plt.tight_layout()# 调整布局使得图像不溢出
    plt.savefig("test.svg", format='svg', bbox_inches='tight')# 'logs/{}epoch-{}steps.png'.format(epoch,steps))
    plt.show()