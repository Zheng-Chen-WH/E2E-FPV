"""
自动扫描模型文件：根据命名模式 master_{k}_{avg_reward}_{policy_loss}_{avg_step}_model.pt 扫描所有 .pt 文件

智能筛选：

从文件名解析 avg_reward
avg_reward < 1 的模型直接删除
avg_reward >= 1 的模型加入测试队列
批量测试：按 avg_reward 从高到低依次测试每个模型

结果保存：所有测试结果保存到 ./test_results/test_results_{时间戳}.txt

可配置参数（在文件开头的 TEST_CONFIG 字典中）：

参数	默认值	说明
test_episode	50	每个模型测试轮数
max_steps_per_episode	200	每轮最大步数
avg_reward_threshold	1.0	删除阈值
delete_low_reward	True	是否删除低奖励模型
dry_run	False	干跑模式（只打印不删除）
    main()
"""

import config as cfg
from env import env
import numpy as np
from sac import SAC
from ppo import PPO
import time
from utils import map_value
import os
import glob
import re
from datetime import datetime

# ==================== 可配置参数 ====================
TEST_CONFIG = {
    'test_episode': 50,           # 每个模型测试的轮数
    'max_steps_per_episode': 200, # 每轮最大步数
    'avg_reward_threshold': 1.0,  # avg_reward 阈值，低于此值的模型将被删除
    'rl_algorithm': 'SAC',        # 强化学习算法，SAC 或 PPO
    'model_pattern': 'master_*_model.pt',  # 模型文件匹配模式
    'output_folder': './test_results/',    # 测试结果输出文件夹
    'delete_low_reward': True,    # 是否删除低奖励模型
    'dry_run': False,             # 干跑模式：True时只打印要删除的文件，不实际删除
}

# 超参数字典（与 main.py 保持一致）
agent_args = {
    'device': cfg.device,
    'critic_param': cfg.critic_param,
    'actor_param': cfg.actor_param,
    'SAC_param': cfg.SAC_param,
    'PPO_param': cfg.PPO_param,
}

# MPC参数
mpc_params = cfg.mpc_params
# env参数
env_params = cfg.env_params


def parse_model_filename(filename):
    """
    解析模型文件名，提取各项指标
    文件名格式: master_{k}_{avg_reward}_{policy_loss}_{avg_step}_model.pt
    
    Args:
        filename: 模型文件名（不含路径）
    
    Returns:
        dict: 包含解析出的各项指标，解析失败返回 None
    """
    # 移除 _model.pt 后缀
    base_name = filename.replace('_model.pt', '')
    
    # 使用正则表达式匹配
    # 格式: master_{k}_{avg_reward}_{policy_loss}_{avg_step}
    # 注意: avg_reward, policy_loss 可能为负数
    pattern = r'^master_(\d+)_([-]?\d+\.?\d*)_([-]?\d+\.?\d*)_([-]?\d+\.?\d*)$'
    match = re.match(pattern, base_name)
    
    if match:
        return {
            'k': int(match.group(1)),
            'avg_reward': float(match.group(2)),
            'policy_loss': float(match.group(3)),
            'avg_step': float(match.group(4)),
            'filename': filename
        }
    
    # 尝试匹配其他可能的格式（如 best_master_model.pt）
    if 'best_master' in filename:
        return {
            'k': -1,  # 特殊标记
            'avg_reward': float('inf'),  # 最佳模型不删除
            'policy_loss': 0,
            'avg_step': 0,
            'filename': filename,
            'is_best': True
        }
    
    return None


def get_all_model_files(directory='.', pattern='master_*_model.pt'):
    """
    获取目录下所有符合模式的模型文件
    
    Args:
        directory: 搜索目录
        pattern: 文件名匹配模式
    
    Returns:
        list: 模型文件路径列表
    """
    search_path = os.path.join(directory, pattern)
    return glob.glob(search_path)


def test_model(agent, airsim_environment, model_name, test_episodes, max_steps, log_file):
    """
    测试单个模型
    
    Args:
        agent: SAC/PPO agent
        airsim_environment: 环境实例
        model_name: 模型名称（不含 _model.pt 后缀）
        test_episodes: 测试轮数
        max_steps: 每轮最大步数
        log_file: 日志文件句柄
    
    Returns:
        dict: 测试结果统计
    """
    # 加载模型
    try:
        agent.load_model(model_name, evaluate=True)
    except Exception as e:
        error_msg = f"[ERROR] 加载模型 {model_name} 失败: {str(e)}"
        print(error_msg)
        log_file.write(error_msg + '\n')
        return None
    
    # 固定测试种子
    test_seeds = [1000000 + i for i in range(test_episodes)]
    
    done_num = 0
    total_reward = 0
    total_steps = 0
    episode_results = []
    
    start_time = time.time()
    
    for episode_idx, seed in enumerate(test_seeds):
        episode_reward = 0
        episode_steps = 0
        success = False
        
        # 重置环境和agent
        current_drone_state, final_target_state, waypoints_y, \
            door_z_positions, door_param, img_tensor, critic_state, \
            final_pi_target, elapsed_time, relative_next_target_pos, \
            attitude_9d, relative_next_target_vel, fpv_angular_vel = airsim_environment.reset(seed=seed)
        agent.reset()
        
        while True:
            # 选择动作（评估模式）
            NN_action, _, _, _ = agent.select_action(img_tensor, final_pi_target, critic_state, evaluate=True)
            
            # 将动作映射回MPC空间
            rescaled_NN_action = map_value(
                NN_action, 
                agent_args['actor_param']['scaled_min_action'], 
                agent_args['actor_param']['scaled_max_action'],
                mpc_params['control_min'], 
                mpc_params['control_max']
            )
            
            # 执行动作
            next_drone_state, next_img_tensor, next_critic_state, \
                reward, done, phase_idx, info, elapsed_time, \
                relative_next_target_pos, attitude_9d, \
                relative_next_target_vel, fpv_angular_vel = airsim_environment.step(rescaled_NN_action)
            
            episode_reward += reward
            episode_steps += 1
            
            # 更新状态
            current_drone_state = next_drone_state
            img_tensor = next_img_tensor
            critic_state = next_critic_state
            
            if done or episode_steps >= max_steps:
                if info:
                    success = True
                    done_num += 1
                break
        
        total_reward += episode_reward
        total_steps += episode_steps
        episode_results.append({
            'episode': episode_idx + 1,
            'reward': episode_reward,
            'steps': episode_steps,
            'success': success
        })
        
        # 实时打印进度
        progress_msg = f"  Episode {episode_idx + 1}/{test_episodes}: reward={round(episode_reward, 2)}, success={success}"
        print(progress_msg)
    
    elapsed_time = time.time() - start_time
    avg_reward = total_reward / test_episodes
    avg_steps = total_steps / test_episodes
    success_rate = done_num / test_episodes
    
    # 汇总结果
    result = {
        'model_name': model_name,
        'test_episodes': test_episodes,
        'avg_reward': avg_reward,
        'total_success': done_num,
        'success_rate': success_rate,
        'avg_steps': avg_steps,
        'elapsed_time': elapsed_time,
        'episode_results': episode_results
    }
    
    return result


def write_result_to_log(log_file, result, model_info):
    """
    将测试结果写入日志文件
    """
    log_file.write("=" * 70 + '\n')
    log_file.write(f"模型文件: {model_info['filename']}\n")
    log_file.write(f"文件名中的指标 - avg_reward: {model_info['avg_reward']}, "
                   f"policy_loss: {model_info['policy_loss']}, avg_step: {model_info['avg_step']}\n")
    log_file.write("-" * 70 + '\n')
    
    if result is None:
        log_file.write("[ERROR] 模型测试失败\n")
    else:
        log_file.write(f"测试轮数: {result['test_episodes']}\n")
        log_file.write(f"平均奖励: {round(result['avg_reward'], 4)}\n")
        log_file.write(f"成功次数: {result['total_success']}\n")
        log_file.write(f"成功率: {round(result['success_rate'] * 100, 2)}%\n")
        log_file.write(f"平均步数: {round(result['avg_steps'], 2)}\n")
        log_file.write(f"测试耗时: {round(result['elapsed_time'], 2)}秒\n")
        log_file.write("-" * 70 + '\n')
        log_file.write("各轮详情:\n")
        for ep in result['episode_results']:
            log_file.write(f"  Episode {ep['episode']}: reward={round(ep['reward'], 2)}, "
                          f"steps={ep['steps']}, success={ep['success']}\n")
    
    log_file.write("=" * 70 + '\n\n')
    log_file.flush()  # 立即写入磁盘


def main():
    print("=" * 70)
    print("自动模型测试脚本启动")
    print("=" * 70)
    
    # 创建输出文件夹
    if not os.path.exists(TEST_CONFIG['output_folder']):
        os.makedirs(TEST_CONFIG['output_folder'])
    
    # 生成输出文件名（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(TEST_CONFIG['output_folder'], f'test_results_{timestamp}.txt')
    
    # 获取所有模型文件
    model_files = get_all_model_files(pattern=TEST_CONFIG['model_pattern'])
    print(f"找到 {len(model_files)} 个模型文件")
    
    if len(model_files) == 0:
        print("未找到任何模型文件，退出。")
        return
    
    # 解析所有模型文件
    models_to_delete = []
    models_to_test = []
    models_parse_failed = []
    
    for model_path in model_files:
        filename = os.path.basename(model_path)
        info = parse_model_filename(filename)
        
        if info is None:
            models_parse_failed.append(model_path)
            print(f"[警告] 无法解析文件名: {filename}")
            continue
        
        info['full_path'] = model_path
        
        if info['avg_reward'] < TEST_CONFIG['avg_reward_threshold']:
            models_to_delete.append(info)
        else:
            models_to_test.append(info)
    
    # 按 avg_reward 排序待测试模型（从高到低）
    models_to_test.sort(key=lambda x: x['avg_reward'], reverse=True)
    
    print(f"\n统计:")
    print(f"  - 待删除（avg_reward < {TEST_CONFIG['avg_reward_threshold']}）: {len(models_to_delete)} 个")
    print(f"  - 待测试（avg_reward >= {TEST_CONFIG['avg_reward_threshold']}）: {len(models_to_test)} 个")
    print(f"  - 解析失败: {len(models_parse_failed)} 个")
    
    # 删除低奖励模型
    if TEST_CONFIG['delete_low_reward'] and len(models_to_delete) > 0:
        print(f"\n{'=' * 70}")
        print("开始删除低奖励模型...")
        for info in models_to_delete:
            if TEST_CONFIG['dry_run']:
                print(f"  [干跑] 将删除: {info['filename']} (avg_reward={info['avg_reward']})")
            else:
                try:
                    os.remove(info['full_path'])
                    print(f"  [已删除] {info['filename']} (avg_reward={info['avg_reward']})")
                except Exception as e:
                    print(f"  [删除失败] {info['filename']}: {str(e)}")
    
    # 如果没有需要测试的模型，退出
    if len(models_to_test) == 0:
        print("\n没有需要测试的模型，退出。")
        return
    
    # 初始化环境和Agent
    print(f"\n{'=' * 70}")
    print("初始化环境和Agent...")
    airsim_environment = env(env_params)
    
    if TEST_CONFIG['rl_algorithm'] == 'PPO':
        agent = PPO(agent_args)
    else:
        agent = SAC(agent_args)
    
    print("初始化完成！")
    
    # 开始测试
    print(f"\n{'=' * 70}")
    print(f"开始测试 {len(models_to_test)} 个模型")
    print(f"每个模型测试 {TEST_CONFIG['test_episode']} 轮")
    print(f"测试结果将保存至: {output_file}")
    print("=" * 70)
    
    # 打开日志文件
    with open(output_file, 'w', encoding='utf-8') as log_file:
        # 写入测试配置信息
        log_file.write("=" * 70 + '\n')
        log_file.write("自动模型测试报告\n")
        log_file.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("=" * 70 + '\n')
        log_file.write(f"测试配置:\n")
        log_file.write(f"  - 测试轮数/模型: {TEST_CONFIG['test_episode']}\n")
        log_file.write(f"  - 每轮最大步数: {TEST_CONFIG['max_steps_per_episode']}\n")
        log_file.write(f"  - 算法: {TEST_CONFIG['rl_algorithm']}\n")
        log_file.write(f"  - avg_reward阈值: {TEST_CONFIG['avg_reward_threshold']}\n")
        log_file.write("=" * 70 + '\n')
        log_file.write(f"已删除的低奖励模型 ({len(models_to_delete)} 个):\n")
        for info in models_to_delete:
            log_file.write(f"  - {info['filename']} (avg_reward={info['avg_reward']})\n")
        log_file.write("=" * 70 + '\n\n')
        log_file.write(f"待测试模型 ({len(models_to_test)} 个):\n")
        for info in models_to_test:
            log_file.write(f"  - {info['filename']} (avg_reward={info['avg_reward']})\n")
        log_file.write("\n" + "=" * 70 + '\n')
        log_file.write("测试结果详情:\n")
        log_file.write("=" * 70 + '\n\n')
        
        # 汇总统计
        all_results = []
        
        # 测试每个模型
        for idx, model_info in enumerate(models_to_test):
            print(f"\n[{idx + 1}/{len(models_to_test)}] 测试模型: {model_info['filename']}")
            print(f"  文件名指标 - avg_reward: {model_info['avg_reward']}, "
                  f"policy_loss: {model_info['policy_loss']}, avg_step: {model_info['avg_step']}")
            print("-" * 50)
            
            # 提取模型名称（不含 _model.pt 后缀）
            model_name = model_info['filename'].replace('_model.pt', '')
            
            # 测试模型
            result = test_model(
                agent=agent,
                airsim_environment=airsim_environment,
                model_name=model_name,
                test_episodes=TEST_CONFIG['test_episode'],
                max_steps=TEST_CONFIG['max_steps_per_episode'],
                log_file=log_file
            )
            
            # 写入结果
            write_result_to_log(log_file, result, model_info)
            
            if result is not None:
                all_results.append({
                    'model_info': model_info,
                    'result': result
                })
                
                # 打印汇总
                print("-" * 50)
                print(f"  测试完成: avg_reward={round(result['avg_reward'], 4)}, "
                      f"success_rate={round(result['success_rate'] * 100, 2)}%")
        
        # 写入汇总表格
        log_file.write("\n" + "=" * 70 + '\n')
        log_file.write("汇总排名（按实测avg_reward降序）:\n")
        log_file.write("=" * 70 + '\n')
        log_file.write(f"{'排名':<6}{'模型名':<45}{'文件名reward':<15}{'实测reward':<15}{'成功率':<12}\n")
        log_file.write("-" * 70 + '\n')
        
        # 按实测 avg_reward 排序
        all_results.sort(key=lambda x: x['result']['avg_reward'], reverse=True)
        
        for rank, item in enumerate(all_results, 1):
            model_name = item['model_info']['filename']
            file_reward = item['model_info']['avg_reward']
            test_reward = item['result']['avg_reward']
            success_rate = item['result']['success_rate'] * 100
            log_file.write(f"{rank:<6}{model_name:<45}{file_reward:<15.2f}{test_reward:<15.4f}{success_rate:<12.2f}%\n")
        
        log_file.write("=" * 70 + '\n')
        log_file.write(f"\n测试完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("\n" + "=" * 70)
    print(f"所有测试完成！结果已保存至: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
