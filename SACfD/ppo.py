import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import BatchSampler, SubsetRandomSampler
from torch.optim import AdamW
from model import GaussianPolicy, ValueNetwork, init_weights
from utils import six_d_to_rot_mat, physics_MSE
from replay_memory import RolloutBuffer, ReplayMemory
from tqdm import tqdm  # 新增导入

class PPO(object):
    def __init__(self, args):
        PPO_dict = args['PPO_param']
        self.device = args['device']
        self.warm_up_steps = PPO_dict['warm_up_steps']
        self.base_lr = PPO_dict['lr']
        self.clip_param = PPO_dict['clip']
        self.ppo_epoch = PPO_dict['ppo_epoch']
        self.mini_batch_size = PPO_dict['mini_batch_size']
        self.value_loss_coef = PPO_dict['value_loss_coef']
        self.entropy_coef = PPO_dict['entropy_coef']
        self.max_norm_grad = PPO_dict['max_norm_grad']
        self.adv_normalization = PPO_dict['adv_normalization'] # 是否对优势函数进行归一化
        
        # Loss Weights
        loss_dict = PPO_dict['loss_weight']
        self.aux_loss_weight = loss_dict['aux_loss_weight']
        self.pos_loss_weight = loss_dict['pos_loss_weight']
        self.rot_loss_weight = loss_dict['rot_loss_weight']
        self.vel_loss_weight = loss_dict['vel_loss_weight']
        self.ang_vel_loss_weight = loss_dict['ang_vel_loss_weight']
        self.il_weight = loss_dict['il_loss_weight']

        # il与rl损失动态混合机制参数
        self.loss_dynamic_change_window = PPO_dict['loss_dynamic_change_window']
        self.rl_loss_weight_target = PPO_dict['rl_loss_weight_target']

        # Buffer
        # PPO:短期 Rollout Buffer (On-Policy)
        self.rollout_buffer = RolloutBuffer(PPO_dict['rollout_buffer'])
        # IL:长期Expert Buffer (Off-Policy)
        self.expert_buffer = ReplayMemory(PPO_dict['buffer_param']) 
        self.expert_sample_config = PPO_dict['batch_size']  # 用于IL采样的batch配置

        self.critic = ValueNetwork(args['critic_param']).to(self.device)
        self.policy = GaussianPolicy(args['actor_param']).to(self.device)
        self.critic.apply(init_weights)
        self.policy.apply(init_weights)
        # 特殊初始化，让策略和v初始输出接近0
        nn.init.constant_(self.policy.log_std_layer.weight, 0)
        nn.init.constant_(self.policy.log_std_layer.bias, 0)
        nn.init.uniform_(self.policy.mu_layer.weight, -PPO_dict['mu_init_boundary'], PPO_dict['mu_init_boundary'])
        nn.init.uniform_(self.policy.mu_layer.bias, -PPO_dict['mu_init_boundary'], PPO_dict['mu_init_boundary'])
        nn.init.uniform_(self.critic.v_net[-2].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.critic.v_net[-2].bias, -3e-3, 3e-3)

        self.critic_optim = AdamW(self.critic.parameters(), lr=self.base_lr)
        self.policy_optim = AdamW(self.policy.parameters(), lr=self.base_lr, weight_decay=0.01)
        
        self.hidden_state = None

    def reset(self):
        self.hidden_state = None
    
    def aux_loss(self, first_aux, second_aux, gt_pos, gt_rot, gt_vel, gt_ang):
        """根据辅助头输出与实际标签值计算辅助头的多任务学习loss

        Args:
            first_aux_output (_type_): 第一模块的辅助头输出 (B, T, 9)
            second_aux_output (_type_): 第二模块的辅助头输出 (B, T, 9)
            gt_pos (_type_): 相对位置真值 (B, T, 3)
            gt_rot_mat (_type_): 相对姿态真值 (旋转矩阵)
            gt_vel (_type_): 相对真值 (B, T, 3)
            gt_ang_vel (_type_): 相对角速度真值 (B, T, 3)

        Returns:
            total_loss: 辅助头损失总值
        """
        pred_pos = first_aux[..., 0:3]
        pred_rot_6d = first_aux[..., 3:9]
        pred_vel = second_aux[..., 0:3]
        pred_ang = second_aux[..., 3:6]

        loss_pos = F.mse_loss(pred_pos, gt_pos)
        loss_vel = F.mse_loss(pred_vel, gt_vel)
        loss_ang = F.mse_loss(pred_ang, gt_ang)
        
        # 6D 转 旋转矩阵
        pred_rot_flat = pred_rot_6d.reshape(-1, 6)
        gt_rot_flat = gt_rot.reshape(-1, 3, 3)
        R_pred = six_d_to_rot_mat(pred_rot_flat)
        loss_rot = F.mse_loss(R_pred, gt_rot_flat)

        return (self.pos_loss_weight * loss_pos + 
                self.rot_loss_weight * loss_rot + 
                self.vel_loss_weight * loss_vel + 
                self.ang_vel_loss_weight * loss_ang)

    def select_action(self, img_sequence, state, V_state, evaluate=False):
        """
        PPO 交互时需要返回 action, log_prob 和 value
        """
        img_sequence = img_sequence.to(self.device)
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        V_state_tensor = torch.FloatTensor(V_state).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            # 获取 Value
            value = self.critic(V_state_tensor)
            # 获取 Action 和 Log Prob
            # 使用 sample 方法，它内部处理了 hidden_state
            action, log_prob, mean, _, _, new_hidden = self.policy.sample(img_sequence, state_tensor, self.hidden_state)
            if evaluate:
                action = mean
        # 更新hidden state（用于GRU）
        input_hidden = self.hidden_state
        if new_hidden is not None:
            self.hidden_state = new_hidden
        # 返回numpy数组（用于环境交互）
        return action.cpu().numpy()[0], log_prob.cpu().item(), value.cpu().item(), input_hidden

    def evaluate_actions(self, img_seq, state, action, hidden_state=None):
        """
        在 Update 循环中评估动作的概率
        Recurrent PPO 关键改变:
        - 输入现在是序列形式: (batch, seq_len, ...)
        - 从 hidden_state 开始，让 GRU 在序列上前向传播
        - 这样 GRU 可以正确地处理时序依赖
        现代PPO标准实现，正确处理Tanh Squashed Gaussian分布
        Args:
            img_seq: (batch * seq_len, T, C, H, W) 展平后的图像序列
            state: (batch * seq_len, state_dim) 展平后的状态
            action: (batch * seq_len, action_dim) 展平后的动作
            hidden_state: GRU 初始隐状态（可选）
        
        Returns:
            log_prob, entropy, first_aux, second_aux
        """
        # img_seq: (batch, seq_len, T, C, H, W)
        batch_size, seq_len = img_seq.shape[:2]
        if hidden_state is not None:
            # 预分配结果张量（避免动态 append）
            action_dim = action.shape[-1]
            T = img_seq.shape[2]  # 图像帧数
            all_means = torch.empty(batch_size, seq_len, action_dim, device=self.device)
            all_log_stds = torch.empty(batch_size, seq_len, action_dim, device=self.device)
            all_first_aux = torch.empty(batch_size, seq_len, T, 9, device=self.device)
            all_second_aux = torch.empty(batch_size, seq_len, T, 6, device=self.device)
            current_hidden = hidden_state

            # 逐时间步前向传播，保持时序依赖
            for t in range(seq_len):
                # 取出当前时间步的数据
                # img_seq[:, t] 形状: (batch, T, C, H, W) - 单个时间步的观测
                '''
                需要保持【张量内存连续性】，否则会报错
                PyTorch 张量在内存中是按行优先（Row-Major） 顺序存储的，
                当做切片、转置、permute 等操作时，PyTorch 不会复制数据，而是改变访问方式
                因此当仅仅用t切片时，传给GRU的张量不是连续的内存块，而是“访问”的一种方式
                GRU中的view要求数据在内存中必须连续，因为它只是改变"形状解释"，不复制数据
                .contiguous()会复制数据到连续内存，从而满足view的要求
                '''

                img_t = img_seq[:, t].contiguous()      
                state_t = state[:, t].contiguous()      # (batch, state_dim)
                # 单步前向，传入并更新 hidden state
                mean, log_std, first_aux, second_aux, current_hidden = self.policy(
                    img_t, state_t, current_hidden
                )
                # 直接写入预分配的张量
                all_means[:, t] = mean
                all_log_stds[:, t] = log_std
                all_first_aux[:, t] = first_aux
                all_second_aux[:, t] = second_aux
        else:
            # 如果hidden_state=None，说明没有RNN
            # 直接展平+前向传播
            T = img_seq.shape[2]  # 图像帧数
            print(f"img_seq shape for eval: {img_seq.shape}")
            print(f"img_seq view shape: {img_seq.view(-1, *img_seq.shape[2:]).shape}")
            print(f"state shape for eval: {state.shape}")
            print(f"state view shape: {state.view(-1, state.shape[-1]).shape}")
            mean, log_std, all_first_aux, all_second_aux, _ = self.policy(
                img_seq.contiguous().view(-1, *img_seq.shape[2:]), state.contiguous().view(-1, state.shape[-1])
            )
            all_means = mean.view(batch_size, seq_len, -1)
            all_log_stds = log_std.view(batch_size, seq_len, -1)
            all_first_aux = all_first_aux.view(batch_size, seq_len, T, -1)
            all_second_aux = all_second_aux.view(batch_size, seq_len, T, -1)

        std = all_log_stds.exp()
        normal = torch.distributions.Normal(all_means, std)
        
        # Tanh Squashed Gaussian 分布的 log_prob 计算
        action_scale = self.policy.action_scale
        action_bias = self.policy.action_bias
        action_normalized = (action - action_bias) / action_scale
        action_clipped = torch.clamp(action_normalized, -0.9999999, 0.9999999)
        
        x_t = torch.atanh(action_clipped)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action_clipped.pow(2) + 1e-6)
        # 在计算old_log_prob时已经在model.py里-log(action_scale)，这里必须再减一次以对齐
        log_prob -= torch.log(action_scale + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        entropy = normal.entropy().sum(-1, keepdim=True)
        return log_prob, entropy, all_first_aux, all_second_aux

    def push_data(self, source_name, data):
        """
        将数据推送到相应的buffer
        
        Args:
            source_name: 'expert', 'dagger' -> expert_buffer
                        'rollout' -> rollout_buffer
            data: 数据元组
        """
        if source_name in ['expert', 'dagger']:
            # 推送到expert_buffer（用于IL）
            self.expert_buffer.push(source_name, data)
            
        elif source_name == 'rollout':
            # 推送到rollout_buffer（用于PPO更新）
            # data格式: (img_seq, V_state, pi_state, action, reward, done, log_prob, value, 
            #            pos, rot, vel, ang, hidden_state)
            img_seq, V_state, pi_state, action, reward, done, log_prob, value, \
                pos, rot, vel, ang, hidden_state = data
            self.rollout_buffer.push(data)
            if done:
                # 计算 GAE 和 returns
                '''
                在Vt+1已经done的情况下，终止状态的value设为0
                Value表示State的预期回报，因此done状态（没有未来）的Value为0
                '''
                self.rollout_buffer.finish_path(0.0, done)

    def update(self, updates):
        """
        PPO更新， 采用Recurrent PPO以适配GRU
        
        关键点:
        - 数据现在是序列形式: (batch, seq_len, ...)
        - 使用 mask 来处理变长序列（padding 的位置不计入损失）
        - 在序列开头使用存储的 hidden_state 初始化 GRU
        
        Args:
            updates: 当前更新次数（用于日志）
        
        Returns:
            policy_loss, value_loss, rl_loss, il_loss, aux_loss (用于日志)
        """
        
        # 累计损失（用于日志）
        total_policy_loss = 0
        total_value_loss = 0
        total_aux_loss = 0
        total_il_loss = 0
        total_entropy = 0
        total_rl_loss = 0
        n_updates = 0

        # 先进行模仿学习损失计算数据准备, 从专家buffer采样
        expert_batch = self.expert_buffer.sample(self.expert_sample_config)
        if expert_batch is not None:
            # 解包（根据replay_memory.py的返回格式）
            # main.py中push到expert_buffer的格式:
            # (img_tensor, critic_state, scaled_MPC_action, NN_action, next_img_tensor, 
            #  next_critic_state, reward, done, final_pi_target, relative_next_target_pos, 
            #  attitude_9d, relative_next_target_vel, fpv_angular_vel)
            *transitions, source = expert_batch
            (exp_img, exp_critic_state, exp_action, _, _, _, _, _, exp_pi_state, 
                exp_rel_pos, exp_attitude, exp_rel_vel, exp_fpv_ang) = transitions
            
            # 转换为Tensor
            exp_img = torch.from_numpy(np.array(exp_img)).float().to(self.device).squeeze(1)
            exp_pi_state = torch.from_numpy(np.array(exp_pi_state)).float().to(self.device)
            exp_target = torch.from_numpy(np.array(exp_action)).float().to(self.device)
            # 准备所有 aux 标签
            exp_rel_pos = torch.from_numpy(np.array(exp_rel_pos)).float().to(self.device)
            exp_attitude = torch.from_numpy(np.array(exp_attitude)).float().to(self.device)
            exp_rel_vel = torch.from_numpy(np.array(exp_rel_vel)).float().to(self.device)
            exp_fpv_ang = torch.from_numpy(np.array(exp_fpv_ang)).float().to(self.device)
        
        # 计算总batches数
        # total_batches = self.ppo_epoch * (self.rollout_buffer.size // self.mini_batch_size)
        # 进度条显示
        pbar = tqdm(desc=f"PPO Update", leave=False) 
        
        # PPO K epochs多轮更新
        for epoch in range(self.ppo_epoch):
            data_loader = self.rollout_buffer.get_data_loader(self.mini_batch_size)
            for batch in data_loader:
                # 解包 batch 数据 (Recurrent PPO 格式)
                # 形状: (batch_size, seq_len, ...)
                # yield返回，每次返回一个minibatch
                (img_seqs, V_states, pi_states, actions, old_log_probs, returns, advantages, old_values,
                 aux_pos, aux_rot, aux_vel, aux_ang, masks, init_hidden) = batch
                batch_size, seq_len = masks.shape
                # 先不在这里展平，保持序列形式传入 evaluate_actions
                # 用当前策略重新评估动作
                # 这里传入 init_hidden，让 GRU 从正确的状态开始
                new_log_probs, entropy, first_aux, second_aux = self.evaluate_actions(
                    img_seqs, pi_states, actions, init_hidden)
                # 展平序列维度用于计算
                new_log_probs_flat = new_log_probs.view(batch_size * seq_len, -1)
                entropy_flat = entropy.view(batch_size * seq_len, -1)
                # (batch, seq_len, ...) -> (batch * seq_len, ...)
                V_states_flat = V_states.view(batch_size * seq_len, -1)
                old_log_probs_flat = old_log_probs.view(batch_size * seq_len, -1)
                returns_flat = returns.view(batch_size * seq_len)
                advantages_flat = advantages.view(batch_size * seq_len)
                old_values_flat = old_values.view(batch_size * seq_len)
                masks_flat = masks.view(batch_size * seq_len)
                
                # 辅助任务目标展平
                # view是用于改变张量形状的方法，但它不复制数据，只是改变"如何解读"内存中的数据
                T =  first_aux.shape[2]
                first_aux_flat = first_aux.view(batch_size * seq_len, T, -1)
                second_aux_flat = second_aux.view(batch_size * seq_len, T, -1)
                aux_pos_flat = aux_pos.view(batch_size * seq_len, T, -1)
                aux_rot_flat = aux_rot.view(batch_size * seq_len, T, -1)
                aux_vel_flat = aux_vel.view(batch_size * seq_len, T, -1)
                aux_ang_flat = aux_ang.view(batch_size * seq_len, T, -1)

                # Advantage对有效位置归一化
                '''
                归一化核心目的：稳定训练，防止梯度过大或过小。
                1. 问题背景：Advantage 的数值范围不稳定
                    Advantage的数值范围取决于：
                    1. Reward 的尺度：如果reward是+100或-50，那Advantage可能是几十甚至上百。
                    2. 训练阶段：初期策略很差，Advantage波动剧烈；后期策略变好，Advantage 趋近于 0。
                    3. Episode长度：长 episode 累积的 GAE 数值更大。
                2. 不归一化会怎样？
                        PPO 的策略梯度核心是clipped surrogate loss, 
                    如果advantage很大（比如+100），梯度会非常大，导致策略剧烈更新，可能一步就"跑飞"。
                    如果advantage很小（比如0.001），梯度会非常小，导致学习极慢。
                    不同 Mini-Batch 之间 Advantage 尺度差异大，训练不稳定。
                3. 归一化如何帮助稳定训练？
                    归一化后，
                        Advantage均值为 0：好动作（A>0）和差动作（A<0）的比例相对平衡。
                        标准差为 1：梯度的尺度始终在一个可控范围内。
                    这样无论 reward 是什么尺度、训练进行到什么阶段，梯度的大小都相对稳定。
                4.  为什么归一化不会破坏训练？
                    1. PPO 关心的是相对大小，而非绝对值
                        PPO 的策略梯度核心逻辑是：
                            Advantage > 0：增加该动作的概率。
                            Advantage < 0：减少该动作的概率。
                        归一化后，虽然个别样本的正负可能翻转，但相对排序和比例关系是保留的
                        原本最好的动作，归一化后仍然是最好的。PPO 会让最好的动作概率增加得最多，最差的动作概率减少得最多。
                5. 什么时候不该归一化？
                    有一种特殊情况需要注意：Batch 太小。
                    如果 mini_batch_size 非常小（比如只有 2-4 个样本），归一化可能会引入过大的噪声
                '''
                valid_advantages = advantages_flat[masks_flat > 0]
                if self.adv_normalization and len(valid_advantages) > 1:
                    adv_mean = valid_advantages.mean()
                    adv_std = valid_advantages.std() + 1e-8
                    # Padding 位置的归一化结果会被 mask 掉
                    advantages_flat = (advantages_flat - adv_mean) / adv_std
                else:
                    advantages_flat = advantages_flat

                new_values = self.critic(V_states_flat)

                # PPO Clipped Surrogate Loss (只计算有效位置)
                ratio = torch.exp(new_log_probs_flat - old_log_probs_flat.squeeze())
                surr1 = ratio * advantages_flat
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_flat
                
                # 使用mask过滤 padding 位置
                policy_surrogate_loss = -torch.min(surr1, surr2)
                # 除以sum使loss只反映每个有效样本的平均质量，与样本数量无关。
                policy_surrogate_loss = (policy_surrogate_loss * masks_flat).sum() / masks_flat.sum()

                # Value Loss (Clipped, 只计算有效位置)
                # v_pred_clipped 被约束为不能偏离 old_values 太远
                v_pred_clipped = old_values_flat + torch.clamp(new_values - old_values_flat, -self.clip_param, self.clip_param)
                v_loss_unclipped = (new_values - returns_flat).pow(2)
                v_loss_clipped = (v_pred_clipped - returns_flat).pow(2)
                '''
                当 new_value 朝正确方向更新但幅度过大时，v_loss_clipped > v_loss_unclipped
                取 max 会选择更大的 loss，起到惩罚过大更新的效果
                '''
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped)
                value_loss = (value_loss * masks_flat).sum() / masks_flat.sum()

                # Entropy (只计算有效位置)
                entropy_loss = (entropy_flat * masks_flat).sum() / masks_flat.sum()
                
                # Auxiliary Task Loss (辅助预测任务，只计算有效位置)
                # 只取有效位置计算 aux_loss
                valid_mask = masks_flat > 0
                # print(f"first_aux_flat shape: {first_aux_flat[valid_mask].shape}")
                # print(f"second_aux_flat shape: {second_aux_flat[valid_mask].shape}")
                # print(f"aux_pos_flat shape: {aux_pos_flat[valid_mask].shape}")
                # print(f"aux_rot_flat shape: {aux_rot_flat[valid_mask].shape}")
                # print(f"aux_vel_flat shape: {aux_vel_flat[valid_mask].shape}")
                # print(f"aux_ang_flat shape: {aux_ang_flat[valid_mask].shape}")
                aux_loss_val = self.aux_loss(
                    first_aux_flat[valid_mask], second_aux_flat[valid_mask],
                    aux_pos_flat[valid_mask], aux_rot_flat[valid_mask],
                    aux_vel_flat[valid_mask], aux_ang_flat[valid_mask])

                # RL总损失 (PPO标准形式)
                rl_loss = policy_surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss
                
                # 用当前策略预测专家动作（行为克隆）
                # Policy使用pi_state作为输入
                _, _, pred_action, pred_first_aux, pred_second_aux, _ = self.policy.sample(exp_img, exp_pi_state)
                il_loss = physics_MSE(pred_action, exp_target)

                il_aux_loss = self.aux_loss(pred_first_aux, pred_second_aux, 
                                            exp_rel_pos, exp_attitude, 
                                            exp_rel_vel, exp_fpv_ang)
                
                aux_loss_val += il_aux_loss
                
                # 总损失
                w_rl = min(1, updates / self.loss_dynamic_change_window) * self.rl_loss_weight_target
                policy_loss = w_rl * rl_loss + (1 - w_rl) * il_loss * self.il_weight + self.aux_loss_weight * aux_loss_val
                
                if updates < self.warm_up_steps:
                    # 计算当前步的学习率：从0线性增长到 base_lr
                    current_lr = (updates / self.warm_up_steps) * self.base_lr
                    
                    # 应用到 Critic 优化器
                    for param_group in self.critic_optim.param_groups:
                        param_group['lr'] = current_lr
                    
                    # 应用到 Policy 优化器
                    for param_group in self.policy_optim.param_groups:
                        param_group['lr'] = current_lr
                
                # --- 优化步骤 ---
                self.policy_optim.zero_grad()
                self.critic_optim.zero_grad()
                # policy_loss中的rl_loss包含了value_loss，会随着计算图传回去
                policy_loss.backward()
                
                # 梯度裁剪（重要: 防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_norm_grad)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_norm_grad)
                
                self.policy_optim.step()
                self.critic_optim.step()
                
                # 累计损失
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_aux_loss += aux_loss_val.item()
                total_il_loss += il_loss.item()
                total_entropy += entropy.mean().item()
                total_rl_loss += rl_loss.item()
                n_updates += 1

                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    'epoch': f'{epoch+1}/{self.ppo_epoch}',
                    'policy': f'{policy_loss.item():.4f}',
                    'value': f'{value_loss.item():.4f}',
                    'rl': f'{rl_loss.item():.4f}',
                    'il': f'{il_loss.item():.4f}',
                    'ratio': f'{ratio.mean().item():.3f}'
                })

        # 关闭进度条
        pbar.close()

        # 清空Rollout Buffer（On-Policy特性）
        self.rollout_buffer.reset()
        
        # 返回平均损失（用于日志）
        avg_policy_loss = total_policy_loss / n_updates if n_updates > 0 else 0
        avg_value_loss = total_value_loss / n_updates if n_updates > 0 else 0
        avg_rl_loss = total_rl_loss / n_updates if n_updates > 0 else 0
        avg_il_loss = total_il_loss / n_updates if n_updates > 0 else 0
        avg_aux_loss = total_aux_loss / n_updates if n_updates > 0 else 0
        return avg_policy_loss, avg_value_loss, avg_rl_loss, avg_il_loss, avg_aux_loss, n_updates

    def save_model(self, filename):
        torch.save(self.policy.state_dict(), filename + "_policy.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")

    def load_model(self, filename, evaluate=False):
        self.policy.load_state_dict(torch.load(filename + "_policy.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(filename + "_critic.pth", map_location=self.device))
        if evaluate:
            self.policy.eval()
            self.critic.eval()
        else:
            self.policy.train()
            self.critic.train()