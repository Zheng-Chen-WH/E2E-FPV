import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import BatchSampler, SubsetRandomSampler
from torch.optim import AdamW
from model import GaussianPolicy, ValueNetwork, init_weights
from utils import six_d_to_rot_mat
from replay_memory import RolloutBuffer, ReplayMemory

class PPO(object):
    def __init__(self, args):
        PPO_dict = args['PPO_param']
        self.device = args['device']
        
        self.gamma = PPO_dict['gamma']
        self.lr = PPO_dict['lr']
        self.clip_param = PPO_dict['clip']
        self.ppo_epoch = PPO_dict['ppo_epoch']
        self.mini_batch_size = PPO_dict['mini_batch_size']
        self.value_loss_coef = PPO_dict['value_loss_coef']
        self.entropy_coef = PPO_dict['entropy_coef']
        self.max_norm_grad = PPO_dict['max_norm_grad']
        
        # Loss Weights
        loss_dict = PPO_dict['loss_weight']
        self.aux_loss_weight = loss_dict['aux_loss_weight']
        self.pos_loss_weight = loss_dict['pos_loss_weight']
        self.rot_loss_weight = loss_dict['rot_loss_weight']
        self.vel_loss_weight = loss_dict['vel_loss_weight']
        self.ang_vel_loss_weight = loss_dict['ang_vel_loss_weight']

        # Buffer
        # PPO:短期 Rollout Buffer (On-Policy)
        self.rollout_buffer = RolloutBuffer(PPO_dict['rollout_buffer'])
        # IL:长期Expert Buffer(Off-Policy)
        self.expert_buffer = ReplayMemory(PPO_dict['buffer_param']) 
        
        self.il_weight = PPO_dict['loss_weight']['il_loss_weight']

        # Networks
        self.critic = ValueNetwork(args['critic_param']).to(self.device)
        self.policy = GaussianPolicy(args['actor_param']).to(self.device)
        
        # Init Weights
        self.critic.apply(init_weights)
        self.policy.apply(init_weights)
        # 特殊初始化：让策略初始输出接近0
        nn.init.constant_(self.policy.log_std_layer.weight, 0)
        nn.init.constant_(self.policy.log_std_layer.bias, 0)
        nn.init.uniform_(self.policy.mu_layer.weight, -PPO_dict['mu_init_boundary'], PPO_dict['mu_init_boundary'])
        nn.init.uniform_(self.policy.mu_layer.bias, -PPO_dict['mu_init_boundary'], PPO_dict['mu_init_boundary'])

        self.critic_optim = AdamW(self.critic.parameters(), lr=self.lr)
        self.policy_optim = AdamW(self.policy.parameters(), lr=self.lr, weight_decay=0.01)
        
        self.hidden_state = None

    def reset(self):
        self.hidden_state = None
        self.rollout_buffer.reset()

    def select_action(self, img_sequence, state, evaluate=False):
        """
        PPO 交互时需要返回 action, log_prob 和 value
        """
        img_sequence = img_sequence.to(self.device)
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            # 1. 获取 Value
            value = self.critic(state_tensor)
            
            # 2. 获取 Action 和 Log Prob
            # 注意：这里我们使用 sample 方法，它内部处理了 hidden_state
            action, log_prob, mean, _, _, new_hidden = self.policy.sample(img_sequence, state_tensor, self.hidden_state)
            
            if evaluate:
                action = torch.tanh(mean) # 确定性策略

        if new_hidden is not None:
            self.hidden_state = new_hidden
            
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]

    def evaluate_actions(self, img_seq, state, action):
        """
        在 Update 循环中评估动作的概率 (Re-evaluate)
        """
        # 注意：这里不能用 sample，因为我们要评估的是 buffer 里的 action
        # 我们需要手动调用 forward 然后计算 log_prob
        mean, log_std, first_aux, second_aux, _ = self.policy(img_seq, state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        
        # --- Tanh 分布修正 (关键) ---
        # 1. 限制 action 范围防止 atanh 溢出
        action_clipped = torch.clamp(action, -0.999999, 0.999999)
        # 2. 反推 pre-tanh 值
        x_t = torch.atanh(action_clipped)
        # 3. 计算 log_prob
        log_prob = normal.log_prob(x_t)
        # 4. 减去 Jacobian log determinant
        log_prob -= torch.log(1 - action_clipped.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        entropy = normal.entropy().sum(1, keepdim=True)
        
        return log_prob, entropy, first_aux, second_aux

    def aux_loss(self, first_aux, second_aux, gt_pos, gt_rot, gt_vel, gt_ang):
        """计算辅助损失"""
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
        
        # 这里需要引入 utils 里的转换函数，或者把那个函数变成静态方法
        # 假设 six_d_to_rot_mat 已经 import
        R_pred = six_d_to_rot_mat(pred_rot_flat)
        loss_rot = F.mse_loss(R_pred, gt_rot_flat)

        return (self.pos_loss_weight * loss_pos + 
                self.rot_loss_weight * loss_rot + 
                self.vel_loss_weight * loss_vel + 
                self.ang_vel_loss_weight * loss_ang)

    def update(self):
        """
        PPO 核心更新逻辑
        """
        # 1. 准备数据 (计算 GAE 等)
        # 注意：需要在 main.py 中调用 buffer.finish_path() 后再调用 update
        
        data_loader = self.buffer.get_data_loader(self.mini_batch_size)
        
        total_policy_loss = 0
        total_value_loss = 0
        
        # 2. PPO Epoch 循环
        for _ in range(self.ppo_epoch):
            for batch in data_loader:
                (img_seqs, states, actions, old_log_probs, returns, advantages, old_values,
                 aux_pos, aux_rot, aux_vel, aux_ang) = batch

                # 归一化 Advantage
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # --- 评估当前策略 ---
                new_log_probs, entropy, first_aux, second_aux = self.evaluate_actions(img_seqs, states, actions)
                new_values = self.critic(states)

                # --- Ratio & Surrogate Loss ---
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- Value Loss (Clipped) ---
                v_pred_clipped = old_values + torch.clamp(new_values - old_values, -self.clip_param, self.clip_param)
                v_loss_unclipped = F.mse_loss(new_values, returns)
                v_loss_clipped = F.mse_loss(v_pred_clipped, returns)
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped)

                # --- Aux Loss ---
                aux_loss_val = self.aux_loss(first_aux, second_aux, aux_pos, aux_rot, aux_vel, aux_ang)

                # --- Total Loss ---
                rl_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

                il_loss = 0.0
                if self.il_weight > 0 and self.expert_buffer.__len__('expert') > self.mini_batch_size:
                    # 从专家 Buffer 中随机采样
                    # 注意：这里不需要计算 log_prob 或 advantage，只需要 state 和 expert_action
                    expert_batch = self.expert_buffer.sample({'expert': self.mini_batch_size})
                    
                    # 解包 expert_batch (根据你 replay_memory.py 的返回格式)
                    # 假设返回: (pi_img, q_state, mpc_action, ...)
                    (exp_img, _, exp_action, _, _, _, _, _, _, _, _, _, _, _) = expert_batch
                    
                    exp_img = torch.FloatTensor(exp_img).to(self.device).squeeze(1)
                    exp_target = torch.FloatTensor(exp_action).to(self.device)
                    
                    # 让当前 Policy 预测专家状态下的动作
                    # 这里的 mean 是确定性动作，适合模仿
                    pred_mean, _, _, _, _ = self.policy(exp_img, None) # 假设 policy 接受 None state
                    pred_action = torch.tanh(pred_mean)
                    
                    il_loss = F.mse_loss(pred_action, exp_target)
                
                # 总 Loss
                loss = (rl_loss + 
                        self.aux_loss_weight * aux_loss_val + 
                        self.il_weight * il_loss)
                # --- Optimize ---
                self.policy_optim.zero_grad()
                self.critic_optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_norm_grad)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_norm_grad)
                self.policy_optim.step()
                self.critic_optim.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

        # 清空 Buffer
        self.buffer.reset()
        
        return total_policy_loss, total_value_loss

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