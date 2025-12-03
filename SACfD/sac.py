import os
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from utils import soft_update, hard_update, six_d_to_rot_mat, physics_MSE
from model import GaussianPolicy, QNetwork, init_weights
import torch.nn as nn
import time
import numpy as np
from replay_memory import ReplayMemory

class SAC(object):
    def __init__(self, args):

        # 解包SAC算法参数
        SAC_dict = args['SAC_param']
        self.gamma = SAC_dict['gamma']
        self.tau = SAC_dict['tau']
        self.alpha = SAC_dict['alpha']
        self.seed = SAC_dict['seed']
        self.target_update_interval = SAC_dict['target_update_interval']
        self.automatic_entropy_tuning = SAC_dict['automatic_entropy_tuning']
        self.warm_up_steps = SAC_dict['warm_up_steps']
        self.base_lr = SAC_dict['lr']
        self.max_norm_grad = SAC_dict['max_norm_grad']
        self.chunk_update = SAC_dict['chunk_update']  # 是否使用序列更新及推理时是否每步重置隐藏状态为None

        # il与rl损失动态混合机制参数
        self.loss_dynamic_change_window = SAC_dict['loss_dynamic_change_window']
        self.rl_loss_weight_target = SAC_dict['rl_loss_weight_target']

        # 解包损失函数权重字典
        loss_dict = SAC_dict['loss_weight']
        self.aux_loss_weight = loss_dict['aux_loss_weight']
        self.pos_loss_weight = loss_dict['pos_loss_weight']
        self.rot_loss_weight = loss_dict['rot_loss_weight']
        self.vel_loss_weight = loss_dict['vel_loss_weight']
        self.ang_vel_loss_weight = loss_dict['ang_vel_loss_weight']
        self.il_weight = loss_dict['il_loss_weight']

        # 初始化灵活定义版buffer
        self.memory = ReplayMemory(SAC_dict['buffer_param'])
        self.training_args = SAC_dict['batch_size']

        # 定义设备
        self.device = args['device']

        # 定义各网络与初始化

        # 定义并初始化critic
        self.critic = QNetwork(args['critic_param']).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), self.base_lr)
        self.critic_target = QNetwork(args['critic_param']).to(self.device)
        self.critic.apply(init_weights)
        nn.init.uniform_(self.critic.Q_network_1[-2].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.critic.Q_network_1[-2].bias, -3e-3, 3e-3)
        nn.init.uniform_(self.critic.Q_network_2[-2].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.critic.Q_network_2[-2].bias, -3e-3, 3e-3)
        hard_update(self.critic_target, self.critic) #初始化的时候直接硬更新
        
        # 定义并初始化alpha自动调整
        # Target Entropy = −dim(A)，原论文直接认为目标熵就是动作空间维度乘积的负值，在这里就是Box的“体积”
        if self.automatic_entropy_tuning is True:
            # self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item() #torch.prod()是一个函数，用于计算张量中所有元素的乘积
            self.target_entropy = - SAC_dict['action_dim'] # 对于一维动作空间向量，目标值就是这个
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device) #初始化log_alpha
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.base_lr)
        else:
            self.alpha = SAC_dict['alpha']

        # 定义并初始化policy
        self.policy = GaussianPolicy(args['actor_param']).to(self.device)
        self.policy.apply(init_weights)  
        nn.init.constant_(self.policy.log_std_layer.weight, 0)
        nn.init.constant_(self.policy.log_std_layer.bias, -1.5) #让初始log_std偏小一些
        nn.init.uniform_(self.policy.mu_layer.weight, - SAC_dict['mu_init_boundary'], SAC_dict['mu_init_boundary']) 
        nn.init.uniform_(self.policy.mu_layer.bias, - SAC_dict['mu_init_boundary'], SAC_dict['mu_init_boundary'])
        self.policy_optim = AdamW(self.policy.parameters(), self.base_lr, weight_decay = 0.01)
        self.hidden_state = None

        if self.automatic_entropy_tuning:
            self.alpha_t = self.alpha
        else:
            self.alpha_t = torch.tensor(self.alpha).to(self.device)

    def reset(self): # 为了发挥GRU时序能力，每次训练前要重置GRU的隐藏状态
        self.hidden_state = None
    
    def aux_loss(self, first_aux_output, second_aux_output, gt_pos, gt_rot_mat, gt_vel, gt_ang_vel): # gr:ground truth
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
        # 切分预测值
        # 感知模块的输出结果(B, T, 9)
        pred_pos = first_aux_output[..., 0:3] # 切出相对位置
        pred_rot_6d = first_aux_output[..., 3:9] # 切出相对姿态

        # 时序模块的输出(B, T, 6)
        pred_vel = second_aux_output[..., 0:3] # 切出相对速度
        pred_ang_vel = second_aux_output[..., 3:6] # 相对角速度

        # 对位置速度角速度直接算mse
        loss_pos = F.mse_loss(pred_pos, gt_pos)
        loss_vel = F.mse_loss(pred_vel, gt_vel)
        loss_ang_vel = F.mse_loss(pred_ang_vel, gt_ang_vel)

        # 姿态要转9D旋转矩阵，所以要对批次处理一下
        pred_rot_6d_flat = pred_rot_6d.reshape(-1, 6)
        gt_rot_mat_flat = gt_rot_mat.reshape(-1, 3, 3)
        R_pred_flat = six_d_to_rot_mat(pred_rot_6d_flat)
        loss_rot = F.mse_loss(R_pred_flat, gt_rot_mat_flat)
        # print(f"pred_pos:{pred_pos[0:5]}, true_pos:{gt_pos[0:5]}")
        # print(f"pred_rot:{R_pred_flat[0:5]}, true_rot:{gt_rot_mat_flat[0:5]}")
        # print(f"pred_vel:{pred_vel[0:5]}, true_vel:{gt_vel[0:5]}")
        # print(f"pred_ang:{pred_ang_vel[0:5]}, true_ang:{gt_ang_vel[0:5]}")
        # 加权求和
        total_loss = (self.pos_loss_weight * loss_pos +
                  self.rot_loss_weight * loss_rot +
                  self.vel_loss_weight * loss_vel +
                  self.ang_vel_loss_weight * loss_ang_vel)

        return total_loss, loss_pos, loss_rot, loss_vel, loss_ang_vel

    def select_action(self, img_sequence, state, V_state, evaluate=False):
        """输入图片序列与目标位置，返回动作；仅用在main.py前向传播中，训练时直接用sample函数
        所以训练时自然而然就有hidden_state总为none

        Args:
            img_sequence (_type_): 图片序列张量
            state (_type_): 目标位置 (归一化值)
            evaluate (bool, optional): 是否为评价模式(输出正态分布取样值还是均值) Defaults to False.

        Returns:
            action: action
        """
        img_sequence=img_sequence.to(self.device)
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        # print(f"reasoning hidden state used in action selection: {self.hidden_state}")
        if evaluate is False:
            action, _, _, _, _, _, new_hidden = self.policy.sample(img_sequence, state, self.hidden_state)
        else:
            _, _, action, _, _, _, new_hidden = self.policy.sample(img_sequence, state, self.hidden_state) #如果evaluate为True，输出的动作是网络的mean经过squash的结果
        # 如果使用GRU，更新Agent的隐藏状态，为下一次决策做准备
        if new_hidden is not None and self.chunk_update:
            self.hidden_state = new_hidden.detach() # 使用 detach() 避免梯度累积
        return action.detach().cpu().numpy()[0], None, None, self.hidden_state # 返回numpy格式的动作，同时对齐PPO接口

    def push_data(self, source, data):
        """将经验存入buffer

        Args:
            source (string): 描述数据要存入的buffer名称
            data (tuple): 包含（所有数据）的一个元组，所有buffer存入数据类型都相同
        """
        self.memory.push(source, data)
    
    def check_buffer_len(self, buffer_name):
        """查询buffer容量

        Args:
            buffer (string): 所要查询的buffer名称

        Returns:
            length (int): 所查询的buffer长度
        """
        length = self.memory.__len__(buffer_name)
        return length

    def update(self, updates):
        """网络参数更新

        Args:
            updates (int): 更新次数

        Returns:
            _type_: _description_
        """
        # 数据准备
        sampled_data = self.memory.sample(self.training_args)

        if sampled_data is None:
            # print("buffer中没有足够的数据进行采样。")
            return None, None, None, None, None
        
        # 动态解包数据，最后一项是来源标志
        """Python特性: 可迭代对象解包，使用了星号*的扩展形式。
           当星号*出现在赋值语句的左侧时,它的意思是：“把所有剩余的项都收集到一个新的列表中”。
           a, *b, c = list, 自动得到a=list[0], c=list[-1], b构建list中其他项组成的列表"""
        *transitions, source = sampled_data
        (pi_img, q_state, mpc_action, nn_action, next_pi_img, next_q_state, reward,
            done, goal, aux_pos, aux_att, aux_vel, aux_ang) = transitions
        
        # 创建布尔掩码取代复杂拼接和切片
        source = np.array(source) # 转换为numpy以便进行逻辑操作
        expert_mask = (source == 'expert')
        dagger_old_mask = (source == 'dagger_old')
        dagger_recent_mask = (source == 'dagger_recent')
        
        # 以取并集的方式灵活调整掩码
        # dagger数据的nn action才有用
        dagger_mask = dagger_old_mask | dagger_recent_mask
        # 用于模仿的数据
        il_mask = expert_mask | dagger_mask # expert, old, recent都可用于模仿
        # 辅助任务(Aux)使用所有数据
        aux_mask = expert_mask | dagger_mask


        # 准备Q函数（Critic）更新所需的数据
        """
        numpy.where 是一个三元操作符，基本语法是：np.where(condition, x, y)
            它会检查 condition 数组中的每一个元素。
            如果 condition 中某个位置的元素是 True，它就从 x 数组的对应位置取值。
            如果 condition 中某个位置的元素是 False，它就从 y 数组的对应位置取值。
            最终返回一个和 condition 形状相同的新数组。
        expert_mask[:, np.newaxis]
            解决形状不匹配问题。
            is_expert_mask 的形状是 (batch_size,)，mpc_action 和 nn_action 的形状是 (batch_size, action_dim)。
            不能直接用一个 (256,) 的掩码去对一个 (256, 6) 的数组进行 if-else 选择。
            np.where 要求 condition 的形状能被广播到与 x 和 y 的形状相匹配，np.newaxis的作用是在指定位置增加一个新的维度，expert_mask[:, np.newaxis] 的形状是 (256, 1)
        对于第 i 行，它检查 is_expert_mask[i] 的值。
        如果 is_expert_mask[i] 是 True，那么 critic_action 的整个第 i 行都将被设置为 mpc_action 的第 i 行。
        如果 is_expert_mask[i] 是 False，那么 critic_action 的整个第 i 行都将被设置为 nn_action 的第 i 行。
        Critic使用所有有效的 (s, a, r, s') transition
        对于 expert 数据，实际执行的动作是 mpc_action；对于 exploration 数据，实际执行的动作是 nn_action。
        如果 这条数据来自 'expert' (专家) Buffer，那么就选择 mpc_action (专家动作) 作为Q函数要评估的动作。
        否则 (即数据来自 'exploration' Buffer)，就选择 nn_action (智能体自己执行的动作) 作为Q函数要评估的动作。
        """
        critic_action = np.where(expert_mask[:, np.newaxis], mpc_action, nn_action)
        
        # 将所有数据转换为Tensor
        # 取出所有采样的数据，它们都可用于off-policy的Critic训练
        q_state_batch = torch.from_numpy(q_state).float().to(self.device)
        action_batch = torch.from_numpy(critic_action).float().to(self.device)
        reward_batch = torch.from_numpy(reward).float().to(self.device).unsqueeze(1)
        next_q_state_batch = torch.from_numpy(next_q_state).float().to(self.device)
        done_batch = torch.from_numpy(done).float().to(self.device).unsqueeze(1)
        next_pi_img_batch = torch.from_numpy(next_pi_img).float().to(self.device).squeeze(1)
        next_goal_batch = torch.from_numpy(goal).float().to(self.device) # 'goal' 和 'next_goal' 通常是相同的

        # 准备策略（Policy）更新所需的数据, 策略更新分为 IL 部分和 RL 部分
        
        # 模仿学习，模仿所有MPC的动作
        # 掩码的使用方式和索引一样
        il_pi_img_batch = torch.from_numpy(pi_img[il_mask]).float().to(self.device).squeeze(1)
        il_goal_batch = torch.from_numpy(goal[il_mask]).float().to(self.device)
        il_target_action_batch = torch.from_numpy(mpc_action[il_mask]).float().to(self.device)

        # 强化学习仅使用NN动作数据
        # 策略需要从自己行为中学习
        rl_policy_mask = dagger_mask
        rl_pi_img_batch = torch.from_numpy(pi_img[rl_policy_mask]).float().to(self.device).squeeze(1)
        rl_goal_batch = torch.from_numpy(goal[rl_policy_mask]).float().to(self.device)
        rl_q_state_for_policy_batch = torch.from_numpy(q_state[rl_policy_mask]).float().to(self.device)

        # 辅助头
        aux_pos_batch = torch.from_numpy(aux_pos[aux_mask]).float().to(self.device)
        aux_att_batch = torch.from_numpy(aux_att[aux_mask]).float().to(self.device)
        aux_vel_batch = torch.from_numpy(aux_vel[aux_mask]).float().to(self.device)
        aux_ang_batch = torch.from_numpy(aux_ang[aux_mask]).float().to(self.device)

        # LR热启动
        if updates < self.warm_up_steps:
            # 计算当前步的学习率：从0线性增长到 base_lr
            current_lr = (updates / self.warm_up_steps) * self.base_lr
            
            # 应用到 Critic 优化器
            for param_group in self.critic_optim.param_groups:
                param_group['lr'] = current_lr
            
            # 应用到 Policy 优化器
            for param_group in self.policy_optim.param_groups:
                param_group['lr'] = current_lr

        # Critic Loss
        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _, _, _, _ = self.policy.sample(next_pi_img_batch, next_goal_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_q_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            target_q_value = reward_batch + (1 - done_batch) * self.gamma * (min_qf_next_target - self.alpha_t * next_state_log_pi)
        qf1, qf2 = self.critic(q_state_batch, action_batch)
        # print(f"min_qf_next_target:{torch.mean(min_qf_next_target)}")
        # print(f"reward:{torch.mean(rl_reward_batch)}, target_q_value:{torch.mean(target_q_value)}, qf1:{torch.mean(qf1)}, qf2:{torch.mean(qf2)}") 
        qf_loss = F.mse_loss(qf1, target_q_value) + F.mse_loss(qf2, target_q_value)
        # print(f"q_loss:{qf_loss}")

        self.critic_optim.zero_grad()
        qf_loss.backward()
        # max_norm 指定了梯度范数（梯度向量长度）的上限阈值。
        # 当计算出的梯度范数超过这个值时，所有梯度会被等比例缩放，使得最终范数恰好等于 max_norm。范数不超过该阈值时，梯度保持不变。
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm = self.max_norm_grad)
        self.critic_optim.step()

        # Policy Loss
        # 为了计算所有loss，我们需要对所有相关的输入进行一次前向传播
        # 将 IL 和 RL 的输入合并，只进行一次 policy 网络计算，提高效率
        combined_pi_img_batch = torch.cat((il_pi_img_batch, rl_pi_img_batch), dim=0)
        combined_goal_batch = torch.cat((il_goal_batch, rl_goal_batch), dim=0)
        
        # 获取 mean (确定性动作) 用于 IL
        pi, log_pi, mean, std, first_aux_output, second_aux_output, _ = self.policy.sample(combined_pi_img_batch, combined_goal_batch)
        
        # 分离 IL 和 RL 的输出
        num_il_samples = il_pi_img_batch.shape[0]
        pi_il = mean[:num_il_samples] # 使用确定性动作 (Mean) 计算 IL Loss
        pi_rl = pi[num_il_samples:]
        log_pi_rl = log_pi[num_il_samples:]
        
        # IL Loss
        il_loss = physics_MSE(pi_il, il_target_action_batch)
        
        # RL Loss
        qf1_pi, qf2_pi = self.critic(rl_q_state_for_policy_batch, pi_rl)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        rl_loss = ((self.alpha_t.detach() * log_pi_rl) - min_qf_pi).mean()
        
        # Aux Loss
        # first_aux_output和second_aux_output也需要被分离以匹配维度
        first_aux_output = first_aux_output[:num_il_samples] # aux与IL维度相同
        second_aux_output = second_aux_output[:num_il_samples]
        aux_loss, aux_l_pos, aux_l_rot, aux_l_vel, aux_l_ang = self.aux_loss(first_aux_output, second_aux_output, aux_pos_batch, aux_att_batch,
                                aux_vel_batch, aux_ang_batch)

        # 动态权重调整
        w_rl = min(1, updates / self.loss_dynamic_change_window) * self.rl_loss_weight_target
        total_policy_loss = w_rl * rl_loss + (1 - w_rl) * il_loss * self.il_weight + self.aux_loss_weight * aux_loss
        
        self.policy_optim.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm = self.max_norm_grad)
        self.policy_optim.step()

        # Alpha Loss
        alpha_loss = torch.tensor(0.).to(self.device)
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi_rl.detach() + self.target_entropy)).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
            self.alpha_t = self.alpha.clone()
        else:
            self.alpha_t = torch.tensor(self.alpha).to(self.device)

        # 更新Target Network
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        # 详细打印loss
        if isinstance(self.alpha, torch.Tensor):
            alpha_val = self.alpha.item()
        else:
            alpha_val = self.alpha
        if updates % 10 == 0:
            # 计算一些额外的统计量用于打印
            q_val = min_qf_pi.mean().item()
            entropy_term = (-self.alpha * log_pi_rl).mean().item()
            avg_reward = reward_batch.mean().item()
            avg_target_q = target_q_value.mean().item()
            
            # 计算方差的均值
            avg_std = std.mean().item()

            
            print(f"| Upd: {updates:5d} | Tot: {total_policy_loss.item():6.2f} | Q_Loss: {qf_loss.item():6.2f} | "
                  f"RL: {rl_loss.item():6.2f} | IL: {il_loss.item():6.2f} | Aux: {aux_loss.item():6.2f} |\n"
                  f"|> RL_Detail | Q_Val: {q_val:6.2f} | Ent_Term: {entropy_term:6.2f} | Alpha: {alpha_val:.3f} | Ent: {-log_pi_rl.mean().item():.2f} |\n"
                  f"|> Critic_Det| Pred_Q: {qf1.mean().item():6.2f} | Targ_Q: {avg_target_q:6.2f} | Avg_R: {avg_reward:6.2f} |\n"
                  f"|> Aux_Detail| Pos: {aux_l_pos.item():.3f} | Rot: {aux_l_rot.item():.3f} | Vel: {aux_l_vel.item():.3f} | Ang: {aux_l_ang.item():.3f} |\n"
                  f"|> Weights   | W_RL: {w_rl:.2f} | W_IL: {(1 - w_rl) * self.il_weight:.2f} |\n"
                  f"|> Std       | Avg_Std: {avg_std:.3f} |\n"
                  "---------------------------------------------------------------")
        
        return total_policy_loss.item(), qf_loss.item(), rl_loss.item(), il_loss.item(), aux_loss.item()

    # Save model parameters
    def save_model(self, filename="master"):
        '''if not os.path.exists('GoodModel/'):
            os.makedirs('GoodModel/')'''

        ckpt_path = filename + "_model.pt"
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_model(self, file_name, evaluate=False):
        if file_name is not None:
            checkpoint = torch.load(file_name + "_model.pt", weights_only=False)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()