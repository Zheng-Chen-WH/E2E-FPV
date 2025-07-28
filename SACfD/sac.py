import os
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from utils import soft_update, hard_update, weighted_mse_loss, physics_MSE
from model import GaussianPolicy, QNetwork, init_weights
import torch.nn as nn
import time
import numpy as np

class SAC(object):
    def __init__(self, args):
        self.gamma = args['gamma']
        self.tau = args['tau']
        self.alpha = args['alpha']
        self.seed=args['seed']
        self.target_update_interval = args['target_update_interval']
        self.automatic_entropy_tuning = args['automatic_entropy_tuning']
        self.warm_up_steps = args['warm_up']
        self.base_lr = args['lr']
        self.aux_loss_weight = args['aux_loss_weight']
        self.pos_loss_weight = args['pos_loss_weight']
        self.rot_loss_weight = args['rot_loss_weight']
        self.vel_loss_weight = args['vel_loss_weight']
        self.ang_vel_loss_weight = args['ang_vel_loss_weight']
        self.dagger_weight = args['dagger_loss_weight']

        self.device = torch.device("cuda" if args['cuda'] else "cpu")

        self.critic = QNetwork(args['Q_network_dim'], args['action_dim'], args['hidden_sizes'], args['activation']).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), self.base_lr)

        self.critic_target = QNetwork(args['Q_network_dim'], args['action_dim'], args['hidden_sizes'], args['activation']).to(self.device)
        self.critic.apply(init_weights)
        nn.init.uniform_(self.critic.Q_network_1[-2].weight, -1e-3, 1e-3)
        nn.init.uniform_(self.critic.Q_network_2[-2].weight, -1e-3, 1e-3)
        hard_update(self.critic_target, self.critic) #初始化的时候直接硬更新
        self.critic_target.apply(init_weights)
        nn.init.uniform_(self.critic_target.Q_network_1[-2].weight, -1e-3, 1e-3)
        nn.init.uniform_(self.critic_target.Q_network_2[-2].weight, -1e-3, 1e-3)

        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning is True: #原论文直接认为目标熵就是动作空间维度乘积的负值，在这里就是Box的“体积”
            # self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item() #torch.prod()是一个函数，用于计算张量中所有元素的乘积
            self.target_entropy = - args['action_dim'] # 对于一维动作空间向量，目标值就是这个
            self.alpha = torch.zeros(1, requires_grad=True, device=self.device) #原论文没用log，但是这里用的，总之先改成无log状态试试
            #self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device) #初始化log_alpha
            self.alpha_optim = Adam([self.alpha], lr=self.base_lr)

        self.policy = GaussianPolicy(args['embedding_dim'], args['Pi_mlp_dim'], args['action_dim'], args['hidden_sizes'], 
                                     args['activation'], args['max_action'], args['min_action'],
                                     args['resnet_aux_dim'], args['gru_aux_dim'],
                                     args['gru_layer'], args['drop_out']).to(self.device)
        self.policy.apply(init_weights)  
        nn.init.constant_(self.policy.log_std_layer.weight, 0)
        nn.init.constant_(self.policy.log_std_layer.bias, 0)
        nn.init.uniform_(self.policy.mu_layer.weight, -1.0, 1.0) 
        self.policy_optim = AdamW(self.policy.parameters(), self.base_lr, weight_decay = 0.01) # Gemini说transformer适合用
        self.hidden_state = None
        self.avg_td_error = None
        self.avg_disagreement = None
         # 带棘轮效应的动态基线参数
        self.baseline_update_window = args['baseline_update_window']
        self.baseline_update_gamma = args['baseline_update_gamma']
        # 归一化分母（基线），初始化为1.0
        self.baseline_td = 1.0
        self.baseline_dis = 1.0
        
        # 目标基线，在第一次评估时设定
        self.target_baseline_td = 1.0
        self.target_baseline_dis = 1.0

        # 初始化基线下降的每步增量初始化为0
        self.delta_baseline_td = 0.0
        self.delta_baseline_dis = 0.0

        # 用于在每个窗口内收集数据
        self._window_td_errors = []
        self._window_disagreements = []
        self._is_initial_baseline_set = False # 标记是否已完成第一次基线设置
        self.k_final = args['k_final'] 
        self.k_rl_threshold = args['k_rl_threshold']
        self.avg_il_loss = None

    def reset(self): # 为了发挥GRU时序能力，现在每次训练前要重置GRU的隐藏状态
        self.hidden_state = None

    def six_d_to_rot_mat(self, pred_6d):
        """
        将(N, 6)的6D表示转换为(N, 3, 3)的旋转矩阵.
        这个函数不知道也不关心 N 是 B 还是 B*T，它只是独立处理N个样本。
        """
        # 提取列向量
        a1 = pred_6d[..., 0:3]
        a2 = pred_6d[..., 3:6]
        # 格拉姆-施密特正交化
        b1 = F.normalize(a1, dim=-1)
        dot_product = torch.sum(b1 * a2, dim=-1, keepdim=True)
        a2_orthogonal = a2 - dot_product * b1
        b2 = F.normalize(a2_orthogonal, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)
    
    def aux_loss(self, resnet_output, gru_output, gt_pos, gt_rot_mat, gt_vel, gt_ang_vel): # gr:ground truth
        """
        从DAgger取数据计算模仿学习loss
        """
        # 切分预测值
        # resnet输出结果(B, T, 9)normalized_input
        pred_pos = resnet_output[..., 0:3] # 切出相对位置
        pred_rot_6d = resnet_output[..., 3:9] # 切出相对姿态

        # gru输出(B, T, 6)
        pred_vel = gru_output[..., 0:3] # 切出相对速度
        pred_ang_vel = gru_output[..., 3:6] # 相对角速度

        # 对位置速度角速度直接算mse
        loss_pos = F.mse_loss(pred_pos, gt_pos)
        loss_vel = F.mse_loss(pred_vel, gt_vel)
        loss_ang_vel = F.mse_loss(pred_ang_vel, gt_ang_vel)

        # 姿态要转9D旋转矩阵，所以要对批次处理一下
        pred_rot_6d_flat = pred_rot_6d.reshape(-1, 6)
        gt_rot_mat_flat = gt_rot_mat.reshape(-1, 3, 3)
        R_pred_flat = self.six_d_to_rot_mat(pred_rot_6d_flat)
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

        return total_loss

    def select_action(self,img_sequence, state, evaluate=False):
        img_sequence=img_sequence.to(self.device)
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _, _, _, new_hidden = self.policy.sample(img_sequence, state, self.hidden_state)
        else:
            _, _, action, _, _, new_hidden = self.policy.sample(img_sequence, state, self.hidden_state) #如果evaluate为True，输出的动作是网络的mean经过squash的结果
        # 更新Agent的隐藏状态，为下一次决策做准备
        self.hidden_state = new_hidden.detach() # 使用 detach() 避免梯度累积
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, expert_memory, dagger_memory, exploration_memory, batch_size, updates):
    # 数据准备阶段 (在CPU上准备好所有数据, 然后一次性转移)

        # 1. 从各个Buffer采样 (仍然是Numpy数组)
        exp_pi_img, exp_q_state, exp_action, exp_reward, exp_next_pi_img, \
            exp_next_q_state, exp_done, exp_goal, exp_res_pos, exp_res_att, \
            exp_gru_vel, exp_gru_ang = expert_memory.sample(batch_size=batch_size)

        expl_pi_img, expl_q_state, expl_action, expl_reward, expl_next_pi_img, \
            expl_next_q_state, expl_done, expl_goal, expl_res_pos, expl_res_att, \
            expl_gru_vel, expl_gru_ang = exploration_memory.sample(batch_size=batch_size)

        dag_pi_img, dag_action, dag_goal, dag_res_pos, dag_res_att, \
            dag_gru_vel, dag_gru_ang = dagger_memory.sample(batch_size=batch_size)

        # CPU上构建用于Critic更新的RL批次 (expert + exploration)
        rl_q_state_batch = torch.cat((torch.from_numpy(exp_q_state), torch.from_numpy(expl_q_state)), dim=0)
        rl_action_batch = torch.cat((torch.from_numpy(exp_action), torch.from_numpy(expl_action)), dim=0)
        rl_reward_batch = torch.cat((torch.from_numpy(exp_reward), torch.from_numpy(expl_reward)), dim=0).unsqueeze(1)
        rl_next_q_state_batch = torch.cat((torch.from_numpy(exp_next_q_state), torch.from_numpy(expl_next_q_state)), dim=0)
        rl_done_batch = torch.cat((torch.from_numpy(exp_done), torch.from_numpy(expl_done)), dim=0).unsqueeze(1)
        rl_next_pi_img_batch = torch.cat((torch.from_numpy(exp_next_pi_img), torch.from_numpy(expl_next_pi_img)), dim=0)
        rl_next_goal_batch = torch.cat((torch.from_numpy(exp_goal), torch.from_numpy(expl_goal)), dim=0)
        
        # CPU上构建用于Policy更新的统一批次 (dagger + expert + exploration)
        # 把所有用于策略更新的状态和目标都拼接起来，后面用切片的方法再分开
        policy_pi_img_batch = torch.cat((torch.from_numpy(dag_pi_img), torch.from_numpy(exp_pi_img), torch.from_numpy(expl_pi_img)), dim=0)
        policy_goal_batch = torch.cat((torch.from_numpy(dag_goal), torch.from_numpy(exp_goal), torch.from_numpy(expl_goal)), dim=0)
        
        # policy_q_state批次包含了策略网络需要评估的所有状态
        policy_q_state_batch = torch.cat((torch.from_numpy(exp_q_state), torch.from_numpy(expl_q_state)), dim=0) # Dagger数据没有Q状态

        # 用于计算IL Loss的“真值”动作 (dagger + expert)
        # 注意：探索数据没有对应的专家动作，我们可以用零向量填充，但更简单的做法是在计算loss时只使用相应部分
        il_action_batch = torch.cat((torch.from_numpy(dag_action), torch.from_numpy(exp_action)), dim=0)
        
        # 用于辅助损失的真值
        policy_res_pos_batch = torch.cat((torch.from_numpy(dag_res_pos), torch.from_numpy(exp_res_pos), torch.from_numpy(expl_res_pos)), dim=0)
        policy_res_att_batch = torch.cat((torch.from_numpy(dag_res_att), torch.from_numpy(exp_res_att), torch.from_numpy(expl_res_att)), dim=0)
        policy_gru_vel_batch = torch.cat((torch.from_numpy(dag_gru_vel), torch.from_numpy(exp_gru_vel), torch.from_numpy(expl_gru_vel)), dim=0)
        policy_gru_ang_batch = torch.cat((torch.from_numpy(dag_gru_ang), torch.from_numpy(exp_gru_ang), torch.from_numpy(expl_gru_ang)), dim=0)

        # 一次性将所有数据转移到GPU
        device = self.device
        rl_q_state_batch, rl_action_batch, rl_reward_batch, rl_next_q_state_batch, rl_done_batch, \
        rl_next_pi_img_batch, rl_next_goal_batch = [t.float().to(device) for t in [rl_q_state_batch, rl_action_batch, rl_reward_batch, rl_next_q_state_batch, rl_done_batch, rl_next_pi_img_batch, rl_next_goal_batch]]
        
        policy_pi_img_batch, policy_goal_batch, policy_q_state_batch, il_action_batch, \
        policy_res_pos_batch, policy_res_att_batch, policy_gru_vel_batch, policy_gru_ang_batch = \
            [t.float().to(device) for t in [policy_pi_img_batch, policy_goal_batch, policy_q_state_batch, il_action_batch, policy_res_pos_batch, policy_res_att_batch, policy_gru_vel_batch, policy_gru_ang_batch]]

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

        # Critic 网络更新 (逻辑不变, 使用准备好的rl批次)
        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _, _, _ = self.policy.sample(rl_next_pi_img_batch, rl_next_goal_batch) 
            qf1_next_target, qf2_next_target = self.critic_target(rl_next_q_state_batch, next_state_action) 
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) 
            target_q_value = rl_reward_batch + (1 - rl_done_batch) * self.gamma * (min_qf_next_target - self.alpha * next_state_log_pi)
        qf1, qf2 = self.critic(rl_q_state_batch, rl_action_batch)  
        print(f"min_qf_next_target:{torch.mean(min_qf_next_target)}")
        print(f"reward:{torch.mean(rl_reward_batch)}, target_q_value:{torch.mean(target_q_value)}, qf1:{torch.mean(qf1)}, qf2:{torch.mean(qf2)}") 
        qf_loss = F.mse_loss(qf1, target_q_value) + F.mse_loss(qf2, target_q_value)
        print(f"q_loss:{qf_loss}")
        
        self.critic_optim.zero_grad()
        qf_loss.backward() 
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optim.step()

        # Policy 网络更新
        # 对全部数据统一的策略批次进行一次前向传播
        pi, log_pi, _, resnet_output, gru_output, _ = self.policy.sample(policy_pi_img_batch, policy_goal_batch)
        
        # 计算IL损失组件 (只使用pi中对应dagger和expert的部分)
        # il_action_batch的大小是 (dagger的batch_size + exp的batch_size)
        # policy_batch的顺序是dagger+expert+exploration
        il_slice_size = il_action_batch.shape[0]
        il_policy_loss_component = physics_MSE(pi[:il_slice_size], il_action_batch)

        # 计算RL损失组件 (只使用pi中对应expert和exploration的部分)
        # policy_q_state_batch的大小是 (batch_size + batch_size)
        rl_slice_start = int(batch_size) # Dagger数据之后是Expert数据+exploration数据
        qf1_pi, qf2_pi = self.critic(policy_q_state_batch, pi[rl_slice_start:]) # policy_q_state是exp+explore
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        # 注意log_pi也需要切片
        rl_policy_loss_component = ((self.alpha * log_pi[rl_slice_start:]) - min_qf_pi).mean()

        # 计算辅助损失
        aux_loss = self.aux_loss(resnet_output, gru_output, policy_res_pos_batch, policy_res_att_batch, 
                                policy_gru_vel_batch, policy_gru_ang_batch)
        
        # 计算强化学习损失权重
        with torch.no_grad():
            current_td_error = qf_loss.item()
            disagreement = torch.abs(qf1_pi - qf2_pi).mean().item()
            
            # 步骤A: 收集当前窗口的数据
            self._window_td_errors.append(current_td_error)
            self._window_disagreements.append(disagreement)

            # 到达窗口末尾时评估和更新基线
            if updates % self.baseline_update_window == 0:
                # 计算当前窗口的候选基线
                candidate_td = np.mean(self._window_td_errors) + 1e-8
                candidate_dis = np.mean(self._window_disagreements) + 1e-8

                if not self._is_initial_baseline_set:
                    # 特殊情况：第一次设置基线，无条件接受
                    self.baseline_td = candidate_td
                    self.baseline_dis = candidate_dis
                    self.initial_td = candidate_td # 设置最初td_error值，避免训练末期rl权重反而过小
                    self.initial_dis = candidate_dis
                    self._is_initial_baseline_set = True
                    self.target_baseline_td = candidate_td # 设置目标值，实现缓慢下降而非阶跃变化
                    self.target_baseline_dis = candidate_dis
                    print("--- Initial baseline set ---")
                else:
                    # 如果候选值大于旧基线，或者远小于旧基线，则更新；在这里剪裁，避免候选值过小
                    if candidate_td > self.baseline_td or candidate_td < self.baseline_update_gamma * self.baseline_td:
                        self.target_baseline_td = max(candidate_td, self.initial_td * self.k_rl_threshold)
                        self.delta_baseline_td = (self.baseline_td - self.target_baseline_td) / self.baseline_update_window
                    
                    if candidate_dis > self.baseline_dis or candidate_dis < self.baseline_update_gamma * self.baseline_dis:
                        self.target_baseline_dis = max(candidate_dis, self.initial_dis * self.k_rl_threshold)
                        self.delta_baseline_dis = (self.baseline_dis - self.target_baseline_dis) / self.baseline_update_window

                    print("--- Baseline re-evaluated ---")

                # 打印当前基线值以供监控
                print(f"  New target TD Baseline: {self.target_baseline_td:.4f} (Candidate: {candidate_td:.4f})")
                print(f"  New target Dis. Baseline: {self.target_baseline_dis:.4f} (Candidate: {candidate_dis:.4f})")

                # 清空窗口数据，为下一个周期做准备
                self._window_td_errors = []
                self._window_disagreements = []
        
            self.baseline_dis = self.baseline_dis - self.delta_baseline_dis
            self.baseline_td = self.baseline_td - self.delta_baseline_td
            
            # 使用当前基线进行归一化并计算权重
            if not self._is_initial_baseline_set:
                # 在第一个基线建立之前，倾向于模仿
                w_rl = torch.tensor(0.0, device=self.device)
            else:
                norm_td = min(current_td_error / self.baseline_td, 2.0) # 裁剪值可以适当放大
                norm_dis = min(disagreement / self.baseline_dis, 2.0)
                
                hybrid_metric = max(norm_td, norm_dis)
                w_rl = torch.exp(torch.tensor(-self.k_final * hybrid_metric, device=self.device))

            w_il = 1 - w_rl

        # 计算最终加权总损失
        total_policy_loss = w_rl * rl_policy_loss_component + w_il * il_policy_loss_component * self.dagger_weight + self.aux_loss_weight * aux_loss
        
        self.policy_optim.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optim.step()

        # Alpha和Target Network更新
        alpha_loss = torch.tensor(0.).to(device)
        alpha_tlogs = torch.tensor(self.alpha)
        if self.automatic_entropy_tuning:
            # 注意log_pi也需要用RL上下文的部分
            alpha_loss = -(self.alpha * (log_pi[rl_slice_start:].detach() + self.target_entropy)).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            alpha_tlogs = self.alpha.clone()

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        print(f"RL weight:{w_rl}, RL:{rl_policy_loss_component}, IL:{il_policy_loss_component}")
        return total_policy_loss.item(), rl_policy_loss_component.item(), il_policy_loss_component.item(), alpha_loss.item(), alpha_tlogs.item()

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