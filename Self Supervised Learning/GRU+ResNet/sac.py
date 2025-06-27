import os
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork

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

        self.device = torch.device("cuda" if args['cuda'] else "cpu")

        self.critic = QNetwork(args['Q_network_dim'], args['action_dim'], args['hidden_sizes'], args['activation']).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), self.base_lr)

        self.critic_target = QNetwork(args['Q_network_dim'], args['action_dim'], args['hidden_sizes'], args['activation']).to(self.device)
        hard_update(self.critic_target, self.critic) #初始化的时候直接硬更新
        
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
        self.policy_optim = AdamW(self.policy.parameters(), self.base_lr, weight_decay = 0.01) # Gemini说transformer适合用

    def six_d_to_rot_mat(pred_6d):
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
        # resnet输出结果(B, T, 9)
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
            action, _, _, _, _, _, _ = self.policy.sample(img_sequence, state)
        else:
            _, _, action, _, _, _, _ = self.policy.sample(img_sequence, state) #如果evaluate为True，输出的动作是网络的mean经过squash的结果
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, rl_memory, dagger_memory, batch_size, updates):
        rl_pi_img_batch,rl_pi_state_batch, rl_q_state_batch, rl_action_batch, rl_reward_batch, rl_next_pi_img_batch, \
            rl_next_pi_state_batch, rl_next_q_state_batch, rl_done_batch, rl_goal_batch,\
                  rl_resnet_position_batch, rl_resnet_attitude_batch, rl_gru_velocity_batch, rl_gru_angular_batch  = rl_memory.sample(batch_size=batch_size)
        # print("pi_img_batch shape:", pi_img_batch.shape)
        # print("next_pi_img_batch shape:", next_pi_img_batch.shape)
        rl_pi_img_batch = torch.FloatTensor(rl_pi_img_batch).to(self.device)
        rl_pi_state_batch = torch.FloatTensor(rl_pi_state_batch).to(self.device)
        rl_q_state_batch = torch.FloatTensor(rl_q_state_batch).to(self.device)
        rl_action_batch = torch.FloatTensor(rl_action_batch).to(self.device)
        rl_reward_batch = torch.FloatTensor(rl_reward_batch).to(self.device).unsqueeze(1)
        rl_next_pi_img_batch = torch.FloatTensor(rl_next_pi_img_batch).to(self.device)
        rl_next_pi_state_batch = torch.FloatTensor(rl_next_pi_state_batch).to(self.device)
        rl_next_q_state_batch = torch.FloatTensor(rl_next_q_state_batch).to(self.device)
        rl_done_batch = torch.FloatTensor(rl_done_batch).to(self.device).unsqueeze(1)
        rl_goal_batch = torch.FloatTensor(rl_goal_batch).to(self.device)

        rl_resnet_position_batch = torch.FloatTensor(rl_resnet_position_batch).to(self.device)
        rl_resnet_attitude_batch = torch.FloatTensor(rl_resnet_attitude_batch).to(self.device) # 需要是(B, T, 3, 3)大小
        rl_gru_velocity_batch = torch.FloatTensor(rl_gru_velocity_batch).to(self.device)
        rl_gru_angular_batch = torch.FloatTensor(rl_gru_angular_batch).to(self.device)
        
        if updates < self.warm_up_steps:
            # 计算当前步的学习率：从0线性增长到 base_lr
            current_lr = (updates / self.warm_up_steps) * self.base_lr
            
            # 应用到 Critic 优化器
            for param_group in self.critic_optim.param_groups:
                param_group['lr'] = current_lr
            
            # 应用到 Policy 优化器
            for param_group in self.policy_optim.param_groups:
                param_group['lr'] = current_lr

        # critic更新
        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _, _ = self.policy.sample(rl_next_pi_img_batch,(torch.cat([rl_next_pi_state_batch, rl_goal_batch],dim=1))) #policy网络算出来的action
            qf1_next_target, qf2_next_target = self.critic_target(rl_next_q_state_batch, next_state_action) #target算出来的q值
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) #选择较小的Q值
            target_q_value = rl_reward_batch + (1 - rl_done_batch) * self.gamma * (min_qf_next_target - self.alpha * next_state_log_pi) #原论文(2),(3)式
            # 上式为bellman backup,备份一个状态 或是状态动作对，是贝尔曼方程的右边，即reward+next value
        qf1, qf2 = self.critic(rl_next_q_state_batch, next_state_action)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, target_q_value)  # MSEloss是对一个batch中所有样本的loss取差值平方后求平均
        qf2_loss = F.mse_loss(qf2, target_q_value)  # JQ = ��(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(��st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss
        self.critic_optim.zero_grad()
        qf_loss.backward() #这里的qf_loss保留了梯度信息而非简单相加，因此(loss1+loss2)整体对两个网络做梯度反向传播时，loss2对q1网络的梯度为0
        self.critic_optim.step()

        # policy更新
        dagger_pi_img_batch, dagger_pi_state_batch, dagger_action_batch, dagger_goal_batch,\
              dagger_resnet_position_batch, dagger_resnet_attitude_batch, dagger_gru_velocity_batch, dagger_gru_angular_batch = dagger_memory.sample(batch_size=batch_size)
        dagger_pi_img_batch = torch.FloatTensor(dagger_pi_img_batch).to(self.device)
        dagger_pi_state_batch = torch.FloatTensor(dagger_pi_state_batch).to(self.device)
        dagger_action_batch = torch.FloatTensor(dagger_action_batch).to(self.device)
        dagger_goal_batch = torch.FloatTensor(dagger_goal_batch).to(self.device)
        dagger_resnet_position_batch = torch.FloatTensor(dagger_resnet_position_batch).to(self.device)
        dagger_resnet_attitude_batch = torch.FloatTensor(dagger_resnet_attitude_batch).to(self.device) # 需要是(B, T, 3, 3)大小
        dagger_gru_velocity_batch = torch.FloatTensor(dagger_gru_velocity_batch).to(self.device)
        dagger_gru_angular_batch = torch.FloatTensor(dagger_gru_angular_batch).to(self.device)
        
        rl_pi, rl_log_pi, _, rl_resnet_output, rl_gru_output = self.policy.sample(rl_pi_img_batch, torch.cat([rl_pi_state_batch, rl_goal_batch],dim=1))
        dagger_pi, _, _, dagger_resnet_output, dagger_gru_output = self.policy.sample(dagger_pi_img_batch, torch.cat([dagger_pi_state_batch, dagger_goal_batch],dim=1))
        # qf1_pi, qf2_pi = self.critic(torch.cat([q_state_batch, goal_batch],dim=1), pi)
        # min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = ��st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))] 原论文式(7)
        
        # 主动作损失
        rl_policy_loss = F.mse_loss(rl_pi, rl_action_batch)
        dagger_policy_loss = F.mse_loss(dagger_pi, dagger_action_batch)

        # 辅助头损失
        rl_aux_loss = self.aux_loss(
        rl_resnet_output, rl_gru_output, rl_resnet_position_batch, rl_resnet_attitude_batch, 
        rl_gru_velocity_batch, rl_gru_angular_batch)

        dagger_aux_loss = self.aux_loss(
        dagger_resnet_output, dagger_gru_output, dagger_resnet_position_batch, dagger_resnet_attitude_batch, 
        dagger_gru_velocity_batch, dagger_gru_angular_batch)

        # 总损失
        total_policy_loss = self.aux_loss_weight * (rl_aux_loss + dagger_aux_loss) + (rl_policy_loss + dagger_policy_loss)
        
        # 加权处理损失
        self.policy_optim.zero_grad()
        total_policy_loss.backward()
        # for name, param in self.policy.named_parameters():
        #     if param.grad is not None:
        #         # 打印梯度范数，可以看到梯度大小
        #         print(f"{name}: Grad Norm = {param.grad.norm().item()}")
        self.policy_optim.step()

        # 预训练阶段固定alpha，这个公式不适用模仿学习
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.alpha * (rl_log_pi + self.target_entropy).detach()).mean() #原论文里的J函数就是loss，不需要再在代码里给出∇形式

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau) #对目标网络软更新

        return qf1_loss.item(), qf2_loss.item(), total_policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

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
            checkpoint = torch.load(file_name + "_model.pt")
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