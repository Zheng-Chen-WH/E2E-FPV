import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork
import numpy as np
import random

class SAC(object):
    def __init__(self, pi_num_inputs, Q_num_inputs, action_space, args):
        self.gamma = args['gamma']
        self.tau = args['tau']
        self.alpha = args['alpha']
        self.seed=args['seed']
        self.target_update_interval = args['target_update_interval']
        self.automatic_entropy_tuning = args['automatic_entropy_tuning']
        embedding_dim=args['embedding_dim']
        frames=args['num_frames']

        self.device = torch.device("cuda" if args['cuda'] else "cpu")

        self.critic = QNetwork(Q_num_inputs, action_space.shape[0], args['hidden_sizes'], args['activation']).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args['lr'])

        self.critic_target = QNetwork(Q_num_inputs, action_space.shape[0], args['hidden_sizes'], args['activation']).to(self.device)
        hard_update(self.critic_target, self.critic) #初始化的时候直接硬更新
        
        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning is True: #原论文直接认为目标熵就是动作空间维度乘积的负值，在这里就是Box的“体积”
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item() #torch.prod()是一个函数，用于计算张量中所有元素的乘积
            self.alpha = torch.zeros(1, requires_grad=True, device=self.device) #原论文没用log，但是这里用的，总之先改成无log状态试试
            #self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device) #初始化log_alpha
            self.alpha_optim = Adam([self.alpha], lr=args['lr'])

        self.policy = GaussianPolicy(embedding_dim, frames, pi_num_inputs, action_space.shape[0], args['hidden_sizes'], args['activation'], action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args['lr'])

    def select_action(self,img_sequence, state, evaluate=False):
        img_sequence=img_sequence.to(self.device)
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(img_sequence, state)
        else:
            _, _, action = self.policy.sample(img_sequence, state) #如果evaluate为True，输出的动作是网络的mean经过squash的结果
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        pi_img_batch, pi_state_batch, q_state_batch, action_batch, reward_batch, next_pi_img_batch, next_pi_state_batch, next_q_state_batch, done_batch, goal_batch = memory.sample(batch_size=batch_size)
        # print("pi_img_batch shape:", pi_img_batch.shape)
        # print("next_pi_img_batch shape:", next_pi_img_batch.shape)
        pi_img_batch = torch.FloatTensor(pi_img_batch).to(self.device)
        pi_state_batch = torch.FloatTensor(pi_state_batch).to(self.device)
        q_state_batch = torch.FloatTensor(q_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_pi_img_batch = torch.FloatTensor(next_pi_img_batch).to(self.device)
        next_pi_state_batch = torch.FloatTensor(next_pi_state_batch).to(self.device)
        next_q_state_batch = torch.FloatTensor(next_q_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)
        goal_batch = torch.FloatTensor(goal_batch).to(self.device)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_pi_img_batch,(torch.cat([next_pi_state_batch, goal_batch],dim=1))) #policy网络算出来的action
            qf1_next_target, qf2_next_target = self.critic_target(torch.cat([next_q_state_batch, goal_batch],dim=1), next_state_action) #target算出来的q值
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) #选择较小的Q值
            target_q_value = reward_batch + (1 - done_batch) * self.gamma * (min_qf_next_target - self.alpha * next_state_log_pi) #原论文(2),(3)式
            # 上式为bellman backup,备份一个状态 或是状态动作对，是贝尔曼方程的右边，即reward+next value
        qf1, qf2 = self.critic(torch.cat([q_state_batch,goal_batch],dim=1), action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, target_q_value)  # MSEloss是对一个batch中所有样本的loss取差值平方后求平均
        qf2_loss = F.mse_loss(qf2, target_q_value)  # JQ = ��(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(��st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss
        self.critic_optim.zero_grad()
        qf_loss.backward() #这里的qf_loss保留了梯度信息而非简单相加，因此(loss1+loss2)整体对两个网络做梯度反向传播时，loss2对q1网络的梯度为0
        self.critic_optim.step()
        pi, log_pi, _ = self.policy.sample(pi_img_batch, torch.cat([pi_state_batch, goal_batch],dim=1))
        qf1_pi, qf2_pi = self.critic(torch.cat([q_state_batch, goal_batch],dim=1), pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = ��st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))] 原论文式(7)

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.alpha * (log_pi + self.target_entropy).detach()).mean() #原论文里的J函数就是loss，不需要再在代码里给出∇形式

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau) #对目标网络软更新

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, ckpt_path=None):
        '''if not os.path.exists('GoodModel/'):
            os.makedirs('GoodModel/')'''
        if ckpt_path is None:
            ckpt_path = "sac_scene1_attack.pt".format()
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
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