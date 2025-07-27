def update_parameters(self, expert_memory, dagger_memory, exploration_memory, batch_size, updates):
    # --- 数据准备阶段 (FIX 2 & 3: 在CPU上准备好所有数据, 然后一次性转移) ---

    # 1. 从各个Buffer采样 (仍然是Numpy数组)
    exp_pi_img, exp_q_state, exp_action, exp_reward, exp_next_pi_img, \
        exp_next_q_state, exp_done, exp_goal, exp_res_pos, exp_res_att, \
        exp_gru_vel, exp_gru_ang = expert_memory.sample(batch_size=batch_size)

    expl_pi_img, expl_q_state, expl_action, expl_reward, expl_next_pi_img, \
        expl_next_q_state, expl_done, expl_goal, expl_res_pos, expl_res_att, \
        expl_gru_vel, expl_gru_ang = exploration_memory.sample(batch_size=batch_size)

    dag_pi_img, dag_action, dag_goal, dag_res_pos, dag_res_att, \
        dag_gru_vel, dag_gru_ang = dagger_memory.sample(batch_size=int(batch_size/2))

    # 2. 在CPU上构建用于Critic更新的RL批次 (expert + exploration)
    # FIX 3: 注意这里的reward和done不再需要unsqueeze
    rl_q_state_batch = torch.cat((torch.from_numpy(exp_q_state), torch.from_numpy(expl_q_state)), dim=0)
    rl_action_batch = torch.cat((torch.from_numpy(exp_action), torch.from_numpy(expl_action)), dim=0)
    rl_reward_batch = torch.cat((torch.from_numpy(exp_reward), torch.from_numpy(expl_reward)), dim=0)
    rl_next_q_state_batch = torch.cat((torch.from_numpy(exp_next_q_state), torch.from_numpy(expl_next_q_state)), dim=0)
    rl_done_batch = torch.cat((torch.from_numpy(exp_done), torch.from_numpy(expl_done)), dim=0)
    rl_next_pi_img_batch = torch.cat((torch.from_numpy(exp_next_pi_img), torch.from_numpy(expl_next_pi_img)), dim=0)
    rl_next_goal_batch = torch.cat((torch.from_numpy(exp_goal), torch.from_numpy(expl_goal)), dim=0)
    
    # 3. 在CPU上构建用于Policy更新的统一批次 (dagger + expert + exploration)
    # FIX 1: 这是统一数据流的核心
    # 注意：我们将把所有用于策略更新的状态和目标都拼接起来
    policy_pi_img_batch = torch.cat((torch.from_numpy(dag_pi_img), torch.from_numpy(exp_pi_img), torch.from_numpy(expl_pi_img)), dim=0)
    policy_goal_batch = torch.cat((torch.from_numpy(dag_goal), torch.from_numpy(exp_goal), torch.from_numpy(expl_goal)), dim=0)
    
    # 这个批次包含了策略网络需要评估的所有状态
    policy_q_state_batch = torch.cat((torch.from_numpy(exp_q_state), torch.from_numpy(expl_q_state)), dim=0) # Dagger数据没有Q状态

    # 用于计算IL Loss的“真值”动作 (dagger + expert)
    # 注意：探索数据没有对应的专家动作，我们可以用零向量填充，但更简单的做法是在计算loss时只使用相应部分
    il_action_batch = torch.cat((torch.from_numpy(dag_action), torch.from_numpy(exp_action)), dim=0)
    
    # 用于辅助损失的真值
    policy_res_pos_batch = torch.cat((torch.from_numpy(dag_res_pos), torch.from_numpy(exp_res_pos), torch.from_numpy(expl_res_pos)), dim=0)
    policy_res_att_batch = torch.cat((torch.from_numpy(dag_res_att), torch.from_numpy(exp_res_att), torch.from_numpy(expl_res_att)), dim=0)
    policy_gru_vel_batch = torch.cat((torch.from_numpy(dag_gru_vel), torch.from_numpy(exp_gru_vel), torch.from_numpy(expl_gru_vel)), dim=0)
    policy_gru_ang_batch = torch.cat((torch.from_numpy(dag_gru_ang), torch.from_numpy(exp_gru_ang), torch.from_numpy(expl_gru_ang)), dim=0)

    # 4. 一次性将所有数据转移到GPU
    device = self.device
    rl_q_state_batch, rl_action_batch, rl_reward_batch, rl_next_q_state_batch, rl_done_batch, \
    rl_next_pi_img_batch, rl_next_goal_batch = [t.float().to(device) for t in [rl_q_state_batch, rl_action_batch, rl_reward_batch, rl_next_q_state_batch, rl_done_batch, rl_next_pi_img_batch, rl_next_goal_batch]]
    
    policy_pi_img_batch, policy_goal_batch, policy_q_state_batch, il_action_batch, \
    policy_res_pos_batch, policy_res_att_batch, policy_gru_vel_batch, policy_gru_ang_batch = \
        [t.float().to(device) for t in [policy_pi_img_batch, policy_goal_batch, policy_q_state_batch, il_action_batch, policy_res_pos_batch, policy_res_att_batch, policy_gru_vel_batch, policy_gru_ang_batch]]

    # --- LR Warmup ---
    if updates < self.warm_up_steps:
        # ... (这部分不变)
        pass

    # --- Critic 网络更新 (逻辑不变, 使用准备好的rl_*批次) ---
    with torch.no_grad():
        next_state_action, next_state_log_pi, _, _, _ = self.policy.sample(rl_next_pi_img_batch, rl_next_goal_batch) 
        qf1_next_target, qf2_next_target = self.critic_target(rl_next_q_state_batch, next_state_action) 
        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) 
        target_q_value = rl_reward_batch + (1 - rl_done_batch) * self.gamma * (min_qf_next_target - self.alpha * next_state_log_pi) 
    qf1, qf2 = self.critic(rl_q_state_batch, rl_action_batch)  
    qf_loss = F.mse_loss(qf1, target_q_value) + F.mse_loss(qf2, target_q_value)
    
    self.critic_optim.zero_grad()
    qf_loss.backward() 
    self.critic_optim.step()

    # --- Policy 网络更新 (FIX 1: 使用统一的批次和逻辑) ---
    # 1. 对统一的策略批次进行一次前向传播
    pi, log_pi, _, resnet_output, gru_output, _ = self.policy.sample(policy_pi_img_batch, policy_goal_batch)
    
    # 2. 计算IL损失组件 (只使用pi中对应dagger和expert的部分)
    # il_action_batch的大小是 (batch_size/2 + batch_size)
    il_slice_size = il_action_batch.shape[0]
    il_policy_loss_component = physics_MSE(pi[:il_slice_size], il_action_batch)

    # 3. 计算RL损失组件 (只使用pi中对应expert和exploration的部分)
    # policy_q_state_batch的大小是 (batch_size + batch_size)
    rl_slice_start = int(batch_size/2) # Dagger数据之后是Expert数据
    qf1_pi, qf2_pi = self.critic(policy_q_state_batch, pi[rl_slice_start:])
    min_qf_pi = torch.min(qf1_pi, qf2_pi)
    # 注意log_pi也需要切片
    rl_policy_loss_component = ((self.alpha * log_pi[rl_slice_start:]) - min_qf_pi).mean()

    # 4. 计算辅助损失
    aux_loss = self.aux_loss(resnet_output, gru_output, policy_res_pos_batch, policy_res_att_batch, 
                             policy_gru_vel_batch, policy_gru_ang_batch)
    
    # 5. 计算"三重检查"权重
    with torch.no_grad():
        # 信号1: TD-Error
        current_td_error = qf_loss.item()
        if self.avg_td_error is None: self.avg_td_error = current_td_error
        else: self.avg_td_error = 0.99 * self.avg_td_error + 0.01 * current_td_error

        # 信号2: 分歧量 (来自RL上下文)
        disagreement = torch.abs(qf1_pi - qf2_pi).mean().item()
        if self.avg_disagreement is None: self.avg_disagreement = disagreement
        else: self.avg_disagreement = 0.99 * self.avg_disagreement + 0.01 * disagreement
            
        # 信号3: 策略一致性 (IL loss)
        il_deviation = il_policy_loss_component.item()
        if self.avg_il_loss is None: self.avg_il_loss = il_deviation
        else: self.avg_il_loss = 0.99 * self.avg_il_loss + 0.01 * il_deviation
        
        # 组合指标
        scaled_td_error = self.k_td * self.avg_td_error 
        scaled_disagreement = self.k_disagree * self.avg_disagreement
        scaled_il_deviation = self.k_il * self.avg_il_loss
        hybrid_metric = max(scaled_td_error, scaled_disagreement, scaled_il_deviation) 
        
        w_rl = torch.exp(-self.k_final * hybrid_metric)
        w_il = 1 - w_rl

    # 6. 计算最终加权总损失
    total_policy_loss = w_rl * rl_policy_loss_component + w_il * il_policy_loss_component + self.aux_loss_weight * aux_loss
    
    self.policy_optim.zero_grad()
    total_policy_loss.backward()
    self.policy_optim.step()

    # --- Alpha (熵) 和 Target Network 更新 (不变) ---
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

    return total_policy_loss.item(), rl_policy_loss_component.item(), il_policy_loss_component.item(), alpha_loss.item(), alpha_tlogs.item()

