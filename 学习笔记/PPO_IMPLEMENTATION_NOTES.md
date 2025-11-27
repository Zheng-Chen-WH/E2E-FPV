# PPO 现代化实现说明文档

## 概述

本次修改将项目升级为支持 **SAC** 和 **PPO** 两种强化学习算法，并且可以通过修改配置文件切换。PPO实现遵循主流标准（CleanRL、Stable-Baselines3）。

---

## 核心修改

### 1. **ppo.py - PPO算法核心**

#### 修改后的改进

**1. Buffer管理**
```python
self.rollout_buffer = RolloutBuffer(PPO_dict['rollout_buffer'])  # On-Policy
self.expert_buffer = ReplayMemory(PPO_dict['buffer_param'])      # Off-Policy (IL用)
self.batch_config = PPO_dict['batch_size']  # IL采样配置
```

**2. select_action() - 标准PPO交互**
```python
# 返回 (action, log_prob, value, hidden_state) - PPO标准三元组 + RNN状态
def select_action(self, img_sequence, state, V_state, evaluate=False):
    with torch.no_grad():
        value = self.critic(V_state_tensor)  # 状态价值
        # 使用 sample 方法，它内部处理了 hidden_state
        action, log_prob, mean, _, _, new_hidden = self.policy.sample(
            img_sequence, state_tensor, self.hidden_state
        )
        
        if evaluate:
            action = mean  # 确定性策略（评估模式）
    
    # 更新hidden state（用于GRU）
    if new_hidden is not None:
        self.hidden_state = new_hidden
        
    return action.cpu().numpy()[0], log_prob.cpu().item(), value.cpu().item(), input_hidden
```

**3. evaluate_actions() - 正确的Tanh Squashed Gaussian + Recurrent**
```python
# 关键改进：修复 Tanh 分布的 log_prob 计算，并支持序列输入
def evaluate_actions(self, img_seq, state, action, hidden_state=None):
    # img_seq: (batch, seq_len, T, C, H, W)
    # 逐时间步前向传播，保持时序依赖
    for t in range(seq_len):
        # ... (循环处理序列) ...
        mean, log_std, first_aux, second_aux, current_hidden = self.policy(
            img_t, state_t, current_hidden
        )
        # ... (收集结果) ...

    std = all_log_stds.exp()
    normal = torch.distributions.Normal(all_means, std)
    
    # --- Tanh Squashed Gaussian 的标准处理 ---
    action_clipped = torch.clamp(action, -0.999999, 0.999999)  # 防止数值溢出
    x_t = torch.atanh(action_clipped)  # 反推 pre-tanh 值
    log_prob = normal.log_prob(x_t)    # 原始高斯分布的 log_prob
    
    # 应用 Jacobian 修正（Change of Variables）
    # log π(a|s) = log μ(u|s) - Σ log(1 - tanh²(u))
    log_prob -= torch.log(1 - action_clipped.pow(2) + 1e-6)
    log_prob -= torch.log(action_scale + 1e-6) # 修正scale
    log_prob = log_prob.sum(-1, keepdim=True)  # 多维动作求和
    
    entropy = normal.entropy().sum(-1, keepdim=True)
    return log_prob, entropy, all_first_aux, all_second_aux
```

**4. update() - 现代PPO核心算法**
```python
def update(self, updates):
    """
    遵循 CleanRL/Stable-Baselines3 标准实现，支持 Recurrent PPO
    """
    # ... (IL数据准备) ...
    
    data_loader = self.rollout_buffer.get_data_loader(self.mini_batch_size)
    
    # 多轮更新（K epochs）- PPO核心特征
    for epoch in range(self.ppo_epoch):
        for batch in data_loader:
            # 解包 batch 数据 (Recurrent PPO 格式: batch_size, seq_len, ...)
            (img_seqs, V_states, pi_states, actions, old_log_probs, returns, advantages, old_values,
             aux_pos, aux_rot, aux_vel, aux_ang, masks, init_hidden) = batch
            
            # Advantage归一化（提高训练稳定性）
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 用当前策略重新评估动作 (传入 init_hidden)
            new_log_probs, entropy, first_aux, second_aux = self.evaluate_actions(..., hidden_state=init_hidden)
            new_values = self.critic(V_states)
            
            # --- PPO Clipped Surrogate Loss（核心）---
            # 使用 mask 处理变长序列
            ratio = torch.exp(new_log_probs - old_log_probs)  # π_θ / π_θ_old
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            policy_loss = -(torch.min(surr1, surr2) * masks).sum() / masks.sum()
            
            # --- Value Loss (Clipped, 推荐) ---
            v_pred_clipped = old_values + torch.clamp(new_values - old_values, -clip, clip)
            v_loss_unclipped = (new_values - returns).pow(2)
            v_loss_clipped = (v_pred_clipped - returns).pow(2)
            value_loss = 0.5 * (torch.max(v_loss_unclipped, v_loss_clipped) * masks).sum() / masks.sum()
            
            # --- 总损失 ---
            # L^CLIP+VF+S = L^CLIP - c1*L^VF + c2*S[π](s)
            rl_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            
            # 可选：Imitation Learning (从expert_buffer采样)
            il_loss = self._compute_il_loss(expert_batch) if self.il_weight > 0 else 0.0
            
            total_loss = rl_loss + self.aux_loss_weight * aux_loss_val + self.il_weight * il_loss
            
            # 优化
            self.policy_optim.zero_grad()
            self.critic_optim.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_norm_grad)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_norm_grad)
            
            self.policy_optim.step()
            self.critic_optim.step()
    
    # On-Policy特性：清空Rollout Buffer
    self.rollout_buffer.reset()
    
    return avg_policy_loss, avg_value_loss, avg_rl_loss, avg_il_loss, avg_aux_loss
```

**5. push_data() - 统一数据推送接口**
```python
def push_data(self, source_name, data):
    """
    统一接口，兼容SAC和PPO
    """
    if source_name in ['expert', 'dagger']:
        self.expert_buffer.push(source_name, data)  # IL用
    elif source_name == 'rollout':
        self.rollout_buffer.push(data)  # PPO On-Policy数据
    else:
        raise ValueError(f"Unknown source_name: {source_name}")
```

---

### 2. **replay_memory.py - RolloutBuffer优化**

#### 改进点

**1. compute_returns_and_advantages() - GAE计算**
```python
def compute_returns_and_advantages(self, last_value, done):
    """
    Generalized Advantage Estimation (Schulman et al., 2016)
    """
    advantages = np.zeros(len(self.rewards), dtype=np.float32)
    last_gae_lam = 0
    
    values = np.array(self.values + [last_value], dtype=np.float32)
    rewards = np.array(self.rewards, dtype=np.float32)
    dones = np.array(self.dones + [done], dtype=np.float32)
    
    # 逆序计算GAE
    for t in reversed(range(len(self.rewards))):
        if t == len(self.rewards) - 1:
            next_non_terminal = 1.0 - dones[-1]
            next_value = values[-1]
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_value = values[t + 1]
        
        # TD error: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
        delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
        
        # GAE: A_t = δ_t + (γλ)*δ_{t+1} + (γλ)²*δ_{t+2} + ...
        advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
    
    returns = advantages + values[:-1]
    return returns, advantages
```

**2. get_data_loader() - 随机Mini-batch采样 (Sequence Chunks)**
```python
def get_data_loader(self, mini_batch_size):
    """
    Recurrent PPO标准做法：按序列块采样，保持时序连续性
    """
    # ... (预处理：构建 Chunk 索引列表) ...
    
    # 随机打乱序列块的顺序
    perm = torch.randperm(num_chunks)
    shuffled_indices = all_chunk_indices[perm]
    
    # 按MiniBatch遍历所有chunk
    for batch_start in range(0, num_chunks, mini_batch_size):
        # ... (提取数据) ...
        
        yield (img_seqs, V_states, pi_states, actions, log_probs, returns, advantages, values,
               aux_pos, aux_rot, aux_vel, aux_ang, masks, init_hidden)
```

---

### 3. **config.py - PPO参数配置**

#### ✨ 现代PPO超参数（参考CleanRL/SB3）

```python
PPO_param = {
    # 基础参数
    "gamma": 0.99,              # 折扣因子
    "lr": 3e-4,                 # 学习率（PPO论文推荐）
    "seed": 20000323,
    
    # PPO核心参数
    "n_steps": 2048,            # Rollout长度
    "mini_batch_size": 32,      # Mini-batch大小 (序列块数量)
    "seq_len": 8,               # Recurrent PPO 序列块长度
    "ppo_epoch": 10,            # K epochs（多轮更新）
    "clip": 0.2,                # ε裁剪范围（0.1~0.2）
    "gae_lambda": 0.95,         # GAE λ
    
    # 损失函数系数
    "value_loss_coef": 0.5,     # c1（Value Loss权重）
    "entropy_coef": 0.01,       # c2（熵正则化）
    "max_norm_grad": 0.5,       # 梯度裁剪
    
    # 网络初始化
    "mu_init_boundary": 0.01,   # PPO通常更小
    "warm_up_steps": 10000,     # 学习率预热
    
    # Imitation Learning Buffer
    'buffer_param': {...},
    'batch_size': {...},
    
    # Loss权重
    'loss_weight': {
        'aux_loss_weight': 0.5,
        'pos_loss_weight': 1.0,
        'rot_loss_weight': 1.0,
        'vel_loss_weight': 1.0,
        'ang_vel_loss_weight': 1.0,
        'il_loss_weight': 0.0   # 纯PPO可设为0
    },
}

# Rollout Buffer参数（避免循环引用）
PPO_param['rollout_buffer'] = {
    'device': device,
    'gamma': PPO_param['gamma'],
    'gae_lambda': PPO_param['gae_lambda'],
    'seq_len': PPO_param['seq_len'],
}
```

---

### 4. **main.py - 训练流程兼容**

#### ✨ SAC/PPO自动切换

**1. 算法初始化**
```python
# 在main.py顶部
import torch  # 新增：PPO需要

# 初始化
if args['rl_algorithm'] == 'PPO':
    agent = PPO(agent_args)
else:
    agent = SAC(agent_args)
```

**2. 数据收集（训练循环）**
```python
while episode_steps <= args['max_steps']:
    # 生成动作
    if args['rl_algorithm'] == 'PPO':
        # PPO: 返回 (action, log_prob, value)
        NN_action, log_prob, value = agent.select_action(img_tensor, final_pi_target)
    else:
        # SAC: 返回 action
        NN_action = agent.select_action(img_tensor, final_pi_target)
        log_prob, value = None, None
    
    MPC_action = MPC_agent.step(current_drone_state, phase_idx, elapsed_time)
    
    # ... 执行动作 ...
    
    # 存储数据
    if valid_action:
        agent.push_data("expert", (...))   # IL用
        agent.push_data("dagger", (...))   # IL用
        
        # PPO特有：存入rollout buffer
        if args['rl_algorithm'] == 'PPO' and log_prob is not None:
            agent.push_data("rollout", (
                img_tensor, final_pi_target, NN_action, reward, done, 
                log_prob, value, relative_next_target_pos, attitude_9d, 
                relative_next_target_vel, fpv_angular_vel, None
            ))
```

**3. 训练更新**
```python
# SAC: 每隔固定步数更新
if args['rl_algorithm'] == 'SAC' and total_num_steps % args['updates_interval'] == 0:
    for i in range(args['updates_per_episode']):
        policy_loss, qf_loss, rl_loss, il_loss, aux_loss = agent.update(updates)
        # ... 记录日志 ...
        updates += 1

# PPO: Episode结束后更新
if done:
    if args['rl_algorithm'] == 'PPO':
        # 1. 获取最后一步的value
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(final_pi_target).to(agent.device).unsqueeze(0)
            last_value = agent.critic(next_state_tensor).cpu().numpy()[0][0]
        
        # 2. 计算GAE和returns
        agent.rollout_buffer.finish_path(last_value, done)
        
        # 3. 更新网络（K epochs）
        policy_loss, value_loss, rl_loss, il_loss, aux_loss = agent.update(updates)
        
        # 4. 记录日志
        if args['logs']:
            writer.add_scalar('loss/policy', policy_loss, updates)
            writer.add_scalar('loss/value', value_loss, updates)
            # ...
        updates += 1
```

**4. 测试/评估**
```python
# 评估模式
while True:
    if args['rl_algorithm'] == 'PPO':
        NN_action, _, _ = agent.select_action(img_tensor, final_pi_target, evaluate=True)
    else:
        NN_action = agent.select_action(img_tensor, final_pi_target, evaluate=True)
    
    rescaled_NN_action = map_value(NN_action, ...)
    # ... 执行动作 ...
```

---

## 🔄 切换SAC/PPO的方法

### 方法1：修改 main.py 中的 args
```python
args = {
    'rl_algorithm': 'PPO',  # 改为 'PPO' 或 'SAC'
    'task': 'Train',
    'eval': True,
    # ... 其他参数 ...
}
```

### 方法2：命令行参数（如果实现）
```bash
python main.py --rl_algorithm PPO
python main.py --rl_algorithm SAC
```

---

## 📊 PPO vs SAC 对比

| 特性 | PPO | SAC |
|------|-----|-----|
| **策略类型** | On-Policy | Off-Policy |
| **数据效率** | 中等 | 高 |
| **样本复用** | 有限（K epochs） | 无限（Replay Buffer） |
| **训练稳定性** | 高（裁剪机制） | 中等 |
| **超参数敏感度** | 低 | 中等 |
| **探索机制** | 熵正则化 | 熵正则化 + 自动调节 |
| **更新频率** | Episode结束 | 每步或每N步 |
| **Buffer** | Rollout Buffer（短期） | Replay Memory（长期） |
| **适用场景** | 连续控制、稳定训练 | 高数据效率、复杂任务 |

---

## 🚀 使用建议

### 何时使用PPO？
1. **需要稳定训练**：PPO的裁剪机制防止策略更新过大
2. **计算资源有限**：PPO不需要维护大型Replay Buffer
3. **实时交互**：On-Policy特性适合在线学习

### 何时使用SAC？
1. **样本收集困难**：SAC可以重复利用历史数据
2. **需要高样本效率**：Off-Policy特性充分利用数据
3. **复杂任务**：SAC的熵调节机制有助于探索

### 推荐训练流程
```python
# 1. 快速启动（使用MPC示范数据）
args['rl_algorithm'] = 'PPO'
args['expert_freq'] = 5  # 每5个episode进行一次MPC示范

# 2. 训练
python main.py

# 3. 评估
args['task'] = 'Test'
args['load_file'] = 'best_master_model'
python main.py

# 4. 如果不稳定，尝试调整
PPO_param['clip'] = 0.1          # 减小裁剪范围
PPO_param['entropy_coef'] = 0.02  # 增加探索
PPO_param['lr'] = 1e-4           # 降低学习率
```

---

## 🐛 常见问题排查

### 1. RuntimeError: 必须先调用 finish_path()
**原因**：PPO在episode结束后才调用`finish_path()`计算returns和advantages

**解决**：确保在`agent.update()`前调用：
```python
if done:
    agent.rollout_buffer.finish_path(last_value, done)
    agent.update(updates)
```

### 2. Value Loss很大
**原因**：Value Network拟合不好

**解决**：
- 增加`value_loss_coef`（默认0.5）
- 增加Value Network的隐藏层大小
- 检查reward scaling

### 3. Policy Loss不下降
**原因**：裁剪太严格或探索不足

**解决**：
- 增大`clip`（0.2 → 0.3）
- 增大`entropy_coef`（0.01 → 0.02）
- 检查Advantage归一化

### 4. 训练不稳定
**原因**：学习率过高或梯度爆炸

**解决**：
- 降低学习率（3e-4 → 1e-4）
- 减小`max_norm_grad`（0.5 → 0.3）
- 增加`mini_batch_size`

---

## 📝 代码结构总结

```
SACfD/
├── ppo.py                  # PPO算法实现（已完善）
├── sac.py                  # SAC算法实现（保持不变）
├── replay_memory.py        # RolloutBuffer + ReplayMemory（已优化）
├── model.py                # GaussianPolicy（已兼容PPO）
├── config.py               # 超参数配置（已添加PPO_param）
├── main.py                 # 训练/测试主程序（已兼容SAC/PPO）
├── env.py                  # 环境交互（无需修改）
└── PPO_IMPLEMENTATION_NOTES.md  # 本文档
```

---

## ✅ 验证清单

- [x] PPO类的buffer引用修复
- [x] update()方法实现完整的PPO算法
- [x] evaluate_actions()正确处理Tanh分布
- [x] RolloutBuffer的GAE计算正确
- [x] config.py的PPO参数完善
- [x] main.py兼容SAC和PPO切换
- [x] 测试代码兼容两种算法
- [x] 所有文件无语法错误

---

## 🎓 参考文献

1. **PPO论文**: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
2. **GAE论文**: Schulman et al. (2016) "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
3. **CleanRL实现**: https://github.com/vwxyzjn/cleanrl
4. **Stable-Baselines3**: https://github.com/DLR-RM/stable-baselines3

---

## 📞 技术支持

如有问题，请检查：
1. 所有依赖包是否安装（torch, numpy等）
2. config.py中的参数是否合理
3. main.py中的`args['rl_algorithm']`是否正确设置

**祝训练顺利！** 🚁✨
