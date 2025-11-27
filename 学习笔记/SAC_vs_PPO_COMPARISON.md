# SAC vs PPO 对比

| 维度 | PPO | SAC |
|------|-----|-----|
| **策略类型** | On-Policy（在线策略） | Off-Policy（离线策略） |
| **数据使用** | 只能用最新数据（K epochs后丢弃） | 可重复使用历史数据 |
| **样本效率** | 中等 | 高|
| **训练稳定性** | 非常稳定 | 较稳定 |
| **收敛速度** | 中等 | 较快 |
| **内存占用** | 小 | 大，需要Replay Buffer |
| **超参数敏感度** | 低，易调 | 中等 |

## 代码层面对比

### 1. Buffer管理

#### PPO (Recurrent)
```python
# 短期Rollout Buffer（On-Policy, 支持序列采样）
rollout_buffer = RolloutBuffer(PPO_dict['rollout_buffer'])
# 长期Expert Buffer（Off-Policy, 用于IL）
expert_buffer = ReplayMemory(PPO_dict['buffer_param'])

# Episode中收集数据 (包含hidden_state)
rollout_buffer.push((img_seq, V_state, pi_state, action, reward, done, 
                     log_prob, value, pos, rot, vel, ang, hidden_state))

# Episode结束后
rollout_buffer.finish_path(last_value, done)  # 计算GAE
agent.update()  # 更新K epochs
rollout_buffer.reset()  # ✅ 清空（不再使用）
```

#### SAC
```python
# 长期Replay Memory（Off-Policy）
replay_memory = ReplayMemory(capacity=500000)

# 每步收集数据
replay_memory.push(source_name, (state, action, reward, next_state, done, ...))

# 每步或每N步
agent.update()  # 从memory随机采样
# ✅ 数据保留，可重复使用
```

---

### 2. 训练更新时机

#### PPO
```python
# Episode结束后才更新
if done:
    # 1. 计算returns和advantages
    agent.rollout_buffer.finish_path(last_value, done)
    
    # 2. 多轮更新（K=10 epochs）
    for epoch in range(10):
        for batch in data_loader:
            agent.update()
    
    # 3. 清空buffer
    agent.rollout_buffer.reset()
```

#### SAC
```python
# 每步都可以更新
if total_steps % update_interval == 0:
    for _ in range(updates_per_step):
        # 从Replay Memory随机采样
        batch = replay_memory.sample(batch_size)
        agent.update(batch)
```

---

### 3. Loss函数

#### PPO (Recurrent + IL)
```python
# Clipped Surrogate Loss
ratio = π_new / π_old
loss_clipped = -min(
    ratio * advantage,
    clip(ratio, 1-ε, 1+ε) * advantage
)

# Value Loss (Clipped)
loss_value = max(
    (V_new - return)²,
    (V_old + clip(V_new - V_old, -ε, ε) - return)²
)

# Auxiliary Loss (多任务学习)
loss_aux = mse(pred_pos, gt_pos) + mse(pred_rot, gt_rot) + ...

# Imitation Learning Loss (可选)
loss_il = mse(action, expert_action)

# Total Loss
loss = loss_clipped + c1*loss_value - c2*entropy + w_aux*loss_aux + w_il*loss_il
```

#### SAC
```python
# Policy Loss（最大化Q值和熵）
loss_policy = -mean(Q(s, π(s)) - α*log_prob(π(s)|s))

# Q Loss（TD error）
loss_Q = mean((Q(s,a) - (r + γ*V_target(s')))²)

# 可选：自动调节α
loss_alpha = -mean(α * (log_prob + target_entropy))
```

---

### 4. 探索机制

#### PPO
```python
# 通过熵正则化鼓励探索
policy_loss = policy_loss - entropy_coef * entropy

# entropy_coef通常固定（如0.01）
```

#### SAC
```python
# 通过最大化熵鼓励探索
policy_loss = -(Q_value - α * log_prob)

# α可自动调节
if automatic_entropy_tuning:
    alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach())
    self.alpha = self.log_alpha.exp()
```

---

## 使用场景推荐

### 选择 PPO 的场景

1. **需要稳定训练**
   - 无人机控制（如本项目）
   - 机器人运动控制
   - 连续控制任务

2. **计算资源有限**
   - 不需要大型Replay Buffer
   - 内存占用小

3. **实时在线学习**
   - 与环境实时交互
   - 不依赖历史数据

4. **初次尝试RL**
   - 超参数不敏感
   - 易于调试

**示例代码：**
```python
# main.py
args = {
    'rl_algorithm': 'PPO',
    'task': 'Train',
    # PPO特化参数
    'updates_interval': 1,      # 每步收集，episode结束更新
    'updates_per_episode': 1,   # 每个episode只更新一次
}
```

---

### 选择 SAC 的场景

1. **样本收集困难/昂贵**
   - 真实机器人实验
   - 仿真速度慢
   - 需要重复利用数据

2. **复杂高维任务**
   - 复杂状态空间
   - 需要强探索

3. **追求高样本效率**
   - 有限样本下快速收敛
   - Off-Policy优势

4. **有充足计算资源**
   - 可以维护大型Buffer
   - GPU/内存充足

**示例代码：**
```python
# main.py
args = {
    'rl_algorithm': 'SAC',
    'task': 'Train',
    # SAC特化参数
    'updates_interval': 1,      # 每步更新
    'updates_per_episode': 1,   # 每步更新一次
}
```

---

## 超参数对比

### PPO 关键超参数

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `lr` | 3e-4 | 学习率（PPO论文推荐） |
| `clip` | 0.2 | 裁剪范围（核心参数） |
| `ppo_epoch` | 10 | 更新轮数（K epochs） |
| `n_steps` | 2048 | Rollout长度 |
| `mini_batch_size` | 32 | **序列块数量** (非Step数) |
| `seq_len` | 8 | **序列块长度** (Recurrent PPO) |
| `gae_lambda` | 0.95 | GAE平滑系数 |
| `value_loss_coef` | 0.5 | Value Loss权重 |
| `entropy_coef` | 0.01 | 熵正则化 |
| `il_loss_weight` | 0.0 | 模仿学习权重 |

### SAC 关键超参数

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `lr` | 1e-4 | 学习率 |
| `alpha` | 0.2 | 温度系数（熵权重） |
| `tau` | 0.01 | 目标网络软更新 |
| `gamma` | 0.99 | 折扣因子 |
| `buffer_size` | 500k | Replay Buffer大小 |
| `batch_size` | 64 | Mini-batch大小 |
| `automatic_entropy_tuning` | False | 是否自动调节α |

---

## 训练曲线对比

### PPO 典型曲线
```
Reward ↑
  │     ╱─────
  │    ╱
  │   ╱
  │  ╱
  │ ╱
  └─────────→ Episodes
  
特点：
✓ 平滑上升
✓ 波动小
✓ 收敛稳定
✗ 初期慢
```

### SAC 典型曲线
```
Reward ↑
  │  ╱──╲─╱─
  │ ╱    ╲
  │╱      ╲
 ╱│        ╲
╱ │
└─────────→ Episodes

特点：
✓ 初期快
✓ 样本效率高
✗ 波动大
✗ 需要调参
```

---

## 切换方法

### Step 1: 修改算法选择
```python
# main.py
args = {
    'rl_algorithm': 'PPO',  # 改为 'PPO' 或 'SAC'
    # ... 其他参数 ...
}
```

### Step 2: 调整日志文件夹（可选）
```python
args = {
    'logs_folder': './runs/PPO/'  # 或 './runs/SAC/'
}
```

### Step 3: 运行训练
```bash
python main.py
```

### Step 4: 查看TensorBoard
```bash
tensorboard --logdir=./runs/PPO/
# 或
tensorboard --logdir=./runs/SAC/
```

---

## 理论对比

### PPO核心思想
> "限制策略更新幅度，防止破坏性更新"

- **Clipped Objective**: 裁剪ratio在[1-ε, 1+ε]
- **Trust Region**: 隐式信赖域方法
- **稳定性**: 通过裁剪保证单调改进

### SAC核心思想
> "最大化累积奖励和熵，平衡利用与探索"

- **Maximum Entropy RL**: max E[Σ(r + α*H(π))]
- **Off-Policy**: 充分利用历史数据
- **自动调节**: 动态调整探索-利用平衡
