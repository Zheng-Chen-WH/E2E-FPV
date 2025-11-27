# PPO算法

## 1. 策略梯度（Policy Gradient）
所有策略梯度方法的基础思想都很直观：如果某个动作序列带来了好的结果，就调整策略网络，让它未来更有可能做出这些动作；反之，如果结果不好，就降低做出这些动作的可能性。

数学上，其目标函数的梯度可以简化为：
$$
\nabla J(\theta) \approx \mathbb{E}[ \nabla \log \pi_\theta(a|s) \cdot A(s, a) ]
$$

*   $\pi_\theta(a|s)$ 是带参数 $\theta$ 的策略网络，表示在状态 $s$ 下选择动作 $a$ 的概率。
*   $A(s, a)$ 是优势函数 (Advantage Function)，表示在状态 $s$ 下，选择动作 $a$ 比平均水平好多少。
*   $\log \pi_\theta(a|s)$ 是一种技巧，$\nabla \log(p)$ 等于 $\nabla p / p$，使得只需要计算做出过的动作的梯度。

这个公式的问题在于：更新策略 $\theta_{new} = \theta_{old} + \alpha \cdot \nabla J(\theta)$ 时，学习率 $\alpha$ 非常难调。

*   $\alpha$ 太小，训练会极其缓慢。
*   $\alpha$ 太大，一次糟糕的更新就可能让策略网络彻底“跑偏”，再也无法恢复。想象一下，策略可能突然开始选择一些极端的、从未探索过的动作，导致后续收集到的数据质量极差，形成恶性循环，最终训练崩溃。

## 2. PPO的前身：信任区域策略优化 (TRPO)
为了解决步长问题，TRPO (Trust Region Policy Optimization) 提出，不应该用一个固定的学习率 $\alpha$，而是应该在策略更新的“幅度”上做一个限制。

TRPO的目标是最大化策略表现，但增加了一个约束条件：新策略和旧策略的KL散度（一种衡量两个概率分布差异的指标）不能超过一个很小的阈值 $\delta$。

这相当于说：“你可以在一个可信任的‘区域’内（Trust Region）尽可能地优化策略，但不要跑出这个圈。”

TRPO的效果非常好，非常稳定。但它的致命缺点是计算太复杂，需要求解一个带约束的优化问题，计算量很大，难以应用在复杂的项目中。

## 3. PPO的诞生：简化版的TRPO
PPO的核心目标和TRPO一样：限制策略更新的幅度。但它用了一种更简单、更易于实现的方法来达到类似TRPO的效果。PPO有两种主要形式，这里代码中使用的是最常见的 PPO-Clip。

### PPO-Clip的核心思想
PPO-Clip没有使用复杂的KL散度约束，而是直接修改了目标函数。代码中的这几行：

```python
# ppo.py update 方法内
ratio = torch.exp(new_log_probs - old_log_probs)
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
policy_loss = -torch.min(surr1, surr2).mean()
```

这几行代码就是PPO-Clip的精髓：

#### 关键概念一：重要性采样比率 (The Ratio)
PPO 是 On-Policy 算法，但为了能在一个 Batch 的数据上多训练几轮（代码里 `ppo_epoch` 循环了多次），我们需要比较“当前的策略”和“收集数据时的策略”。

我们定义一个比率 $r_t(\theta)$：
$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

*   如果 $r > 1$，说明新策略比旧策略更倾向于做这个动作。
*   如果 $r < 1$，说明新策略降低了这个动作的概率。

代码对应 (`ClassicPPO.update` 方法中):

```python
# 你的代码使用了对数概率相减，数学上等同于概率相除：exp(log(a) - log(b)) = a / b
ratio = torch.exp(new_log_probs - old_log_probs)
```
*   `new_log_probs`: 当前正在更新的网络计算出的概率。
*   `old_log_probs`: 之前收集数据时（Buffer里存的）的概率。

#### 关键概念二： 裁剪 (Clipping)
这是 PPO 理论中最著名公式的来源。我们希望最大化这个目标函数：

$$
L^{CLIP} = \mathbb{E} \left[ \min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t) \right]
$$

这里的 $A_t$ 是 **优势函数 (Advantage)**，表示“这个动作比平均水平好多少”。

这个公式的物理含义（非常重要）：

**当动作是好的 ($A_t > 0$)：**

*   我们希望增加这个动作的概率（让 `ratio` 变大）。
*   但是，为了防止步子迈太大，如果 `ratio` 超过了 $1 + \epsilon$（例如 1.2），我们就截断它，不再给予更多的奖励。这意味着：“你变好是可以的，但不要为了变好而让策略发生剧变。”

**当动作是坏的 ($A_t < 0$)：**

*   我们希望减少这个动作的概率（让 `ratio` 变小）。
*   同样，如果 `ratio` 低于 $1 - \epsilon$（例如 0.8），我们也截断。这意味着：“你修正错误是可以的，但不要矫枉过正。”

代码对应:

```python
# surr1 是未裁剪的原始目标：Ratio * Advantage
surr1 = ratio * advantages

# surr2 是裁剪后的目标：限制 Ratio 在 [1-clip, 1+clip] 之间
surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages

# 取最小值 (min)，这是一种“悲观”策略，保证更新是保守且安全的
# 注意前面有个负号，因为 PyTorch 是做最小化优化，而强化学习是最大化奖励
policy_loss = -torch.min(surr1, surr2).mean()
```
这里的 `self.clip_param` 通常设为 0.1 或 0.2，对应公式里的 $\epsilon$。

**surr1 (无约束的目标)**
`surr1 = ratio * advantages`。这其实就是标准的重要性采样（Importance Sampling）下的策略梯度目标。如果直接用这个来更新，就又回到了传统PG方法不稳定的老路上。

**surr2 (带裁剪的目标)**
`surr2 = torch.clamp(ratio, 1 - ε, 1 + ε) * advantages`。这是PPO的“神来之笔”。`self.clip_param` 就是超参数 $\epsilon$ (epsilon)，通常取0.1或0.2。

`torch.clamp(ratio, 1-ε, 1+ε)` 的意思是，把 `ratio` 强行限制在 $[1-\epsilon, 1+\epsilon]$ 这个区间内。

**policy_loss = -torch.min(surr1, surr2).mean()**
通过取 `surr1` 和 `surr2` 的最小值（然后取负号以进行梯度上升），PPO实现了一种悲观的、保守的更新。我们分两种情况讨论：

**情况一：advantages > 0 (这是一个好动作)**

*   我们希望增大 `ratio` 来鼓励这个动作。
*   但是，`surr2` 中的 `clamp` 操作会把 `ratio` 的上限卡在 $1+\epsilon$。
*   因此，当 `ratio` 增长到超过 $1+\epsilon$ 时，`surr1` 会继续变大，但 `surr2` 会被“摁住”在 $(1+\epsilon) \cdot advantages$。
*   `torch.min` 会选择 `surr2` 这个更小的值。这意味着，即使策略想变得非常激进，目标函数也不会提供超过 $1+\epsilon$ 的奖励，从而限制了更新的幅度。

**情况二：advantages < 0 (这是一个坏动作)**

*   我们希望减小 `ratio` 来抑制这个动作。
*   `surr2` 中的 `clamp` 操作会把 `ratio` 的下限卡在 $1-\epsilon$。
*   当 `ratio` 减小到低于 $1-\epsilon$ 时，`surr1` 会继续减小（更接近0），但 `surr2` 会被“摁住”在 $(1-\epsilon) \cdot advantages$。
*   `torch.min` 同样会选择 `surr2`（因为advantage是负数，所以 $(1-\epsilon) \cdot adv$ 比 $ratio \cdot adv$ 更大，即更不“负”）。这同样限制了惩罚的幅度，防止过度修正。

**总结PPO-Clip**：通过这个巧妙的 `min` 和 `clip` 操作，PPO构建了一个新的目标函数。在这个函数中，当策略变化过大时，目标函数会变得“扁平”，梯度会趋近于0。这相当于自动给策略更新踩了刹车，从而在不使用复杂约束的情况下，实现了“信任区域”的稳定效果。

#### 关键概念三：优势函数与 GAE (Generalized Advantage Estimation)
怎么判断一个动作好不好？我们不能只看当前的奖励，要看“长远的实际回报”减去“预期回报”。

$$
A(s, a) = Q(s, a) - V(s)
$$

*   $Q(s, a)$: 实际采取动作 $a$ 后获得的真实回报（现实）。
*   $V(s)$: 评论家（Critic）认为在这个状态下应该获得的回报（预期）。

PPO 使用 GAE 来平衡偏差 (Bias) 和 方差 (Variance)。
*   如果我们看整条轨迹（Monte Carlo），虽然无偏，但方差巨大（因为随机性累积） $A(s,a) = R - V(s)$（$R$是蒙特卡洛回报）虽然无偏，但方差巨大。
*   使用TD误差 $\delta = r + \gamma V(s') - V(s)$ 作为优势虽然方差小，但有偏。

GAE 通过 `gae_lambda` 参数在偏差和方差之间做了一个权衡，是目前计算优势函数的标准做法，能显著提升训练的稳定性和速度。

代码对应 (`RolloutBuffer.compute_returns_and_advantages`):

```python
# delta 就是一步的优势 (TD Error)
delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]

# GAE 公式：当前优势 + 折扣因子 * 下一步优势
advantages[t] = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
```

## 4. 结合代码的其他理论点

### On-Policy vs. Off-Policy 和 多Epoch更新

*   **On-Policy (在线策略)**：产生数据的策略和要优化的策略是同一个。传统PG方法是严格的On-Policy，采集一批数据，更新一次网络，然后丢掉这批数据。非常浪费！
*   **Off-Policy (离线策略)**：可以用旧策略产生的数据来优化当前策略。

PPO通过 `ratio` (重要性采样) 修正了新旧策略的差异，使其能够在一定程度上离线学习。这就是为什么您的代码中可以有一个 `for _ in range(self.ppo_epoch):` 循环。PPO可以用同一批采集来的数据，对网络进行多次更新，大大提高了数据利用率（Sample Efficiency），这是它相比A2C等算法的一大优势。


### Actor-Critic 架构与熵正则化
PPO 依然属于 Actor-Critic 框架：

*   **Actor (演员/策略)**：负责做动作 (`self.policy`)。
*   **Critic (评论家/价值)**：负责打分，它的打分越准，Advantage 计算就越准，Actor 学得就越快 (`self.critic`)。

此外，为了防止 Actor 在训练初期就“自以为是”地只选某一个动作（陷入局部最优），PPO 加入了 **熵 (Entropy)** 奖励。熵越大，代表分布越随机（探索性越强）。

代码对应:

```python
# 计算熵
entropy = normal.entropy().sum(1, keepdim=True)

# 总 Loss = 策略 Loss + 价值 Loss - 熵系数 * 熵
# 减去熵意味着：我们希望熵越大越好（Loss越小越好），鼓励探索
rl_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
```
---