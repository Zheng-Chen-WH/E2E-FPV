# 从检查点恢复训练指南 (Checkpoint Training Guide)

## 概述 (Overview)

`pretrain_aux.py` 现在支持从检查点（checkpoint）恢复训练。当训练中断或你想继续之前的训练时，这个功能非常有用。

## 快速开始 (Quick Start)

### 1. 正常开始训练（从头训练）

```python
# 在 pretrain_aux.py 顶部设置
RESUME_FROM_CHECKPOINT = False  # 不从检查点恢复
CHECKPOINT_PATH = None

# 运行
python pretrain_aux.py
```

### 2. 从检查点恢复训练

```python
# 在 pretrain_aux.py 顶部设置
RESUME_FROM_CHECKPOINT = True  # 启用检查点恢复
CHECKPOINT_PATH = "pretrained_models/aux_pretrain_expert_20250108_123456/checkpoint_ep100.pt"

# 运行
python pretrain_aux.py
```

## 详细说明 (Detailed Instructions)

### 检查点包含什么？

每个检查点保存以下信息：
- ✅ **模型权重** (`model_state_dict`) - 神经网络的所有参数
- ✅ **优化器状态** (`optimizer_state_dict`) - Adam/AdamW 的动量等状态
- ✅ **训练进度** (`episode`) - 当前训练到第几个 episode
- ✅ **最佳损失** (`best_loss`) - 目前为止的最低损失值
- ✅ **全局步数** (`global_step`) - TensorBoard 记录用的全局计数
- ✅ **统计信息** (`total_successes`, `total_collisions`) - 累计成功/碰撞次数
- ⚠️ **注意**: 数据缓冲区（buffer）**不会**保存，恢复后会从空 buffer 开始

### 检查点文件在哪？

训练时会自动保存两种检查点：

1. **周期性检查点** (每 50 个 episode 保存一次):
   ```
   pretrained_models/aux_pretrain_expert_20250108_123456/
       ├── checkpoint_ep50.pt
       ├── checkpoint_ep100.pt
       ├── checkpoint_ep150.pt
       └── ...
   ```

2. **最佳模型** (当损失降低时自动更新):
   ```
   pretrained_models/aux_pretrain_expert_20250108_123456/
       └── best_aux_model.pt
   ```

3. **最终模型** (训练完成后):
   ```
   pretrained_models/aux_pretrain_expert_20250108_123456/
       └── final_aux_model.pt
   ```

### 如何选择检查点？

| 场景 | 使用哪个检查点 |
|------|---------------|
| **训练意外中断** | 使用最新的 `checkpoint_epXXX.pt` |
| **想从最佳状态继续** | 使用 `best_aux_model.pt` |
| **训练完成，想继续训练更多 episodes** | 使用 `final_aux_model.pt` |

### 恢复训练时会发生什么？

```python
# 第1步: 加载检查点
Loading checkpoint from: pretrained_models/.../checkpoint_ep100.pt
  Resuming from episode 101        # 从第 101 个 episode 继续
  Best loss so far: 0.012345       # 已知的最佳损失
  Global step: 500                 # TensorBoard 全局步数
  Total successes: 45              # 累计成功次数
  Total collisions: 55             # 累计碰撞次数
  Note: Starting with empty buffer # Buffer 重新开始（会收集新数据）

# 第2步: 继续训练
Episode 101/2000: Collecting data...
Episode 102/2000: Collecting data...
...
```

## 配置参数说明 (Configuration)

在 `pretrain_aux.py` 顶部修改这些参数：

```python
# ====== 从检查点恢复相关设置 ======
RESUME_FROM_CHECKPOINT = False  # True = 从检查点恢复, False = 从头开始
CHECKPOINT_PATH = None          # 检查点文件的路径（如果 RESUME_FROM_CHECKPOINT=True）

# ====== 训练相关设置 ======
PRETRAIN_EPISODES = 2000        # 总共训练多少个 episode
PRETRAIN_SAVE_INTERVAL = 50     # 每多少个 episode 保存一次检查点
PRETRAIN_BATCH_SIZE = 32
PRETRAIN_LEARNING_RATE = 1e-3
PRETRAIN_EPOCHS_PER_EPISODE = 5

# ====== 数据收集模式 ======
DATA_COLLECTION_MODE = 'expert'  # 'expert', 'random', 或 'policy'
```

## 使用示例 (Examples)

### 示例 1: 训练中断后恢复

```bash
# 场景：你在训练第 120 个 episode 时 Ctrl+C 中断了

# 1. 找到最新的检查点
ls pretrained_models/aux_pretrain_expert_20250108_123456/
# 输出: checkpoint_ep100.pt, checkpoint_ep50.pt, ...

# 2. 修改 pretrain_aux.py
RESUME_FROM_CHECKPOINT = True
CHECKPOINT_PATH = "pretrained_models/aux_pretrain_expert_20250108_123456/checkpoint_ep100.pt"

# 3. 重新运行
python pretrain_aux.py

# ✅ 训练会从第 101 个 episode 开始继续
```

### 示例 2: 延长训练时间

```bash
# 场景：训练 500 episodes 完成了，但你想继续训练到 1000 episodes

# 1. 修改 pretrain_aux.py
PRETRAIN_EPISODES = 1000  # 增加总 episode 数
RESUME_FROM_CHECKPOINT = True
CHECKPOINT_PATH = "pretrained_models/aux_pretrain_expert_20250108_123456/final_aux_model.pt"

# 2. 运行
python pretrain_aux.py

# ✅ 训练会从第 501 个 episode 继续到 1000
```

### 示例 3: 从最佳模型继续微调

```bash
# 场景：训练了 200 episodes，但最佳损失在第 150 episode，你想从那里继续

# 1. 修改 pretrain_aux.py
RESUME_FROM_CHECKPOINT = True
CHECKPOINT_PATH = "pretrained_models/aux_pretrain_expert_20250108_123456/best_aux_model.pt"
PRETRAIN_LEARNING_RATE = 5e-4  # 可选：降低学习率进行微调

# 2. 运行
python pretrain_aux.py
```

## 常见问题 (FAQ)

### Q1: Buffer 为什么不保存？
**A:** Buffer 数据量太大（10000+ 样本），保存会让检查点文件非常大。重新收集数据反而能增加数据多样性。

### Q2: 可以切换数据收集模式吗？
**A:** 可以！比如先用 `'expert'` 训练 100 episodes，然后恢复时改成 `'random'` 继续训练。

### Q3: TensorBoard 日志会保留吗？
**A:** 恢复训练会创建**新的** TensorBoard 日志文件夹（带新时间戳）。要查看完整历史，需要在 TensorBoard 中加载多个日志文件夹。

### Q4: 如何确认检查点有效？
**A:** 加载检查点后，脚本会打印所有恢复的信息。检查 `Resuming from episode XXX` 和 `Best loss so far` 是否符合预期。

### Q5: 可以在不同机器上恢复训练吗？
**A:** 可以，只需要：
1. 复制整个 `pretrained_models/aux_pretrain_XXX/` 文件夹
2. 在新机器上设置正确的 `CHECKPOINT_PATH`
3. 确保 PyTorch 版本兼容

## 高级用法 (Advanced)

### 修改优化器学习率

```python
# 恢复后调整学习率
if RESUME_FROM_CHECKPOINT and CHECKPOINT_PATH is not None:
    # ... 加载检查点 ...
    
    # 修改学习率（微调时很有用）
    for param_group in optimizer.param_groups:
        param_group['lr'] = 5e-4  # 新学习率
    print(f"  Learning rate adjusted to: {5e-4}")
```

### 手动查看检查点内容

```python
import torch

checkpoint = torch.load("checkpoint_ep100.pt", map_location='cpu')
print(f"Episode: {checkpoint['episode']}")
print(f"Best loss: {checkpoint['best_loss']}")
print(f"Global step: {checkpoint['global_step']}")
print(f"Buffer size was: {checkpoint['buffer_size']}")
```

## 最佳实践 (Best Practices)

1. ✅ **定期检查检查点** - 每训练一段时间，检查 `checkpoint_epXXX.pt` 文件是否正常生成
2. ✅ **备份最佳模型** - `best_aux_model.pt` 很重要，建议手动备份到其他位置
3. ✅ **监控 TensorBoard** - 恢复训练后，确保损失曲线连续，没有突然跳变
4. ✅ **记录训练日志** - 保存终端输出，方便追踪训练历史
5. ⚠️ **不要修改模型结构** - 恢复训练时，模型结构（层数、维度）必须与保存时完全一致

## 故障排除 (Troubleshooting)

### 错误: "KeyError: 'best_loss'"
**原因**: 旧版检查点没有这个字段  
**解决**: 使用较新的检查点，或在代码中用 `checkpoint.get('best_loss', float('inf'))` 兼容

### 错误: "RuntimeError: size mismatch"
**原因**: 模型结构与检查点不匹配  
**解决**: 确保 `GRU_LAYER`, `RESNET_AUX_DIM` 等参数与保存时一致

### 恢复后损失突然升高
**原因**: Buffer 是空的，前几个 episode 的数据可能质量不佳  
**解决**: 这是正常的，等收集足够数据后损失会下降

---

**💡 提示**: 如果有任何问题，检查终端输出中的 "Resuming from episode XXX" 信息，确认恢复状态正确。
