# 使用预训练的ResNet+GRU视觉编码器

## 概述

现在 `main.py` 支持加载预训练的ResNet+GRU权重，这可以显著加速训练收敛并提升性能。

## 快速开始

### 1. 配置选项

在 `main.py` 的 `args` 字典中有两个新的配置项：

```python
args = {
    ...
    'USE_PRETRAINED_VISION': True,  # 是否使用预训练的ResNet+GRU模型
    'PRETRAINED_CHECKPOINT_PATH': 'pretrained_models/aux_pretrain_expert_20251108_165904/checkpoint_ep3300.pt',  # 预训练模型路径
    ...
}
```

### 2. 使用方法

#### 方法A：使用预训练模型（推荐）

1. **确保已经运行了预训练**：
   ```bash
   python pretrain_aux.py
   ```

2. **设置正确的检查点路径**：
   - 在 `main.py` 中将 `USE_PRETRAINED_VISION` 设置为 `True`
   - 将 `PRETRAINED_CHECKPOINT_PATH` 设置为你的预训练检查点路径
   - 例如：`'pretrained_models/aux_pretrain_expert_20251108_165904/best_aux_model.pt'`

3. **运行训练**：
   ```bash
   python main.py
   ```

#### 方法B：从头开始训练（不使用预训练）

直接在 `main.py` 中设置：
```python
'USE_PRETRAINED_VISION': False,
```

## 工作原理

### 启动时的日志输出

**使用预训练模型时：**
```
============================================================
LOADING PRETRAINED VISION ENCODER
============================================================
Loading from: pretrained_models/aux_pretrain_expert_20251108_165904/checkpoint_ep3300.pt
✓ Successfully loaded pretrained ResNet+GRU weights!
  - ResNet auxiliary head: position (3D) + rotation (6D)
  - GRU auxiliary head: velocity (3D) + angular velocity (3D)
============================================================
```

**不使用预训练时：**
```
Skipping pretrained vision encoder (USE_PRETRAINED_VISION=False)
```

**文件未找到时：**
```
✗ Pretrained checkpoint not found. Starting with random weights.
  Expected path: pretrained_models/xxx/checkpoint_epXXX.pt
  Set USE_PRETRAINED_VISION=False or run pretrain_aux.py first.
============================================================
```

## 加载的内容

预训练模型包含：
- **ResNet特征提取器**：卷积层权重和批归一化参数
- **ResNet辅助头**：预测相对位置(3D) + 旋转矩阵(6D→9D)
- **GRU时序模块**：隐藏层权重和门控参数
- **GRU辅助头**：预测相对速度(3D) + 角速度(3D)

**注意**：策略网络的MLP部分和Q网络仍然是随机初始化的。

## 路径示例

常见的预训练模型路径格式：

```python
# 使用最佳模型（推荐）
'PRETRAINED_CHECKPOINT_PATH': 'pretrained_models/aux_pretrain_expert_20251108_165904/best_aux_model.pt'

# 使用特定epoch的检查点
'PRETRAINED_CHECKPOINT_PATH': 'pretrained_models/aux_pretrain_expert_20251108_165904/checkpoint_ep3300.pt'

# 使用最终模型
'PRETRAINED_CHECKPOINT_PATH': 'pretrained_models/aux_pretrain_expert_20251108_165904/final_aux_model.pt'
```

## 优势

✓ **更快的收敛**：视觉编码器已经学会了从图像中提取有用的状态信息  
✓ **更好的性能**：预训练提供了更好的特征表示  
✓ **更稳定的训练**：减少了训练初期的不稳定性  
✓ **更少的样本需求**：需要更少的训练数据即可达到相同性能  

## 故障排除

### 问题：找不到预训练文件

**解决方法**：
1. 检查路径是否正确（使用绝对路径或相对于项目根目录的路径）
2. 确认已经运行过 `pretrain_aux.py`
3. 检查 `pretrained_models/` 目录下是否存在对应文件

### 问题：加载时出现维度不匹配错误

**解决方法**：
1. 确保预训练模型的架构与当前模型一致
2. 检查 `config.py` 中的以下参数是否与预训练时相同：
   - `RESNET_AUX_DIM`
   - `GRU_AUX_DIM`
   - `GRU_LAYER`
   - `DROP_OUT`

### 问题：训练性能没有提升

**可能原因**：
1. 预训练不充分（训练轮数太少）
2. 预训练数据质量不高（使用random模式而非expert模式）
3. 预训练环境与主训练环境差异过大

**解决方法**：
- 使用expert模式重新预训练
- 增加预训练的epoch数量（建议至少500轮）
- 确保预训练时的门运动参数与主训练一致

## 高级选项

### 冻结预训练权重（可选）

如果你想在训练初期冻结视觉编码器，可以在加载权重后添加：

```python
# 在main.py的加载代码后添加
if args['USE_PRETRAINED_VISION'] and FREEZE_VISION_ENCODER:
    for param in agent.policy.GRU.parameters():
        param.requires_grad = False
    print("Froze vision encoder for initial training")
```

稍后在训练循环中解冻：

```python
# 在训练循环中
if updates == UNFREEZE_AT_UPDATE:
    for param in agent.policy.GRU.parameters():
        param.requires_grad = True
    print("Unfroze vision encoder")
```

## 相关文件

- `pretrain_aux.py`：预训练脚本
- `main.py`：主训练脚本（已集成预训练加载）
- `example_load_pretrained.py`：加载预训练权重的示例代码
- `config.py`：配置参数

## 参考

更多详细信息请参考：
- `pretrain_aux.py` 顶部的文档字符串
- `example_load_pretrained.py` 中的使用示例
