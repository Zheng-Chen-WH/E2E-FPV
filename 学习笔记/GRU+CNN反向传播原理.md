# PyTorch Autograd在多任务模型中的反向传播

在使用PyTorch构建复杂的深度学习模型时，例如一个包含CNN和RNN并拥有多个损失函数的架构，在反向传播的过程中梯度是如何从一个模块流向另一个模块的？为什么将位于不同位置的损失加权相加后进行反向传播是可行的？

下面以ResNet+GRU多任务模型为例，介绍PyTorch自动微分引擎（Autograd）。
在这一模型中，ResNet处理图像并提取相对位置、姿态信息，并由GRU处理上述信息得到动态的速度、角速度等信息。为利于训练，ResNet和GRU的输出层都包含显式物理量辅助头和特征向量，数据流如下：

```
[图像序列 B,T,C,H,W]   [动态数据序列 B,T,F_dyn]
         |                       |
         | (逐帧 t=1...T)         | (逐帧 t=1...T)
         |                       |
+--------v-----------------------+
|   CustomResNetWithAuxHead      |
|                                |
|   +------------------------+   |
|   |   ResNet 核心部分       |   |
|   +------------------------+   |
|      |               |         |
|      |               |         |
+------|---------------+---------+
       |               |
       | (主特征)      +---------------------> [ResNet姿态预测 B,T,F_pose] -----> MSE Loss
       | B,T,512         (ResNet辅助头)
       |
+------v-----------------v-------+
|  融合 (Concatenate)            |
+--------------------------------+
       |
       | (融合特征序列 B,T,512+F_dyn)
       |
+------v-------------------------+
|           GRU                  |
+--------------------------------+
       |
       | (GRU输出序列 B,T,H_gru)
       |
+------|----------------------------------------------------+
       |                                                    |
       |                                                    |
+------v-------------+                             +--------v-----------+
| GRU辅助头 (Linear) |                             | 取最后隐藏状态    |
+--------------------+                             +--------------------+
       |                                                    |
       |                                                    |
       v                                                    v
[GRU速度预测 B,T,F_vel] -----> MSE Loss        [最终特征向量 B,H_gru] -----> 最终任务 (如分类)
                                                                             |
                                                                             v
                                                                       CrossEntropy Loss
```
在进行反向传播时，代码为：
```
total_loss = loss_final + w_resnet * loss_resnet + w_gru * loss_gru
# 反向传播 (一次反向传播会计算所有部分的梯度)
optimizer.zero_grad()
total_loss.backward()
optimizer.step()
```

### 核心基石：动态计算图 (Dynamic Computation Graph)

要理解一切，首先要记住PyTorch的核心机制：**动态计算图**。

当执行前向传播时，PyTorch会像录像一样，记录下所有Tensor的操作以及它们之间的依赖关系，从而构建一个有向无环图（DAG）。

- **节点 (Nodes)**：代表Tensor的操作（如 `+`, `*`, `torch.cat`, `nn.Conv2d`）。
- **边 (Edges)**：代表Tensor本身，它在不同操作之间流动。

这个图包含了从输入到**所有**输出（包括用于计算损失的中间输出）的完整路径。当我们调用 `.backward()` 时，Autograd引擎就会沿着这个图反向追溯，计算梯度。

---

### 问题一：梯度从GRU到ResNet

**场景**: 最终的分类损失 `loss_final` 是在GRU的输出上计算的。这个梯度是如何作用到序列开头的ResNet上的？

**答案**: 通过**时间反向传播（BPTT）**和**共享权重**机制。

> **Step 1: 梯度的起点**
>
> 当我们调用 `total_loss.backward()`，梯度从最终损失 `loss_final` 开始反向传播。

> **Step 2: 穿越GRU的递归网络**
>
> GRU的最后一个隐藏状态（例如 `t=4`）依赖于前一个状态（`t=3`）和当前的输入（`gru_input_4`）。根据链式法则，梯度会从 `t=4` 的状态反向传播到 `t=3` 的状态和 `t=4` 的输入。这个过程会像多米诺骨牌一样，一直回溯到序列的起点 `t=1`。这就是所谓的**时间反向传播**。

> **Step 3: 在“融合点”解耦**
>
> 在每个时间步 `t`，GRU的输入 `gru_input_t` 是由 `resnet_main_feat_t` 和 `dynamic_data_t` 拼接（`torch.cat`）而成的。
>
> ```python
> # 前向传播中的操作
> gru_input_t = torch.cat((resnet_main_feat_t, dyn_t), dim=1)
> ```
>
> Autograd知道这个拼接操作。当梯度流回 `gru_input_t` 时，它会自动“解开”这个操作，将梯度的一部分分配给 `resnet_main_feat_t`，另一部分分配给 `dyn_t`。

> **Step 4: 汇聚到共享的ResNet**
>
> 这是最关键的一步！在我们的模型中，序列中每一帧的图像特征 `resnet_main_feat_1`, `...`, `resnet_main_feat_4` 都是由**同一个** `self.image_feature_extractor` 实例计算的。
>
> 因此，从GRU反向传播回来的、分属于不同时间步的梯度（`grad_t1`, `grad_t2`, `grad_t3`, `grad_t4`），最终都会流向同一个ResNet模型的参数上。PyTorch会自动将这些梯度**累加**起来，作为该ResNet参数的总梯度。
>
> **最终结果**: ResNet的权重更新，是它在**整个序列中所有时间步表现**的综合结果，而不仅仅是最后一帧。

---

### 问题二：损失的汇合：为何简单的加法如此强大？

**场景**: 我们的总损失是三个位于模型不同位置的损失的加权和。为什么可以这样做？`backward()`如何处理？

```python
# 位于模型不同位置的三个损失
loss_resnet = loss_fn_resnet_aux(resnet_preds, mock_pose_labels)
loss_gru = loss_fn_gru_aux(gru_preds, mock_velocity_labels)
loss_final = loss_fn_final(final_classification_preds, mock_final_labels)

# 简单地加权求和
total_loss = loss_final + w_resnet * loss_resnet + w_gru * loss_gru

# 一次反向传播
total_loss.backward()
```
**答案**: 因为计算图将所有损失都视为图的 **“叶子”** 节点，total_loss 只是一个新的、**连接**它们的根节点。

#### Step 1: 构建总损失节点

当执行 total_loss = ... 时，并没有做什么神奇的事情, 只是在计算图的末尾增加了几个操作节点（乘法和加法），并将 loss_final, loss_resnet, loss_gru 作为它们的输入，最终汇合到一个名为 total_loss 的新节点。

#### Step 2: 从新的“根”开始反向传播

调用 total_loss.backward() 时，Autograd从这个唯一的根节点出发。根据链式法则：

$
d(Total\_Loss)/dW = d(Loss\_final)/dW + w\_resnet*d(Loss\_resnet)/dW + w\_gru*d(Loss\_gru)/dW
$

这意味着，从 total_loss 出发的梯度会根据权重分配给三个子损失，然后同时地、并行地沿着它们各自的路径在计算图中反向传播。

#### Step 3: 梯度的自动累加

想象一下ResNet中的一个权重参数 W_res。它的梯度会从两个源头传来：

- 来自 loss_resnet 的直接路径。
- 来自 loss_final 和 loss_gru 穿越GRU后的间接路径。

PyTorch的Autograd引擎会自动处理这一切。当梯度从不同路径到达 W_res 时，它们会被自动累加到该参数的 .grad 属性上。您无需手动干预。

**为何“分步反向传播”是错误的？**

设想的“先对A反向传播并更新，再对B反向传播...”是不可行的。因为一旦您执行了第一次更新（optimizer.step()），模型的权重就发生了改变，这会使您最初构建的计算图失效。所有梯度都必须基于同一次前向传播的、未被修改的计算图来计算，而 total_loss.backward() 完美地保证了这一点。

### 结论
PyTorch的Autograd机制极其强大和优雅。作为用户，需要关注的是：

- 正确构建模型的前向传播逻辑。
- 在计算图的任何位置定义您需要的损失函数。
- 将所有损失以数学上合理的方式组合成一个标量（total_loss）。

完成这三步后，只需调用一次 .backward()，然后信任PyTorch会为您正确地计算和累加所有相关的梯度。