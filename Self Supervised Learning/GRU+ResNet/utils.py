import math
import torch
import os
from torchvision import transforms
from PIL import Image

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def img_load(file_names):
    folder_path = "/media/zheng/A214861F1485F697/Dataset"  # 图像序列文件夹路径
    # 图像预处理, 将图像转换为模型可接受的格式，同时调整尺寸
    transform = transforms.Compose([
        transforms.Resize((256, 144)),  # 调整图像尺寸为模型输入的尺寸
        transforms.ToTensor(),          # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])  # 标准化
    # 加载和预处理图像序列
    img_sequence = []
    for filename in file_names:
        file_path = os.path.join(folder_path, filename)  # 获取完整路径
        image = Image.open(file_path).convert("RGB")    # 打开图像，确保是 RGB 格式
        img_sequence.append(transform(image))         # 应用预处理并添加到序列
    # 将序列堆叠为 Tensor (num_frames, channels, height, width)
    input_sequence = torch.stack(input_sequence, dim=0)  # 输出维度: (4, 3, 256, 144)
    # 添加 Batch 维度 (batch_size, num_frames, channels, height, width)
    input_sequence = input_sequence.unsqueeze(0)  # 输出维度: (1, 4, 3, 256, 144)
    return img_sequence #返回处理好的张量

def map_value(x, a, b, c, d):
    """
    将值 x 从范围 [a, b] 映射到范围 [c, d]。
    参数:
    x: 待映射的值。
    a: 原范围的下限。
    b: 原范围的上限。
    c: 目标范围的下限。
    d: 目标范围的上限。
    返回: 映射后的值。
    """
    # 确保 x 在范围 [a, b] 内
    # if x < a or x > b:
    #     raise ValueError(f"x 应在 {a} 和 {b} 之间")

    # 线性映射公式
    mapped_value = c + (d - c) * ((x - a) / (b - a))
    return mapped_value

import torch
import torch.nn.functional as F

def weighted_mse_loss(y_pred, y_true):
  """
  计算加权均方误差 (Weighted MSE)。
  权重是根据真实值与批次内真实值均值的距离动态生成的。
  这会惩罚那些远离批次均值的样本，鼓励模型学习数据的完整分布，
  而不是仅仅输出一个全局的平均值。

  Args:
    y_pred (torch.Tensor): 模型的预测值。
    y_true (torch.Tensor): 专家提供的真实值。

  Returns:
    torch.Tensor: 一个标量的损失值。
  """
  # 1. 动态计算批次内专家动作的均值
  # 使用 .detach() 来确保这个计算不会成为反向传播图的一部分
  with torch.no_grad():
    batch_mean = torch.mean(y_true)
  
    # 2. 计算每个样本的权重
    # 专家动作离批次均值越远，权重越大。
    # 加1.0是为了保证基础权重至少为1。
    weights = 1.0 + torch.abs(y_true - batch_mean)

  # 3. 计算每个样本原始的MSE
  per_sample_mse = F.mse_loss(y_pred, y_true, reduction='none')

  # 4. 应用权重并计算最终的平均损失
  weighted_mse = per_sample_mse * weights
  final_loss = torch.mean(weighted_mse)
  
  return final_loss


    
