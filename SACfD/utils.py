import math
import torch
import os
from torchvision import transforms
from PIL import Image
from pathlib import Path

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
  # 动态计算批次内专家动作的均值
  # 使用 .detach() 来确保这个计算不会成为反向传播图的一部分
  with torch.no_grad():
    batch_mean = torch.mean(y_true)
  
    # 计算每个样本的权重
    # 专家动作离批次均值越远，权重越大。
    # 加1.0是为了保证基础权重至少为1。
    weights = 1.0 + torch.abs(y_true - batch_mean)

  # 3. 计算每个样本原始的MSE
  per_sample_mse = F.mse_loss(y_pred, y_true, reduction='none') # reduction='none'，返回的是每个样本的损失，而不是整体损失的平均值或总和。

  # 4. 应用权重并计算最终的平均损失
  weighted_mse = per_sample_mse * weights
  final_loss = torch.mean(weighted_mse)
  
  return final_loss

def conversion(control_signal):
    rotor_turning_directions = torch.tensor([1.0, 1.0, -1.0, -1.0], device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    total_thrust = -torch.sum(control_signal, dim=1)  # 总推力
    rotor_torques_b = control_signal * rotor_turning_directions.unsqueeze(0) # (K, 4)

    T_FR = control_signal[:, 0]; T_RL = control_signal[:, 1]
    T_FL = control_signal[:, 2]; T_RR = control_signal[:, 3]
    tau_x_b = T_FL + T_RL - T_FR - T_RR
    tau_y_b = T_FR + T_FL - T_RL - T_RR
    tau_z_b = torch.sum(rotor_torques_b, dim=1)
    return total_thrust, tau_x_b, tau_y_b, tau_z_b

def physics_MSE(y_pred, y_true, weighted = False):
    weight = 1.0
    with torch.no_grad():
        # 我们可以计算每个动作的“幅度”来判断其“独特性”。
        # 4个电机PWM的平均值是衡量动作幅度的一个很好的指标。
        batch_mean_by_sample = torch.mean(y_true,dim=1) # 形状: [B]
        batch_mean = torch.mean(batch_mean_by_sample) # 标量
        # 专家行为远离批次平均值的样本会获得更高的权重。
        importance_weights = 1.0 + torch.abs(batch_mean_by_sample - batch_mean) # 形状: [B]

    thrust_pred, tau_x_pred, tau_y_pred, tau_z_pred = conversion(y_pred)
    thrust_true, tau_x_true, tau_y_true, tau_z_true = conversion(y_true)
    thrust_loss =  F.mse_loss(thrust_pred, thrust_true, reduction='none') 
    # print(thrust_loss.shape)
    # print(importance_weights.shape)
    tau_loss = F.mse_loss(tau_x_pred, tau_x_true, reduction='none') + F.mse_loss(tau_y_pred, tau_y_true, reduction='none') +\
                                    0.5 * F.mse_loss(tau_z_pred, tau_z_true, reduction='none')
    # 应用重要性权重并求均值得到最终损失
    
    weighted_thrust_loss = torch.mean(thrust_loss * importance_weights)
    weighted_tau_loss = torch.mean(tau_loss * importance_weights)
    final_loss = weight * weighted_thrust_loss + weighted_tau_loss
    # print(f"thrust:{F.mse_loss(thrust_pred, thrust_true)}, tau:{(F.mse_loss(tau_x_pred, tau_x_true) + F.mse_loss(tau_y_pred, tau_y_true) + F.mse_loss(tau_z_pred, tau_z_true))}")
    # print(f"tau_x:{F.mse_loss(tau_x_pred, tau_x_true)}, tau_y:{F.mse_loss(tau_y_pred, tau_y_true)} ,tau_z:{F.mse_loss(tau_z_pred, tau_z_true)}")
    return final_loss

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

def build_sincos_pos_embed(num_positions: int, embed_dim: int):
    """
    为Transformer生成基于sin/cos函数的位置编码表。
    Args:
        num_positions (int): 序列的长度 (在ViT中等于 num_patches + 1)。
        embed_dim (int): 嵌入向量的维度 (d_model)。

    Returns:
        torch.Tensor: 位置编码张量，形状为 (1, num_positions, embed_dim)。
    """
    
    # 初始化一个位置编码矩阵
    pe = torch.zeros(num_positions, embed_dim)

    # 创建一个位置索引的列向量 [0, 1, ..., num_positions-1]
    position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)

    # 创建一个用于计算分母的项
    # div_term = 1 / (10000^(2i / d_model))
    # 使用exp和log可以获得更好的数值稳定性
    # torch.arange(0, embed_dim, 2)生成一个包含所有从0开始直到embed_dim的偶数的序列，因为原公式中是用2i和2i+1所以这里只需要128个数
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

    # 计算偶数维度的位置编码 (使用sin)
    # 对于位置编码矩阵pe的所有行，取出所有偶数列（0, 2, 4, ...），用 sin 函数和我们计算好的div_term来填充它们。
    pe[:, 0::2] = torch.sin(position * div_term)

    # 计算奇数维度的位置编码 (使用cos)
    # 对于位置编码矩阵pe的所有行，取出所有奇数列（1, 3, 5, ...），用 cos 函数和div_term来填充它们。
    pe[:, 1::2] = torch.cos(position * div_term)
    
    # 增加一个batch维度以匹配输入格式 (1, num_positions, embed_dim)
    pe = pe.unsqueeze(0)
    
    return pe
   