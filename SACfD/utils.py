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

def find_latest_untested_model(model_dir: str, tested_log_file: str) -> str | None:
    """
    遍历模型目录txt，找到最新的、尚未被测试过的模型文件。

    Args:
        model_dir (str): 存放 .pt 模型文件的目录路径。
        tested_log_file (str): 记录已测试模型文件名的日志文件路径。

    Returns:
        str | None: 如果找到，则返回最新未测试模型的完整路径；
                    否则返回 None。
    """
    model_directory = Path(model_dir)
    log_file = Path(tested_log_file)

    # 获取所有已经测试过的模型文件名集合，提高查找效率
    try:
        with open(log_file, 'r') as f:
            # 使用set存储
            tested_models = set(line.strip() for line in f)
        print(f"已加载 {len(tested_models)} 个已测试模型的记录。")
    except FileNotFoundError:
        # 如果日志文件不存在，说明还没有任何模型被测试过
        tested_models = set()
        print("未找到已测试模型日志，将视所有模型为未测试。")

    # 2. 获取目录下所有的 .pt 模型文件
    all_model_paths = list(model_directory.glob("*.pt"))
    if not all_model_paths:
        print(f"警告: 在目录 '{model_dir}' 中没有找到任何 .pt 文件。")
        return None
    
    # 3. 筛选出所有未被测试过的模型
    untested_model_paths = [
        path for path in all_model_paths if path.name not in tested_models
    ]

    if not untested_model_paths:
        print("所有模型均已测试完毕。")
        return None

    # 4. 对未测试的模型列表，根据文件创建时间（或修改时间）进行排序，找到最新的
    #    os.path.getctime: 在Windows上是创建时间，在Unix上是元数据最后修改时间
    #    os.path.getmtime: 文件的最后修改时间。通常更可靠。
    #    我们使用 getmtime 以获得更一致的行为。
    try:
        latest_untested_model = max(untested_model_paths, key=os.path.getmtime)
    except Exception as e:
        print(f"在获取文件时间戳时发生错误: {e}")
        return None
        
    print(f"找到最新的未测试模型: {latest_untested_model.name}")
    return str(latest_untested_model)

def mark_model_as_tested(model_path: str, tested_log_file: str):
    """
    将一个模型文件名记录到已测试日志中。

    Args:
        model_path (str): 被测试的模型文件的完整路径。
        tested_log_file (str): 记录已测试模型文件名的日志文件路径。
    """
    log_file = Path(tested_log_file)
    model_name = Path(model_path).name
    try:
        # 使用追加模式 'a'
        with open(log_file, 'a') as f:
            f.write(model_name + '\n')
        print(f"已将模型 '{model_name}' 标记为已测试。")
    except Exception as e:
        print(f"错误：无法将模型标记为已测试。 {e}")
    
