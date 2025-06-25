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

    
