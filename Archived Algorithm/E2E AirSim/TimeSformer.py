import os
import torch
from PIL import Image
from torchvision import transforms
from timesformer_pytorch import TimeSformer

# Step 1: 初始化 TimeSformer 模型
# 配置 Small 模型（适合计算效率与性能均衡的场景）
model = TimeSformer(
    dim=512,               # Transformer嵌入维度（输出）
    image_size=256,        # 图像尺寸（根据输入图像调整为 256x144）
    patch_size=16,         # Patch大小
    num_frames=4,          # 输入帧数
    num_classes=1,         # 随便给一个分类数量，后面替换分类头
    depth=8,               # Transformer深度（Small模型）
    heads=8                # 注意力头数量
)

# 替换分类头为 Identity 层，直接输出特征
model.cls_head = torch.nn.Identity()
# 移除 to_out 部分
model.to_out = torch.nn.Identity()

# Step 2: 图像序列文件夹路径
folder_path = "/media/zheng/A214861F1485F697/Dataset"  # 替换为图像所在文件夹路径

# Step 3: 图像文件名列表（按照顺序加载）
image_filenames = [f"0_0_{i+1}.png" for i in range(4)]  # 生成文件名列表

# Step 4: 图像预处理
# 将图像转换为模型可接受的格式，同时调整尺寸
transform = transforms.Compose([
    transforms.Resize((256, 144)),  # 调整图像尺寸为模型输入的尺寸
    transforms.ToTensor(),          # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# Step 5: 加载和预处理图像序列
input_sequence = []
for filename in image_filenames:
    file_path = os.path.join(folder_path, filename)  # 获取完整路径
    image = Image.open(file_path).convert("RGB")    # 打开图像，确保是 RGB 格式
    input_sequence.append(transform(image))         # 应用预处理并添加到序列

# 将序列堆叠为 Tensor (num_frames, channels, height, width)
input_sequence = torch.stack(input_sequence, dim=0)  # 输出维度: (4, 3, 256, 144)

# 添加 Batch 维度 (batch_size, num_frames, channels, height, width)
input_sequence = input_sequence.unsqueeze(0)  # 输出维度: (1, 4, 3, 256, 144)
print("input sequence shape", input_sequence.shape)
print("sssss:",torch.cat(torch.FloatTensor(input_sequence)).shape)

# Step 6: 输入到 TimeSformer 模型
features = model(input_sequence)  # 提取特征张量
print("feature shape:", features.shape)

# 打印特征输出形状
# print("Extracted features shape:", features.shape)
# print(model)

# Step 7: 用于 MLP 决策
# 假设 MLP 需要的输入是特征张量，直接对 features 进行后续处理
# 可以根据任务需求将 features 展平或进一步处理
mlp_input = features.view(features.size(0), -1)  # 展平特征(1, num_frames * dim)
print("MLP input shape:", mlp_input.shape)

# 示例：定义简单的 MLP
class DecisionMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 定义 MLP（根据任务调整输入维度和输出维度）
mlp = DecisionMLP(
    input_dim=mlp_input.shape[1],  # 输入特征维度
    hidden_dim=128,                # 隐藏层维度（可根据任务调整）
    output_dim=3                   # 输出维度（例如无人机决策类别：左、中、右）
)

# MLP 输出结果
mlp_output = mlp(mlp_input)
print("MLP output shape:", mlp_output.shape)