import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class ResidualBlock(nn.Module):
    """
    基础残差块，包含两个3x3卷积层。
    
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # nn.Conv2d用于执行二维卷积操作
        # 卷积核与感受野内的值进行矩阵相乘并求和，输出一个值
        # in_channels: 输入图像的通道数。对于灰度图像，in_channels 为 1。对于RGB图像，in_channels 为 3。
            # 如果你的输入是上一层卷积的输出，那么 in_channels 就是上一层的 out_channels
        # out_channels: 卷积层输出的特征图的数量，也就是卷积核（或滤波器）的数量
        # kernel_size: 卷积核（或滤波器）的大小。设置为 3，表示卷积核是一个 3x3 的正方形。
            # 也可以使用一个元组来指定非正方形的卷积核，例如 (3, 5) 表示 3 行 5 列的卷积核。
        # stride: 卷积核在输入特征图上滑动的步长
            # stride=1 (默认值)，卷积核每次移动一个像素
            # stride=2，卷积核每次移动两个像素，导致输出特征图的尺寸减半，常用于降采样
            # 可以指定一个元组 (如 stride=(1, 2))，表示水平和垂直方向的步长不同
        # padding: 在输入特征图的边界周围添加的零的数量
            # 主要目的是为了在卷积操作中保留输入特征图的空间尺寸，防止边缘信息丢失，并使得输出特征图的尺寸与输入特征图更接近或相同
        # bias: 一个布尔值，表示是否在卷积操作后添加偏置
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # 创建一个二维批归一化（Batch Normalization）层
        # nn.BatchNorm2d 层紧跟在 nn.Conv2d 层之后，num_features应该与前面 nn.Conv2d 层的 out_channels 相匹配
        # BatchNorm2d 层会对输入数据的每个通道独立地进行归一化操作。
        # 对于每个批次（mini-batch）的输入数据，它会计算每个通道的均值和方差，然后使用这些统计量来归一化该通道的数据，使其均值为 0，方差为 1。
        # 它还会学习两个可训练的参数：缩放因子𝛾(gamma)和偏移因子𝛽(beta)，用来对归一化后的数据进行线性变换，以恢复网络的表达能力
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 定义一个残差跳跃连接（shortcut connection）
        self.shortcut = nn.Sequential() # 一个空的 nn.Sequential()起到恒等映射的作用，它将输入直接传递到输出
        if stride != 1 or in_channels != out_channels: # 为了使跳跃连接的输出尺寸与主路径的输出尺寸匹配，跳跃连接本身也需要进行相应的空间降采样
            self.shortcut = nn.Sequential(
                # 1x1 卷积层（也称为逐点卷积），主要作用不是提取空间特征，而是用来改变特征图的通道数 (in_channels 变为 out_channels)
                # stride与 if 条件中的 stride 保持一致，如果主路径进行了空间降采样，1x1卷积也会执行相同的降采样，确保跳跃连接的输出空间尺寸与主路径的输出匹配
                # 这里不进行padding是匹配主路的kernel=3，如果kernel较大时在这里也需要处理padding以匹配主路特征图
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), 
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x))) # 好像leakyrelu会好一点
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out)
        return out

class ResNet(nn.Module):
    """自定义ResNet，主输出为特征向量，并带有一个用于预测位姿的辅助头。"""
    def __init__(self, num_aux_outputs, input_channels=3): # 不用指定像素，只指定通道数就行
        super(ResNet, self).__init__()
        # 参考真·ResNet，第一层是size=7的卷积核，padding为3，但是这样输出的特征图尺寸是取决于x奇偶性的(x+1)/2
        # 一般都是奇数大小卷积核，有个明确的中心，所以stride=2的情况下输出的图像尺寸一定不确定
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 定义一个二维最大池化层，通过在一个局部区域（由 kernel_size 定义）内取最大值来对输入特征图进行下采样（降采样）
            # 降低维度：减少特征图的空间尺寸，从而减少后续层的计算量和参数数量
            # 提取主要特征：保留局部区域内最显著的特征（最大值），忽略不重要的细节
            # 增强平移不变性：即使输入中的特征发生了轻微的平移，由于取最大值的操作，输出特征也可能保持不变，这有助于模型对特征的位置不那么敏感
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.block1 = ResidualBlock(64, 64, stride=1)
        # 不断通过stride=2下采样，缩小特征图的尺寸同时增加特征图的通道数
        # 深度卷积神经网络中非常常见的模式，用于在网络深层提取更高级、更抽象的特征，同时减少空间维度以节省计算量和参数
        self.block2 = ResidualBlock(64, 128, stride=2)
        self.block3 = ResidualBlock(128, 256, stride=2)
        # self.block4 = ResidualBlock(256, 512, stride=2)

        # 二维自适应平均池化层，指定的是目标输出尺寸，而不是核大小和步长。
        # 网络会根据输入特征图的尺寸，自动计算出合适的 kernel_size 和 stride 来达到您指定的目标输出尺寸
        # 设置 output_size=(1, 1) 时，nn.AdaptiveAvgPool2d会取输入特征图的所有像素的平均值，为每个通道生成一个单一的值
        # 用来替代传统的、在卷积层之后使用的全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 显式输出相对位姿的辅助输出头
        # nn.Flatten将输入的多维张量（Tensor）展平（flatten）成一维张量
        # 保留第一个维度，通常是批量大小，然后将所有后续维度（通道、高度、宽度等）合并（或展平）成一个单一的维度
        # 这里是(batch_size, 512,1,1)被转成(batch_size,512)
        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_aux_outputs)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        # x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # x = self.block4(x)
        aux_output = self.aux_head(x) # 第四个残差块出来就直接去辅助头了
        main_features = self.avgpool(x) # 第四个残差块出来经过平均池化形成主特征向量
        main_features = torch.flatten(main_features, 1)
        return main_features, aux_output # 返回主特征和辅助头的显式输出

class GRU(nn.Module):
    """
    集成了ResNet和GRU的模型。
    - ResNet提取空间特征，并有辅助头预测姿态/位置。
    - GRU处理时序信息，并有辅助头预测速度/角速度。
    - 最终输出一个融合时空信息的特征向量。
    """
    def __init__(self, resnet_aux_outputs, gru_hidden_dim, gru_aux_outputs, gru_layers=1, dropout=0.3):
        """
        Args:
            resnet_aux_outputs (int): ResNet辅助头输出维度 (例如: 6个位姿参数)
            gru_hidden_dim (int): GRU隐藏层维度（特征向量维度）
            gru_aux_outputs (int): GRU辅助头输出数量 (例如: 6个速度/角速度参数)
        """
        super(GRU, self).__init__()
        
        # ResNet逐帧提取特征
        self.image_feature_extractor = ResNet(num_aux_outputs=resnet_aux_outputs)
        resnet_main_feature_dim = 256 # ResNet的主输出维度
        
        # GRU的输入维度 = 图像主特征 + 外部动态特征，暂时先只有图像
        gru_input_dim = resnet_main_feature_dim # + external_dynamic_features
        
        # GRU处理时序输出时序信息
        # input_size是输入特征的维度，即对于序列中的每个时间步，输入到 GRU 单元的数据的特征数量
        # hidden_size是隐藏状态 (hidden state) 的维度。
            # GRU 单元在每个时间步计算并更新一个隐藏状态，hidden_size 定义了这个隐藏状态向量的长度
        # num_layers是堆叠的 GRU 层数。
            # 如果 num_layers > 1，那么 GRU 网络将由多个 GRU 层堆叠而成。
            # 第一个 GRU 层的输入是原始序列数据。随后的每个 GRU 层的输入是前一个 GRU 层的输出序列。
            # 这种堆叠结构可以帮助模型学习更复杂、更高层次的时间依赖关系
        # batch_first是一个布尔值，用于指定输入和输出张量的维度顺序。
            # batch_first=True，那么输入和输出张量的形状将是 (batch, seq_len, features)
        # dropout 除最后一层之外的 GRU 层输出的 Dropout 概率。
            # Dropout 是一种正则化技术，用于防止过拟合。在训练过程中，它会随机地“关闭”一部分神经元的输出。
            # Dropout 只应用于堆叠 GRU 层之间的连接，而不会应用于 GRU 单元内部的循环连接。
        # 【关于GRU的两个门】PyTorch 会自动在内部创建实现这两个门所需的所有权重矩阵和偏置项
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0
        )
        
        # GRU的辅助头: 用于显式预测速度/角速度
        # 它作用于GRU的整个输出序列，以得到每个时间步的预测
        self.gru_aux_head = nn.Linear(gru_hidden_dim, gru_aux_outputs)

    def forward(self, image_sequence, hidden_state=None):
        """
        Args:
            image_sequence (Tensor): 形状为 (Batch批量大小, Time帧数, Channels通道数, Height高度, Width宽度) 的图像序列
        
        返回:
            一个元组，包含：
            - final_feature_vector (Tensor): GRU最后的隐藏状态, 形状为 (B, H_gru)
            - resnet_aux_predictions (Tensor): ResNet的姿态预测, 形状为 (B, T, F_pose)
            - gru_aux_predictions (Tensor): GRU的速度预测, 形状为 (B, T, F_vel)
        """
        # 原始输入形状:
        # image_sequence: (B, T, C, H, W)  (B=批量大小, T=帧数, C=通道数, H=高度, W=宽度)
        B, T, C, H, W = image_sequence.shape

        # 将时间和批次维度“压平” (Flatten/Reshape)
        # 将 (B, T, C, H, W) -> (B * T, C, H, W)
        # 让ResNet一次性处理所有序列中的所有帧，如同一个超大的batch
        input_for_resnet = image_sequence.view(B * T, C, H, W)

        # 一次性通过特征提取器 (Single Forward Pass)
        # 这是关键的并行化步骤。GPU会并行处理这 B*T 张图片。
        # resnet_main_feat 的形状会是 (B * T, feat_dim)
        # resnet_aux_pred 的形状会是 (B * T, 6)  (假设6D姿态)
        resnet_main_feat, resnet_aux_pred = self.image_feature_extractor(input_for_resnet)

        # 恢复时间和批次维度 (Unflatten/Reshape)，将ResNet的输出变回序列格式，以供GRU使用

        # 准备GRU的输入序列
        # 将 (B * T, feat_dim) -> (B, T, feat_dim)
        gru_inputs_sequence = resnet_main_feat.view(B, T, -1) # -1 会自动推断为 feat_dim

        # 整理辅助任务的预测序列
        # 将 (B * T, 6) -> (B, T, 6)
        resnet_aux_predictions = resnet_aux_pred.view(B, T, -1) # -1 会自动推断为 6

        # 现在 gru_inputs_sequence 和 resnet_aux_predictions 就是你想要的序列张量了
        # 并且这个过程比 for 循环快几个数量级。
        
        # GRU处理整个序列
        # 将 gru_inputs_sequence (形状为 (B, T, C')) 传递给 self.gru 时，PyTorch 的 nn.GRU 模块会在内部自动地、高效地循环 T 次。
        # 每次循环中，它会取出序列中的一个时间步 (t) 的所有批次数据 (gru_inputs_sequence[:, t, :])，并与当前的隐藏状态一起，计算出下一个时间步的隐藏状态。
        # 这个内部循环是高度优化的，通常通过 C++ 或 CUDA 实现，比 Python 循环要高效得多
        # gru_output_sequence 是 GRU 在每个时间步的输出（通常是隐藏状态）。形状是 (B, T, gru_hidden_dim)，包含了序列中每个时间步的隐藏状态输出。
        # last_hidden_state 是 GRU 最后一个时间步的隐藏状态，形状是 (num_layers * num_directions, B, gru_hidden_dim)。
        # 如果是单向 GRU，则形状为 (num_layers, B, gru_hidden_dim)。
        gru_output_sequence, last_hidden_state = self.gru(gru_inputs_sequence, hidden_state)
        
        # GRU的输出分两路
        # GRU的辅助头，对每一帧进行显式的速度/角速度预测，（B,T,6）
        gru_aux_predictions = self.gru_aux_head(gru_output_sequence)
        
        # 最终的融合时空特征向量 (取最后一层的最后一个时间步的隐藏状态)
        final_feature_vector = last_hidden_state[-1, :, :]
        
        return final_feature_vector, resnet_aux_predictions, gru_aux_predictions, last_hidden_state

def init_weights(m):
    """
    根据模块类型应用Kaiming, Orthogonal等最佳实践的权重初始化。
    使用方法: model.apply(init_weights)
    """
    if isinstance(m, nn.Conv2d):
        # Kaiming 正态分布初始化，专为ReLU设计
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            # 偏置通常初始化为0
            nn.init.constant_(m.bias, 0)
            
    elif isinstance(m, nn.BatchNorm2d):
        # BN层的gamma初始化为1, beta初始化为0
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                # 输入到隐藏层的权重，使用Xavier均匀分布
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # 隐藏层到隐藏层的权重，使用正交初始化
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # 偏置初始化为0
                param.data.fill_(0)
                
    elif isinstance(m, nn.Linear):
        # Kaiming 正态分布初始化
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        # 还是会执行n-1次，但循环最后一次（j=n-2）时激活函数是恒等映射
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)
    # 生成网络且允许灵活修改，但全都是全连接层，其中size可以是一串序列，每个元素都描述大小；同时j和j+1在循环中自动确保相乘时行数列数相等
    # nn.Identity 意味着网络的输出层将应用恒等映射作为激活函数，即输出值与输入值完全一致，没有经过任何变换
    # 灵活用星号解包
    # nn.Linear(a, b) 【不是一个单纯的全连接层】是 PyTorch 中的一个线性层（linear layer）的构造函数。它创建了一个将输入特征映射到输出特征的线性变换。
    # nn.Linear(a, b) 接受表示输入特征的维度a和输出特征的维度b，线性层的作用是通过学习一组权重和偏置，将输入特征进行线性变换，得到输出特征。
    # output = input * weight^T + bias
    # 其中，input 是输入特征，weight 是形状为 (b, a) 的权重矩阵，bias 是形状为 (b,) 的偏置项。^T 表示权重矩阵的转置。

class GaussianPolicy(nn.Module):
    def __init__(self, embedding_dim, num_inputs, num_actions, hidden_sizes, 
                 activation, max_action, min_action, 
                 resnet_aux_outputs, gru_aux_outputs, 
                 gru_layers, dropout, RE_PARAMETERIZATION=True):
        super(GaussianPolicy, self).__init__()
        self.GRU = GRU(resnet_aux_outputs, embedding_dim, gru_aux_outputs, gru_layers=gru_layers, dropout=dropout)

        # 在拼接后、MLP前加入归一化层LayerNorm
        concatenated_dim = embedding_dim + num_inputs # feature维度+state维度
        # self.concat_norm = nn.LayerNorm(concatenated_dim)

        self.mlp_network=mlp([concatenated_dim] + list(hidden_sizes), activation, activation) #特征向量+目标位置+往期动作
        self.mu_layer = nn.Linear(hidden_sizes[-1], num_actions)
        # 生成mu的层
        self.log_std_layer = nn.Linear(hidden_sizes[-1], num_actions)
        self.re_parameterization=RE_PARAMETERIZATION
        # 动作缩放，这里在外部解决，避免动作相差太小
        self.action_scale = torch.FloatTensor([
                (max_action - min_action ) / 2.])
        self.action_bias = torch.FloatTensor([
                (max_action + min_action ) / 2.])

    def forward(self, img_sequence, state, hidden_state=None):
        # 输入到 GRU
        features, resnet_preds, gru_preds, new_hidden_state = self.GRU(img_sequence, hidden_state)  # 提取特征张量
        concatenated_input = torch.cat([features, state],1) # 拼接特征张量和状态
        # 先进行层归一化
        # normalized_input = self.concat_norm(concatenated_input)
        # 检查normalized_input的量级
        x=self.mlp_network(concatenated_input)
        # print(f"normalized input:{normalized_input}")
        mean = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std, resnet_preds, gru_preds, new_hidden_state

    def sample(self, img_sequence, state, hidden_state=None):
        mean, log_std, resnet_output, gru_output, new_hidden_state = self.forward(img_sequence, state, hidden_state)
        # print(f"mean before tanh:{mean}")
        std = torch.exp(log_std)
        # print(std)
        normal = Normal(mean, std)
        x_t = normal.rsample() # 重参数化
        # 【以下方案是代码作者自己的方案，先得到tanh动作再对这一动作求log】
        y_t = torch.tanh(x_t) 
        action = y_t * self.action_scale + self.action_bias #不是重参数化，只是单纯把值调整到动作空间范围内
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - y_t.pow(2)) + epsilon) 
        # 原论文(21)式
        #log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon) #原论文中公式，但是多了个action_scale
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, resnet_output, gru_output, new_hidden_state # 辅助头输出分别是（B,T,9）和（B,T,6）

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_sizes,activation):
        super(QNetwork, self).__init__()
        #torch.manual_seed(42) #所有随机数种子都用42
        # Q1 architecture
        self.Q_network_1=mlp([num_inputs+num_actions] + list(hidden_sizes)+[1], activation)

        # Q2 architecture
        self.Q_network_2=mlp([num_inputs+num_actions] + list(hidden_sizes)+[1], activation)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        x1 = self.Q_network_1(xu)
        x2 = self.Q_network_2(xu)
        return x1, x2