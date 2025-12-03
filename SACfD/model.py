import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from utils import six_d_to_rot_mat, build_sincos_pos_embed

LOG_SIG_MAX = 2
LOG_SIG_MIN = -6
epsilon = 1e-6

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
            
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
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
        # a=0 表示ReLU, mode='fan_in' 保持前向传播时权重的方差
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class ResidualBlock(nn.Module):
    """定义一个包含两个3*3卷积层的残差块

    Args:
        in_channels: int输入通道数，对RGB而言为3
        out_channels: 输出通道数（卷积核数）
        stride: 卷积块移动步幅
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # nn.Conv2d用于执行二维卷积操作
        '''
        卷积核与感受野内的值进行矩阵相乘并求和，输出一个值
        in_channels: 输入图像的通道数。对于灰度图像，in_channels 为 1。对于RGB图像，in_channels 为 3。
            如果输入是上一层卷积的输出，那么 in_channels 就是上一层的 out_channels
        out_channels: 卷积层输出的特征图的数量，也就是卷积核（或滤波器）的数量
        kernel_size: 卷积核（或滤波器）的大小。设置为 3，表示卷积核是一个 3x3 的正方形。
            也可以使用一个元组来指定非正方形的卷积核，例如 (3, 5) 表示 3 行 5 列的卷积核。
        stride: 卷积核在输入特征图上滑动的步长
            stride=1 (默认值)，卷积核每次移动一个像素
            stride=2，卷积核每次移动两个像素，导致输出特征图的尺寸减半，常用于降采样
            可以指定一个元组 (如 stride=(1, 2))，表示水平和垂直方向的步长不同
        padding: 在输入特征图的边界周围添加的零的数量
            主要目的是为了在卷积操作中保留输入特征图的空间尺寸，防止边缘信息丢失，并使得输出特征图的尺寸与输入特征图更接近或相同
        bias: 一个布尔值，表示是否在卷积操作后添加偏置
        '''
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        # 创建一个二维批归一化（Batch Normalization）层
        '''
        nn.BatchNorm2d 层紧跟在 nn.Conv2d 层之后，num_features应该与前面 nn.Conv2d 层的 out_channels 相匹配
        BatchNorm2d 层会对输入数据的每个通道独立地进行归一化操作。
        对于每个批次（mini-batch）的输入数据，它会计算每个通道的均值和方差，然后使用这些统计量来归一化该通道的数据，使其均值为 0，方差为 1。
        它还会学习两个可训练的参数：缩放因子𝛾(gamma)和偏移因子𝛽(beta)，用来对归一化后的数据进行线性变换，以恢复网络的表达能力'''
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 定义一个残差跳跃连接（shortcut connection）
        self.shortcut = nn.Sequential() # 一个空的 nn.Sequential()起到恒等映射的作用，它将输入直接传递到输出
        if stride != 1 or in_channels != out_channels: 
            '''
            为了使跳跃连接的输出尺寸与主路径的输出尺寸匹配，跳跃连接本身也需要进行相应的空间降采样
            1x1 卷积层（也称为逐点卷积），主要作用不是提取空间特征，而是用来改变特征图的通道数 (in_channels 变为 out_channels)
            stride与 if 条件中的 stride 保持一致，如果主路径进行了空间降采样，1x1卷积也会执行相同的降采样，确保跳跃连接的输出空间尺寸与主路径的输出匹配
            这里不进行padding是匹配主路的kernel=3，如果kernel较大时在这里也需要处理padding以匹配主路特征图'''
            self.shortcut = nn.Sequential(
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
    '''自定义ResNet，主输出为特征向量，并带有一个用于预测位姿的辅助头。
        
        ResNet结构分层为编码器整体-阶段（处理的特征图高和宽都保持不变）-残差块-卷积层

    Args:
        args:包含有所有所需参数的字典
    '''
    def __init__(self, args):
        super(ResNet, self).__init__()
        '''
        参考真·ResNet，第一层是size=7的卷积核，padding为3，但是这样输出的特征图尺寸是取决于x奇偶性的(x+1)/2
        一般都是奇数大小卷积核，有个明确的中心，所以stride=2的情况下输出的图像尺寸一定不确定
        '''
        first_layer_dict = args["first_CNN_layer"]
        self.conv1 = nn.Conv2d(args["input_channels"], first_layer_dict["out_put_channel"],
                                kernel_size = first_layer_dict["kernel_size"], 
                                stride = first_layer_dict["stride"], padding = first_layer_dict["padding"], bias=False)
        self.bn1 = nn.BatchNorm2d(first_layer_dict["out_put_channel"])
        
        # 定义一个二维最大池化层，通过在一个局部区域（由 kernel_size 定义）内取最大值来对输入特征图进行下采样（降采样）
        '''
        降低维度：减少特征图的空间尺寸，从而减少后续层的计算量和参数数量
        提取主要特征：保留局部区域内最显著的特征（最大值），忽略不重要的细节
        增强平移不变性：即使输入中的特征发生了轻微的平移，由于取最大值的操作，输出特征也可能保持不变，这有助于模型对特征的位置不那么敏感
        '''
        self.maxpool = nn.MaxPool2d(kernel_size = args["max_pool"]["kernel_size"],
                                     stride = args["max_pool"]["stride"], padding = args["max_pool"]["padding"])

        """用于动态构建包含若干个残差阶段的ResNet"""
        stages = []
        current_channels = first_layer_dict["out_put_channel"]  # 在conv1和maxpool之后的通道数

        # 遍历每个阶段的配置
        for i, (num_blocks, out_channels) in enumerate(zip(args['block_counts'], args['channel_scales'])):
            layers = []
            '''
            每个阶段的第一个block可能需要改变步长来下采样
            除了第一个阶段外，通过stride=2下采样，缩小特征图的尺寸同时增加特征图的通道数
            深度卷积神经网络中非常常见的模式，用于在网络深层提取更高级、更抽象的特征，同时减少空间维度以节省计算量和参数'''
            stride = 2 if i > 0 else 1 # 第一个stage(64->64)的stride为1，其余为2

            # 添加该阶段的第一个block
            layers.append(ResidualBlock(current_channels, out_channels, stride = stride))
            current_channels = out_channels  # 更新当前通道数

            # 添加该阶段剩余的blocks
            for _ in range(1, num_blocks):
                layers.append(ResidualBlock(current_channels, out_channels, stride = 1))
            
            stages.append(nn.Sequential(*layers))
        
        self.stages =  nn.Sequential(*stages)

        # 二维自适应平均池化层，指定的是目标输出尺寸，而不是核大小和步长。
        '''
        网络会根据输入特征图的尺寸，自动计算出合适的 kernel_size 和 stride 来达到您指定的目标输出尺寸
        设置 output_size=(1, 1) 时，nn.AdaptiveAvgPool2d会取输入特征图的所有像素的平均值，为每个通道生成一个单一的值
        用来替代传统的、在卷积层之后使用的全连接层'''
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 显式输出相对位姿的辅助输出头
        '''
        nn.Flatten将输入的多维张量（Tensor）展平（flatten）成一维张量
        保留第一个维度，通常是批量大小，然后将所有后续维度（通道、高度、宽度等）合并（或展平）成一个单一的维度
        这里是(batch_size, 512,1,1)被转成(batch_size,512)'''
        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(args['channel_scales'][-1], 128), nn.ReLU(),
            nn.Linear(128, args["num_aux_output"])
        )
        
        sincos_time_vector = build_sincos_pos_embed(args["frames"], args["embed_dim"])
        # 将时间位置向量注册为模型的buffer，而不是一个可训练的Parameter
        self.register_buffer('sincos_time_embed', sincos_time_vector)

    def forward(self, x):
        # 原始输入形状: (B, T, C, H, W)  (B=批量大小, T=帧数, C=通道数, H=高度, W=宽度)
        print(x.shape)
        B, T, C, H, W = x.shape

        # 将 (B, T, C, H, W) -> (B * T, C, H, W)
        '''
        将时间和批次维度“压平” (Flatten/Reshape)
        让ResNet一次性处理所有序列中的所有帧，如同一个超大的batch'''
        first_input = x.view(B * T, C, H, W)

        # 并行化步骤, GPU会并行处理这 B*T 张图片。
        '''
        一次性通过特征提取器 (Single Forward Pass)
        resnet_main_feat 的形状会是 (B * T, feat_dim)
        resnet_aux_pred 的形状会是 (B * T, 6)  (假设6D姿态)'''
        x = F.relu(self.bn1(self.conv1(first_input)))
        x = self.maxpool(x)

        # 通过所有动态创建的残差阶段
        x = self.stages(x)
        aux_output = self.aux_head(x) # 最后一个阶段出来直接去辅助头
        main_features = self.avgpool(x) # 最后一个阶段残差块出来经过平均池化形成主特征向量
        main_features = torch.flatten(main_features, 1)

        # 恢复时间和批次维度，将输出变回序列格式，(B * T, feat_dim) -> (B, T, feat_dim)
        main_features = main_features.view(B, T, -1) # -1 会自动推断为 feat_dim
        time_pos_embed = self.sincos_time_embed[:, :T, :]
        main_features_with_time_pos = main_features + time_pos_embed

        # 整理辅助任务的预测序列，将 (B * T, 6) -> (B, T, 6)
        aux_output = aux_output.view(B, T, -1) # -1 会自动推断为 6
        
        return main_features, main_features_with_time_pos, aux_output # 返回主特征和辅助头的显式输出

class VisionTransformer(nn.Module):
    """
    用于1:1替换ResNet的Transformer模块.
    这一模块输入单张图片，并输出特征向量和一个辅助输出头
    """
    def __init__(self, args):
        """
        Args: args:字典，包含以下内容：
                img_size (tuple): 输入图像大小 (H, W).
                patch_size (int): 一个patch边长.
                input_channels (int): 输入图像颜色通道.
                num_aux_outputs (int): 辅助输出头维度.
                embed_dim (int): transformer内部维度(d_model).
                depth (int): transformer编码器层数.
                num_heads (int): 注意力头数.
                mlp_ratio (float): FFN隐藏层大小比例因数，hidden_dimension = embed_dim * mlp_ratio
                dropout (float): Dropout比例.
        """
        super().__init__()
        self.embed_dim = args["embed_dim"]
 
        # 图像嵌入
        '''
        输入：一张 (3, 224, 224) 的图像。
            kernel_size=stride=patchsize (分块)：卷积核将以patch_size的大小，不重叠地在图像上滑动。
            总共会滑动(H/P)*(W/P)= HW/p^2=N次。这意味着它会依次处理N个patch。
            in_channels=3, out_channels=embed_dim(线性投射)：在N个patch位置的每一个位置上，
            卷积层都会用它的embed_dim个滤波器去处理那个patch，并输出一个embed_dim维的向量。
        输出：卷积层最终的输出张量维度是 (embed_dim, H/P, W/P)。
        这个输出【随后】会被展平和重排，变成 (HW/p^2, embed_dim)，得到Transformer 模型所期望的输入格式。
            卷积操作在现代深度学习框架（如 PyTorch, TensorFlow）和硬件（GPU）上是高度优化的。
            将“分块+线性变换”这两步操作用一个单独的、高度优化的 Conv2d 来实现可以将多次内存访问和计算合并为一次大的并行计算。   
        '''
        self.patch_embed = nn.Conv2d(args["input_channels"], args["embed_dim"], 
                                     kernel_size = args["patch_size"], stride = args["patch_size"])

        # 计算patch数
        num_patches = (args["img_size"][0] // args["patch_size"]) * (args["img_size"][1] // args["patch_size"])

        # 设置CLS Token和位置向量
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args["embed_dim"]))

        # sincos方式生成sincos位置向量
        sincos_pos_vector = build_sincos_pos_embed(num_patches + 1, args["embed_dim"])
        sincos_time_vector = build_sincos_pos_embed(args["frames"], args["embed_dim"])
        # 将时空位置向量分别注册为模型的buffer，而不是一个可训练的Parameter
        self.register_buffer('sincos_pos_embed', sincos_pos_vector)
        self.register_buffer('sincos_time_embed', sincos_time_vector)
        
        # 位置向量可学习部分，与sincos相加得到
        self.pos_embed_res = nn.Parameter(torch.zeros(1, num_patches + 1, args["embed_dim"]))
        
        self.pos_drop = nn.Dropout(p = args["dropout"])

        # 使用PyTorch定义的Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = args["embed_dim"],
            nhead = args["num_heads"],
            dim_feedforward = int(args["embed_dim"] * args["mlp_ratio"]), # FFN维度
            dropout = args["dropout"], # dropout
            activation = args["activation"] , # 或者'gelu'
            batch_first = args["batch_first"], # 设定输入和输出张量的维度顺序为(Batch, Seq, Dim)
            norm_first = args["norm_first"]   # Pre-Layer Normalization，在自注意力层和FFN之前进行层归一化，能更稳定一些
        )

        # 堆叠n层transformer
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = args["depth"])

        # 最后做一层归一化
        self.norm = nn.LayerNorm(args["embed_dim"])

        # 辅助头 
        self.aux_head = nn.Sequential(
            nn.Linear(args["embed_dim"], 128),
            nn.ReLU(),
            nn.Linear(128, args["num_aux_outputs"])
        )

    def forward(self, x):
        """
        ViT前向传播.
        Args:
            x (torch.Tensor): 输入图像序列张量(B, T, C, H, W)
        Returns:
            main_features (torch.Tensor): 形如(B, T, embed_dim)的特征张量.
            辅助输出aux_output (torch.Tensor): 形如(B, T, num_aux_outputs)的张量.
        """
        # x.shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # 空间特征提取
        x = x.view(B * T, C, H, W)
        # 嵌入: (B * T, C, H, W) -> (B * T, D, H/P, W/P)
        x = self.patch_embed(x)
        
        # 展平并重排: (B * T, D, H/P, W/P) -> (B * T, N, D), N为num_patches
        x = x.flatten(2).transpose(1, 2)

        # 增加CLS Token: (B, N, D) -> (B, N+1, D)
        cls_tokens = self.cls_token.expand(B * T, -1, -1) # 广播
        x = torch.cat((cls_tokens, x), dim=1) # 拼接

        # 与位置向量相加，位置向量可学习
        final_pos_embed = self.sincos_pos_embed + self.pos_embed_res
        x = x + final_pos_embed
        x = self.pos_drop(x)

        # transformer处理
        x = self.transformer_encoder(x)
        
        # 最后一次layernorm
        x = self.norm(x)

        # 主输出是[CLS] token, shape: (B*T, num_patches+1, embed_dim) -> (B*T, embed_dim)
        main_features = x[:, 0] # Shape: (B*T, D)

        # 两种接辅助头的方式，一种是接cls以外张量，一种是接cls，先试试接cls
        # aux_input = x[:, 1:].mean(dim=1) # Shape: (B, D)
        # aux_output = self.aux_head(aux_input)
        aux_output = self.aux_head(main_features)

        # 恢复时间维度，shape: (B*T, embed_dim) -> (B, T, embed_dim)
        main_features = main_features.view(B, T, -1)
        aux_output = aux_output.view(B, T, -1)

        # 添加时间位置编码
        # 从 buffer 中取出时间编码，并截取当前序列长度 T
        time_pos_embed = self.sincos_time_embed[:, :T, :]
        main_features_with_time_pos = main_features + time_pos_embed

        return main_features, main_features_with_time_pos, aux_output

class TemporalTransformer(nn.Module):
    """
    一个用于处理帧序列特征的Transformer，提取动态信息。
    输入: ViT处理后的帧特征序列，形状为 (B, T, D)。
    输出: 
        - main_output: 输出特征向量。
        - aux_output_dynamics: 辅助任务输出，相对速度、角速度6D信息。
    """
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args["embed_dim"]

        # 定义摘要Token，用于汇总整个时间序列的信息，生成最终的控制决策
        self.summary_token = nn.Parameter(torch.zeros(1, 1, args["embed_dim"]))

        # 时间位置编码已经在ViT的输出中被添加，不需要重复添加
        # 仍然需要一个Transformer Encoder来处理这个序列
        
        # 时间Transformer编码器层
        temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model = args["embed_dim"],
            nhead = args["num_heads"],
            dim_feedforward = int(args["embed_dim"] * args["mlp_ratio"]),
            dropout = args["dropout"],
            activation = args["activation"],
            batch_first = args["batch_first"],
            norm_first = args["norm_first"]
        )


        self.input_projection = nn.Linear(args["input_dim"], args["embed_dim"])

        # 时间Transformer编码器主体
        self.temporal_transformer_encoder = nn.TransformerEncoder(
            temporal_encoder_layer, 
            num_layers = args["depth"]
        )

        self.norm = nn.LayerNorm(args["embed_dim"])

        # 定义辅助输出头，用于预测6D动态信息（相对速度+角速度）
        self.aux_head_dynamics = nn.Sequential(
            nn.Linear(args["embed_dim"], 128),
            nn.ReLU(),
            nn.Linear(128, args["num_aux_outputs"]) # 输出辅助头维度个值
        )

    def forward(self, input, x, hidden_state=None):
        # 输入x的形状: (B, T, D)，即ViT输出的 main_features_with_time_pos
        B = x.shape[0]
        # 维度对齐
        x = self.input_projection(x)
        # 在序列的开头拼接上summary Token
        '''
        ViT中：
            为了高效并行处理多张图片，将B和T两个维度展平，伪装成一个更大的批次；
            B*T 个独立的样本的每一张都需要一个 [CLS] Token来汇总自己的空间信息
            ViT 的自注意力是在一张图片内部的patches之间进行的，目的是理解空间关系。不同帧的patches之间在此阶段完全没有交互
        Temporal Transformer中：
            目标为理解T帧特征之间的动态关系，并将整个序列的动态信息融合成一个单一的特征向量
            T不是批次的一部分，而是要处理的序列长度。这正是TransformerEncoder (当 batch_first=True 时) 所期望的输入格式。
            由于目标是为长度为T的序列生成单一摘要。因此只需要为每个batch里的样本在开头附加一个 summary_token
            TemporalTransformer的自注意力是在一个序列内部的T个帧之间进行的，目的是理解时间关系
        '''
        summary_tokens = self.summary_token.expand(B, -1, -1)
        x = torch.cat((summary_tokens, x), dim=1) # Shape: (B, T+1, D)

        # 将拼接后的序列送入时间Transformer编码器
        x = self.temporal_transformer_encoder(x)

        # 应用层归一化
        x = self.norm(x)

        # 提取不同部分的输出
        summary_token_output = x[:, 0, :]      # (B, D) -> 用于最终决策
        frame_tokens_output = x[:, 1:, :]    # (B, T, D) -> 包含了上下文信息的每帧特征

        # 计算主输出
        main_output = summary_token_output

        # 计算辅助输出 (6D动态信息)
        # avg_frame_features = frame_tokens_output.mean(dim=1) # Shape: (B, D)
        aux_output_dynamics = self.aux_head_dynamics(frame_tokens_output)

        return main_output, aux_output_dynamics, None

class GRU(nn.Module):
    """
    GRU模型。
    - GRU处理时序信息，并有辅助头预测速度/角速度。
    - 最终输出一个融合时空信息的特征向量。
    """
    def __init__(self, args):
        """
        Args:
            resnet_aux_outputs (int): ResNet辅助头输出维度 (例如: 6个位姿参数)
            gru_hidden_dim (int): GRU隐藏层维度（特征向量维度）
            gru_aux_outputs (int): GRU辅助头输出数量 (例如: 6个速度/角速度参数)
        """
        super(GRU, self).__init__()
        
        # GRU的输入维度 = 图像主特征 + 外部动态特征，暂时先只有图像
        gru_input_dim = args["input_dim"] # + external_dynamic_features
        
        # GRU处理时序输出时序信息
        '''
        input_size是输入特征的维度，即对于序列中的每个时间步，输入到 GRU 单元的数据的特征数量
        hidden_size是隐藏状态 (hidden state) 的维度。
            GRU 单元在每个时间步计算并更新一个隐藏状态，hidden_size 定义了这个隐藏状态向量的长度
        num_layers是堆叠的 GRU 层数。
            如果 num_layers > 1，那么 GRU 网络将由多个 GRU 层堆叠而成。
            第一个 GRU 层的输入是原始序列数据。随后的每个 GRU 层的输入是前一个 GRU 层的输出序列。
            这种堆叠结构可以帮助模型学习更复杂、更高层次的时间依赖关系
        batch_first是一个布尔值，用于指定输入和输出张量的维度顺序。
            batch_first=True，那么输入和输出张量的形状将是 (batch, seq_len, features)
        dropout 除最后一层之外的 GRU 层输出的 Dropout 概率。
            Dropout 是一种正则化技术，用于防止过拟合。在训练过程中，它会随机地“关闭”一部分神经元的输出。
            Dropout 只应用于堆叠 GRU 层之间的连接，而不会应用于 GRU 单元内部的循环连接。
        【关于GRU的两个门】PyTorch 会自动在内部创建实现这两个门所需的所有权重矩阵和偏置项
        '''
        self.gru = nn.GRU(
            input_size = gru_input_dim,
            hidden_size = args["gru_hidden_dim"],
            num_layers = args["layer_num"],
            batch_first = args["batch_first"],
            dropout = args["drop_out"] if args["layer_num"] > 1 else 0
        )
        
        # GRU的辅助头: 用于显式预测速度/角速度，它作用于GRU的整个输出序列，以得到每个时间步的预测
        self.gru_aux_head = nn.Sequential(
            nn.Linear(args["gru_hidden_dim"], 128),
            nn.ReLU(),
            nn.Linear(128, args["aux_out_dim"])
        )

    def forward(self, input, input_with_time, hidden_state=None):
        """
        Args:
            image_sequence (Tensor): 形状为 (Batch批量大小, Time帧数, Channels通道数, Height高度, Width宽度) 的图像序列
        
        返回:
            一个元组，包含：
            - final_feature_vector (Tensor): GRU最后的隐藏状态, 形状为 (B, H_gru)
            - resnet_aux_predictions (Tensor): ResNet的姿态预测, 形状为 (B, T, F_pose)
            - gru_aux_predictions (Tensor): GRU的速度预测, 形状为 (B, T, F_vel)
        """
        
        # GRU处理整个序列
        '''
        将 gru_inputs_sequence (形状为 (B, T, C')) 传递给 self.gru 时，PyTorch 的 nn.GRU 模块会在内部自动地、高效地循环 T 次。
        每次循环中，它会取出序列中的一个时间步 (t) 的所有批次数据 (gru_inputs_sequence[:, t, :])，并与当前的隐藏状态一起，计算出下一个时间步的隐藏状态。
        这个内部循环是高度优化的，通常通过 C++ 或 CUDA 实现，比 Python 循环要高效得多
        gru_output_sequence 是 GRU 在每个时间步的输出（通常是隐藏状态）。形状是 (B, T, gru_hidden_dim)，包含了序列中每个时间步的隐藏状态输出。
        last_hidden_state 是 GRU 最后一个时间步的隐藏状态，形状是 (num_layers * num_directions, B, gru_hidden_dim)。
        如果是单向 GRU, 则形状为 (num_layers, B, gru_hidden_dim)。
        '''
        
        gru_output_sequence, last_hidden_state = self.gru(input, hidden_state)

        # GRU的辅助头，对每一帧进行显式的速度/角速度预测，（B,T,6）
        gru_aux_predictions = self.gru_aux_head(gru_output_sequence)
        
        # 最终的融合时空特征向量 (取最后一层的最后一个时间步的隐藏状态)
        final_feature_vector = last_hidden_state[-1, :, :]
        return final_feature_vector, gru_aux_predictions, last_hidden_state


def mlp(sizes, activation, output_activation=nn.Identity):
    '''
    生成网络且允许灵活修改，但全都是全连接层，其中size可以是一串序列，每个元素都描述大小；同时j和j+1在循环中自动确保相乘时行数列数相等
    nn.Identity 意味着网络的输出层将应用恒等映射作为激活函数，即输出值与输入值完全一致，没有经过任何变换
    灵活用星号解包
    nn.Linear(a, b) 【不是一个单纯的全连接层】是 PyTorch 中的一个线性层（linear layer）的构造函数。它创建了一个将输入特征映射到输出特征的线性变换。
    nn.Linear(a, b) 接受表示输入特征的维度a和输出特征的维度b，线性层的作用是通过学习一组权重和偏置，将输入特征进行线性变换，得到输出特征。
    output = input * weight^T + bias
    其中，input 是输入特征，weight 是形状为 (b, a) 的权重矩阵，bias 是形状为 (b,) 的偏置项。^T 表示权重矩阵的转置。
    ''' 
    
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        # 还是会执行n-1次，但循环最后一次（j=n-2）时激活函数是恒等映射
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class GaussianPolicy(nn.Module):
    def __init__(self, args):
        super(GaussianPolicy, self).__init__()
        
        # 定义第一模块
        if args['first_module'] == "ResNet":
            # ResNet逐帧提取特征
            self.image_feature_extractor = ResNet(args['ResNet'])
        elif args['first_module'] == "ViT":
            # Transformer逐帧提取特征
            self.image_feature_extractor = VisionTransformer(args['ViT'])

        # 定义第二模块
        if args['second_module'] == "GRU":
            self.dynamic_feature_extractor = GRU(args['GRU'])
        elif args['second_module'] == "TempT":
            self.dynamic_feature_extractor = TemporalTransformer(args["TemporalTransformer"])
        

        MLP_dict = args['MLP']
        # 在拼接后、MLP前加入归一化层LayerNorm
        concatenated_dim = MLP_dict["input_feature_dim"] + MLP_dict["mlp_state_dim"] # feature维度+state维度
        # self.concat_norm = nn.LayerNorm(concatenated_dim)

        self.mlp_network=mlp([concatenated_dim] + list(MLP_dict["hidden_size"]), MLP_dict["activation"], MLP_dict["activation"]) #特征向量+目标位置+往期动作
        self.mu_layer = nn.Linear(MLP_dict["hidden_size"][-1], args["action_dim"])
        # 生成mu的层
        self.log_std_layer = nn.Linear(MLP_dict["hidden_size"][-1], args["action_dim"])

        # 动作缩放，这里在外部解决，避免动作相差太小
        self.action_scale = torch.FloatTensor([
                (args["scaled_max_action"] - args["scaled_min_action"]) / 2.])
        self.action_bias = torch.FloatTensor([
                (args["scaled_max_action"] + args["scaled_min_action"]) / 2.])
        
        # 打印辅助头结果
        self.print_aux_output = args['print_aux_output']

    def forward(self, img_sequence, state, hidden_state=None):

        first_main_feat, first_main_feat_with_time, first_aux_pred = self.image_feature_extractor(img_sequence)

        # 输入到动态序列模块
        features, second_aux_pred, new_hidden_state = self.dynamic_feature_extractor.forward(first_main_feat, first_main_feat_with_time, hidden_state)  # 提取特征张量
        concatenated_input = torch.cat([features, state],1) # 拼接特征张量和状态
        
        # 先进行层归一化
        # normalized_input = self.concat_norm(concatenated_input)

        # 检查normalized_input的量级
        x = self.mlp_network(concatenated_input)
        # print(f"normalized input:{normalized_input}")
        mean = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std, first_aux_pred, second_aux_pred, new_hidden_state

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
        log_prob -= torch.log(self.action_scale)  # 添加Scaling雅可比 
        # 原论文(21)式
        #log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon) #原论文中公式，但是多了个action_scale
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        # 打印辅助头输出结果
        if self.print_aux_output:
            print(f"pred distance:", resnet_output[..., 0:3])
            pred_rot_6d_flat = resnet_output[..., 3:9].reshape(-1, 6)
            R_pred_flat = six_d_to_rot_mat(pred_rot_6d_flat)
            print(f"pred attitude:", R_pred_flat)
            print(f"pred velocity:", gru_output[..., 0:3])
            print(f"pred angular:", gru_output[..., 3:6])
            print("-------------------------------------------------------")
            
        return action, log_prob, mean, std, resnet_output, gru_output, new_hidden_state # 辅助头输出分别是（B,T,9）和（B,T,6）

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

class QNetwork(nn.Module):
    """
    创建双Q网络
    Args:
        critic_args：包含全部参数的字典，需要包括状态维度、动作维度、隐藏层列表、激活函数

    Returns:
        两个Q网络分别计算的Q1、Q2 
    """
    def __init__(self, critic_args):
        super(QNetwork, self).__init__()
        #torch.manual_seed(42) #所有随机数种子都用42
        # Q1 architecture
        self.Q_network_1=mlp([critic_args["state_dim"] + critic_args["action_dim"]] + list(critic_args["hidden_size"]) + [1],
                              critic_args["activation"])
        # nn.init.uniform_(self.Q_network_2[-1].weight, -1e-3, 1e-3)

        # Q2 architecture
        self.Q_network_2=mlp([critic_args["state_dim"] + critic_args["action_dim"]] + list(critic_args["hidden_size"]) + [1],
                              critic_args["activation"])

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        x1 = self.Q_network_1(xu)
        x2 = self.Q_network_2(xu)
        
        return x1, x2

class ValueNetwork(nn.Module):
    """
    PPO需要的Value Network (Critic)，估计状态价值 V(s)。
    结构上模仿 model.py 中的 QNetwork，但输入仅为 state。
    """
    def __init__(self, args):
        super(ValueNetwork, self).__init__()
        # args 包含 state_dim, hidden_sizes, activation
        # 这里复用 model.py 中的 mlp 构建函数
        self.v_net = mlp(
            [args['state_dim']] + args['hidden_size'] + [1],
            activation=args['activation']
        )
        
    def forward(self, state):
        return self.v_net(state)