import torch
import torch.nn as nn
import torch.nn.functional as F
from gender_classfy import GenderClassifier
import copy
import kornia

class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.0  # 初始值为0，后续动态更新

    def set_alpha(self, new_alpha):
        self.alpha = torch.tensor(new_alpha, requires_grad=False)

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class ChannelGate(nn.Module):
    """轻量级通道注意力（保持原设计）"""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        weights = self.fc(self.avgpool(x).view(b, c)).view(b, c, 1, 1)
        return x * weights


class ResidualBlock(nn.Module):
    """残差块（适配ConvTranspose）"""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return F.relu(x + self.conv(x))


class InformationBottleneck(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.bottleneck = nn.Sequential(
            # 空间维度压缩
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1, stride=(1, 1)),
            nn.InstanceNorm2d(in_channels * 2),
            nn.GELU(),

            # 通道维度压缩
            nn.Conv2d(in_channels * 2, in_channels // 2, 1),
            ChannelGate(in_channels // 2)  # 注意力筛选
        )

    def forward(self, x):
        return self.bottleneck(x)


# 性别编码器（保持原压缩结构）
class GenderEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=(4, 1), padding=1),
            nn.Conv2d(32, 16, 3, stride=(4, 1), padding=1),
            nn.Conv2d(16, 1, 3, stride=(5, 1), padding=1)
        )
        self.lstm = nn.LSTM(1, 16, bidirectional=True)
        self.fc = nn.Linear(32, 1)  # 关键修改：添加全连接层
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        cnn_out = self.cnn(x)
        B, C, H, W = cnn_out.shape
        compressed = F.avg_pool2d(cnn_out, (H, 1))
        lstm_in = compressed.squeeze(2).permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_in)

        pooled = lstm_out.mean(dim=1)  # [B, 32]
        logits = self.fc(pooled)  # [B, 1]
        return compressed, logits


# 中性编码器（保持原结构）
class NeutralEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 1, stride=(2, 1)),
            ResidualBlock(64),
            InformationBottleneck(64),  # ← 关键修改点
            nn.Conv2d(32, 16, (1, 1)),
            nn.Conv2d(16, 8, (1, 1))

        )
        self.grl = GradientReversalLayer()
        self.adv_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 1)  # in_channels 是输入特征图的通道数
        )

    def forward(self, x):
        features = self.encoder(x)
        adv_in = self.grl(features)
        adv_pred = self.adv_classifier(adv_in)
        return features, adv_pred


# 性别解码器（全ConvTranspose）
class GenderDecoder(nn.Module):
    def __init__(self, input_size=(1, 1)):
        super().__init__()
        # 输入维度：[B,32,1,T/4]
        self.decoder = nn.Sequential(
            # Stage 1：频率方向上采样：1 -> 20
            nn.ConvTranspose2d(
                in_channels=1, out_channels=64,
                kernel_size=(5, 1),  # 在H方向使用较大的核
                stride=(5, 1),
                # H: 1->(1-1)*20 + (kernel-2*padding+output_padding) = 20  （设 padding=(1,0)，output_padding=(1,0)）
                padding=(1, 0),
                output_padding=(1, 0)
            ),

            nn.InstanceNorm2d(64),
            nn.GELU(),

            nn.ConvTranspose2d(
                in_channels=64, out_channels=32,
                kernel_size=(3, 1),  # H方向小核保证尺寸不变，W方向核设置为6
                stride=(5, 1),  # H: 不变；W: 上采样4倍： (48-1)*4 - 2*1 +6 = 47*4 -2+6 =188 -2+6 =192
                padding=(1, 0),
                output_padding=(1, 0)
            ),

            nn.InstanceNorm2d(32),
            nn.GELU(),

            # ChannelGate(32),    # 调整通道（不改变尺寸），输出 [B,32,20,192]
            # Stage 3：频率方向上采样：20 -> 80，保持 W=192
            nn.ConvTranspose2d(
                in_channels=32, out_channels=16,
                kernel_size=(3, 1),  # H方向选择6，W方向选择3
                stride=(5, 1),  # H: (20-1)*4 - 2*1 +6 = 19*4 -2+6 =76 -2+6 =80; W: (192-1)*1 -2*1 +3 =191 -2+3 =192
                padding=(1, 0),
                output_padding=(1, 0)

            ),

            nn.InstanceNorm2d(16),
            kornia.filters.GaussianBlur2d(kernel_size=(3, 1), sigma=(1.0, 1.0)),

            # 最后将通道数调整为1
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(3, 1))
        )

    def forward(self, x):
        return self.decoder(x)


class NeutralDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            # Layer 0: [B,256,16,90] → [B,128,32,90]
            nn.ConvTranspose2d(
                in_channels=8,
                out_channels=128,
                kernel_size=(3, 1),  # 高度方向4倍上采样
                stride=(2, 1),  # 高度2倍
                padding=(1, 0),  # 控制边缘
                output_padding=(1, 0)
            ),
            nn.InstanceNorm2d(128),
            nn.GELU(),

            # Layer 1: [B,128,32,90] → [B,64,64,90]
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=(3, 1),
                stride=(1, 1),  # 高度再2倍
                padding=(1, 0),
                output_padding=(0, 0)
            ),
            nn.InstanceNorm2d(64),
            nn.GELU(),

            # Layer 2: [B,64,64,90] → [B,32,64,model_mel_add_classfier_add_two_part_add_gate.py]
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=(3, 1),
                stride=(1, 1),  # 宽度2倍
                padding=(1, 0),
            ),
            nn.InstanceNorm2d(32),
            kornia.filters.GaussianBlur2d(kernel_size=(3, 1), sigma=(1.0, 1.0)),

            # Layer 3: [B,32,64,180] → [B,1,80,192]
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=1,
                kernel_size=(3, 1),
                stride=(1, 1),  # 高宽各2倍
                padding=(1, 0),
            )
        )

    def forward(self, x):
        out = self.decoder(x)
        return out


class VoiceConversionSystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.gender_enc = GenderEncoder()
        self.neutral_enc = NeutralEncoder()
        self.gender_dec = GenderDecoder()
        self.neutral_dec = NeutralDecoder()
        self.gate_enhancement = EnhancedInformationGate(channels=1)

        # 独立的对抗分类器
        self.adv_classifier = nn.Sequential(
            # 维度适配层
            nn.Conv2d(8, 64, kernel_size=3, stride=(2, 2)),  # [B,64,7,44]
            nn.InstanceNorm2d(64),
            nn.ReLU(),

            # 特征提取
            nn.Conv2d(64, 128, 3, padding=1),  # [B,128,7,44]
            nn.MaxPool2d(2),  # [B,128,3,22]

            # 分类头部
            nn.AdaptiveAvgPool2d(1),  # [B,128,1,1]
            nn.Flatten(),
            nn.Linear(128, 1)
        )

        self.grl = GradientReversalLayer()
        # 动态融合层（适配ConvTranspose输出）
        self.fusion = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            # ResidualBlock(64),
            # ChannelGate(64),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        # ===================== 编码阶段 =====================
        # 性别编码器（需要保留性别信息）
        g_feat, g_pred = self.gender_enc(x)  # [B,32,1,T/4]
        n_feat, adv_pred = self.neutral_enc(x)  # [B,256,40,T/2]

        # ===================== 解码阶段 =====================
        g_rec = self.gender_dec(g_feat)  # [B,1,80,T]
        n_rec = self.neutral_dec(n_feat)  # [B,1,80,T]

        # ===================== 特征融合 =====================
        combined = self.gate_enhancement(g_rec, n_rec)

        # ===================== 返回结果 =====================
        return (
            combined,  # 融合后的重构结果
            g_rec,  # 性别相关重构
            n_rec,  # 中性相关重构
            g_feat,
            n_feat,
            g_pred,  # 性别编码器的预测（来自原始特征）
            adv_pred,  # 对抗预测（来自中性特征）
        )


class EnhancedInformationGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scale_g = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.scale_n = nn.Parameter(torch.ones(1, channels, 1, 1))
        # 添加空间调制卷积
        self.spatial_modulation = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

    def forward(self, feat_g, feat_n):
        # print("feat_g",feat_g.size())
        # print("feat_n",feat_n.size())
        base = feat_g * self.scale_g + feat_n * self.scale_n
        return self.spatial_modulation(base)  # 增强局部适应性


import torch
from torchsummary import summary  # 可选，用于打印模型结构


# 请确保已导入你上面定义的所有类
# 如果代码在同一文件内，可以直接使用 VoiceConversionSystem

def test_model():
    # 设置 batch 大小
    batch_size = 256

    # 构造一个随机输入张量，尺寸为 [B, 1, 80, 192]
    dummy_input = torch.randn(batch_size, 1, 80, 192)

    # 实例化模型
    model = VoiceConversionSystem()

    # 前向传播测试（使用 no_grad 节省内存）
    with torch.no_grad():
        outputs = model(dummy_input)

    # 输出结果包含：combined, g_rec, n_rec, g_pred, adv_pred, gender_pred_n, gender_pred_g
    combined, g_rec, n_rec, g_pred, adv_pred, gender_pred_g, gender_pred_n = outputs

    # 打印各输出张量的尺寸
    print("combined shape:", combined.shape)  # 预期：[B, 1, 80, 192]
    print("g_rec shape:", g_rec.shape)  # 预期：[B, 1, 80, 192]
    print("n_rec shape:", n_rec.shape)  # 预期：[B, 1, 80, 192]
    print("g_pred shape:", g_pred.shape)  # 预期：[B] 或 [B,1] 具体取决于 classifier 实现
    print("adv_pred shape:", adv_pred.shape)  # 预期：[B] 或 [B,1]
    print("gender_pred_n shape:", gender_pred_n.shape)  # 预期：[B] 或 [B,1]
    print("gender_pred_g shape:", gender_pred_g.shape)  # 预期：[B] 或 [B,1]


if __name__ == '__main__':
    test_model()
