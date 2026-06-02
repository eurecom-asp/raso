import torch
import numpy as np
import pickle
import random
import os
import soundfile as sf
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from model import Generator_3 as Generator, Generator_6 as F0_Converter
from gender_classfy import DeepGenderClassifier
from utils import pad_seq_to_2, quantize_f0_numpy, build_model, wavegen
from config import config as hparams
from adapter import MelAdapter, ConversionLoss
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MelAdapter(nn.Module):
    """梅尔谱特征修正适配器"""

    def __init__(self, n_mels=80):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(n_mels, 256, 5, padding=2),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.Conv1d(256, n_mels, 3, padding=1)
        )

    def forward(self, x):
        return x + 0.3 * self.conv_block(x)  # 残差连接


class ConversionLoss(nn.Module):
    """联合优化损失函数"""

    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha  # 分类损失权重
        self.beta = beta  # 重建损失权重

    def _mel_reconstruction_loss(self, converted, original):
        return F.mse_loss(converted, original)

    def _f0_trend_loss(self, conv_mel, orig_mel):
        # 基于能量峰值的伪F0趋势计算
        def get_pseudo_f0(mel):
            batch_size = mel.size(0)
            f0 = []
            for b in range(batch_size):
                _, freq_bins = torch.max(mel[b], dim=0)  # [T]
                f0.append(freq_bins.float())
            return torch.stack(f0)  # [B, T]

        f0_conv = get_pseudo_f0(conv_mel)
        f0_orig = get_pseudo_f0(orig_mel)
        return F.l1_loss(
            torch.diff(f0_conv, dim=1),
            torch.diff(f0_orig, dim=1)
        )

    def forward(self, pred_labels, true_labels, conv_mel, orig_mel):
        # 分类损失
        loss_cls = F.cross_entropy(pred_labels, true_labels)

        # 重建损失组件
        loss_mel = self._mel_reconstruction_loss(conv_mel, orig_mel)
        loss_f0 = self._f0_trend_loss(conv_mel, orig_mel)

        # 总损失
        return (
                self.alpha * loss_cls +
                self.beta * (loss_mel + 0.5 * loss_f0)
        )


class EnhancedConversionTester:
    def __init__(self,
                 adapter_path=None,
                 mel_root="dataset/spmel",
                 f0_root="dataset/f0",
                 classifier_path="classifier.pth",
                 g_model_path="generator.ckpt",
                 p_model_path="f0_converter.ckpt"):

        # 初始化基础模型
        self._load_base_models(classifier_path, g_model_path, p_model_path)

        # 初始化适配器
        self.adapter = MelAdapter().to(device)
        if adapter_path and os.path.exists(adapter_path):
            self.adapter.load_state_dict(torch.load(adapter_path))

        # 语音合成器
        self.vocoder = build_model().to(device)
        self.vocoder.load_state_dict(torch.load("vocoder.ckpt")["state_dict"])
        self.vocoder.eval()

        # 路径设置
        self.mel_root = mel_root
        self.f0_root = f0_root
        os.makedirs("converted_results", exist_ok=True)

    def _load_base_models(self, c_path, g_path, p_path):
        """加载预训练模型"""
        # 分类器
        self.classifier = DeepGenderClassifier().to(device)
        self.classifier.load_state_dict(torch.load(c_path))
        self.classifier.eval()

        # 生成器
        self.G = Generator(hparams).eval().to(device)
        g_ckpt = torch.load(g_path)
        self.G.load_state_dict(g_ckpt.get('model', g_ckpt))

        # F0转换器
        self.P = F0_Converter(hparams).eval().to(device)
        p_ckpt = torch.load(p_path)
        self.P.load_state_dict(p_ckpt.get('model', p_ckpt))

    def _load_mel_f0(self, rel_path):
        """加载梅尔谱和F0特征"""
        mel = np.load(os.path.join(self.mel_root, rel_path))
        f0 = np.load(os.path.join(self.f0_root, rel_path))
        return mel, f0

    def _convert_core(self, src_mel, tgt_f0, tgt_emb):
        """音色转换核心逻辑"""
        src_mel = torch.from_numpy(src_mel).unsqueeze(0).to(device)
        tgt_f0 = quantize_f0_numpy(tgt_f0)[0]
        tgt_f0 = torch.from_numpy(tgt_f0).unsqueeze(0).to(device).permute(0, 2, 1)

        with torch.no_grad():
            converted = self.G(
                torch.cat([src_mel, tgt_f0], dim=1).permute(0, 2, 1),
                src_mel.permute(0, 2, 1),
                tgt_emb
            )
        return converted.permute(0, 2, 1)  # [B, 80, T]

    def convert_with_adapter(self, src_info, tgt_info):
        """带适配器的完整转换流程"""
        # 加载特征
        src_mel, src_f0 = self._load_mel_f0(src_info[2])
        tgt_mel, tgt_f0 = self._load_mel_f0(tgt_info[2])

        # 原始转换
        converted = self._convert_core(src_mel, tgt_f0, torch.FloatTensor(tgt_info[1]).to(device))

        # 应用适配器
        adapted = self.adapter(converted)

        # 合成语音
        waveform = wavegen(self.vocoder, adapted.squeeze(0).cpu().numpy().T)
        sf.write(f"converted_results/{src_info[0]}_adapted.wav", waveform, 16000)

        return adapted

    def train_adapter(self, metadata_path, epochs=50, batch_size=32):
        """训练适配器"""
        # 数据准备
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        # 构建训练对
        train_pairs = []
        for i in range(len(metadata) - 1):
            src, tgt = metadata[i], metadata[i + 1]
            if src[4] != tgt[4]:  # 确保性别不同
                train_pairs.append((src, tgt))

        # 优化设置
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=1e-4)
        criterion = ConversionLoss(alpha=0.7, beta=0.3)

        # 训练循环
        self.adapter.train()
        for epoch in range(epochs):
            random.shuffle(train_pairs)
            total_loss = 0

            for i in tqdm(range(0, len(train_pairs), batch_size)):
                batch = train_pairs[i:i + batch_size]
                orig_list, conv_list, label_list = [], [], []

                for src, tgt in batch:
                    try:
                        # 生成转换梅尔谱
                        conv_mel = self._convert_core(
                            *self._load_mel_f0(src[2]),
                            torch.FloatTensor(tgt[1]).to(device)
                        )
                        # 加载原始梅尔谱
                        orig_mel, _ = self._load_mel_f0(src[2])
                        orig_mel = torch.from_numpy(orig_mel).to(device)

                        orig_list.append(orig_mel)
                        conv_list.append(conv_mel)
                        label_list.append(tgt[4])
                    except Exception as e:
                        continue

                if len(orig_list) == 0:
                    continue

                # 转换为Tensor
                orig_tensor = torch.stack(orig_list)
                conv_tensor = torch.stack(conv_list)
                labels = torch.LongTensor(label_list).to(device)

                # 前向计算
                adapted = self.adapter(conv_tensor)
                preds = self.classifier(adapted[:, :80, :])

                # 损失计算
                loss = criterion(preds, labels, adapted, orig_tensor)
                total_loss += loss.item()

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), 1.0)
                optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs} | Avg Loss: {total_loss / len(train_pairs):.4f}")

        # 保存适配器
        torch.save(self.adapter.state_dict(), "trained_adapter.pth")

    def evaluate(self, metadata_path):
        """评估转换效果"""
        # ... [与原测试类相似的评估逻辑，增加适配器调用] ...


if __name__ == "__main__":
    tester = EnhancedConversionTester(
        mel_root="path/to/mel",
        f0_root="path/to/f0",
        classifier_path="path/to/classifier.pth",
        g_model_path="path/to/generator.ckpt",
        p_model_path="path/to/f0_converter.ckpt"
    )

    print("===== 开始训练适配器 =====")
    tester.train_adapter(
        metadata_path="path/to/train_metadata.pkl",
        epochs=50,
        batch_size=16
    )
