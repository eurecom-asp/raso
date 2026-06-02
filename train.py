from distutils.command.config import config
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# from model_mel_two_decoder_add_classfier import *  # 包含 VoiceConversionSystem 所需模块，比如 GenderEncoder、NeutralEncoder、DeepDecoder 等
from model import *
from data_loader import get_data_loader
from vocoder_config import config as hparams
from torch.utils.tensorboard import SummaryWriter  # 引入 TensorBoard
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import soundfile
import pickle
from synthesis import build_model, wavegen

import os
import json
import torch

from hifi_gan.env import AttrDict
from hifi_gan.models import Generator
import numpy as np

h = None
device = None
MAX_WAV_VALUE = 32768.0


class OrthogonalLoss(nn.Module):
    """空间位置级正交损失"""

    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, feat1, feat2):
        """
        输入形状:
        - feat1: [B,1,H,W]
        - feat2: [B,1,H,W]
        """
        # 展平空间维度
        feat1 = feat1.flatten(start_dim=2).to(self.device)  # [B,1,H*W]
        feat2 = feat2.flatten(start_dim=2).to(self.device)  # [B,1,H*W]

        # 计算逐位置余弦相似度
        similarity = F.cosine_similarity(feat1, feat2, dim=1)  # [B,H*W]
        return torch.mean(similarity ** 2)  # 相似度趋近于0


def load_checkpoint(filepath, device):
    """加载检查点文件"""
    print("Loading checkpoint from:", filepath)
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("成功加载检查点")
    return checkpoint_dict


def generate(mel, device):
    checkpoint_file = "hifi_gan/VCTK_V3"
    model_check = "hifi_gan/VCTK_V3/generator_v3"
    config_path = os.path.join(checkpoint_file, 'config.json')
    with open(config_path) as f:
        config = json.load(f)
    h = AttrDict(config)
    generator = Generator(h).to(device)
    checkpoint = load_checkpoint(model_check, device)
    generator.load_state_dict(checkpoint['generator'])
    generator.remove_weight_norm()
    generator.eval()

    # mel = load_mel("dataset/spmel-hifigan/p323/p323_277.npy").to(device)
    generated_audio = generator(mel)
    audio_tensor = generated_audio.squeeze()  # [mel_length]
    audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # 转为 [1, 1, mel_length] 格式（符合 TensorBoard 要求）
    return audio_tensor


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(log_dir="runs/exp_mel_add_gate_small_hifigan")  # 初始化 TensorBoard 日志记录器

    # 初始化模型
    model = VoiceConversionSystem().to(device)
    print("模型结构 (使用 __repr__ 方法):\n", model)

    # 创建保存模型的文件夹
    save_dir = "saved_models_mel_add_gate_small_hifigan"
    os.makedirs(save_dir, exist_ok=True)  # 确保文件夹存在

    # summary(model, input_size=(2, 192, 80))
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=200)

    # 损失函数
    criterion = {
        'recon': torch.nn.MSELoss().to(device),
        'gender': torch.nn.BCEWithLogitsLoss().to(device),
        'adv': torch.nn.BCEWithLogitsLoss().to(device),
        'ortho': OrthogonalLoss(device),
    }

    # 数据加载器
    train_loader = get_data_loader(hparams, mode='train')
    test_loader = get_data_loader(hparams, mode='test')
    # 输出数据集大小
    print(f"训练集总样本数：{len(train_loader.dataset)}")
    print(f"测试集总样本数：{len(test_loader.dataset)}")

    warmup_epochs = 50  # 👈 明确定义
    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        # model.gender_classifier.eval()  # 👈 强制分类器保持评估模式

        epoch_loss = 0.0
        num_batches = 0

        progress = min(epoch / warmup_epochs, 1.0)  # 👈 使用定义好的变量
        model.neutral_enc.grl.set_alpha(2.0 * progress)

        for batch_idx, (mel, emb_org, f0_org, len_org, age, gender, accent) in enumerate(train_loader):
            mel = mel.to(device)  # [B, 1, 80, T]
            mel = mel.permute(0, 2, 1).unsqueeze(1)  # 转换为 [B, 1, 80, 192]
            gender = gender.float().to(device)

            # 前向传播
            recon, g_feat, n_feat, gender_pred, adv_pred, gender_pred_g, gender_pred_n = model(mel)
            # 计算各项损失
            loss_recon = F.l1_loss(recon, mel)
            loss_gender = criterion['gender'](gender_pred_g.squeeze(), gender)
            # 对抗损失（强制 n_feat 无法分类）
            loss_adv = criterion['adv'](gender_pred_n.squeeze(), 0.5 * torch.ones_like(gender))
            # 此处计算 ortho_loss 时，我们假设模型内部有 encoder 接口供提取特征
            g_feat = g_feat.to(device)
            n_feat = n_feat.to(device)
            loss_ortho = criterion['ortho'](g_feat, n_feat)

            ortho_weight = 1.0 if epoch < 50 else 2.0  # 后期加强解耦
            # 总损失（权重可根据实际需求调整）
            total = (
                    2.0 * loss_recon +  # 基础重建
                    1.0 * loss_gender +  # 正常性别分类
                    1.0 * loss_adv +  # 强对抗
                    ortho_weight * loss_ortho  # 强调解耦
            )

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        # 记录 TensorBoard：记录每个 epoch 的各项损失（这里只记录最后一个 batch 的损失作为示例）
        writer.add_scalar("Loss/Recon", loss_recon.item(), epoch)
        writer.add_scalar("Loss/Gender", loss_gender.item(), epoch)
        writer.add_scalar("Loss/Adv", loss_adv.item(), epoch)
        writer.add_scalar("Loss/Ortho", loss_ortho.item(), epoch)
        writer.add_scalar("Loss/Total", total.item(), epoch)

        # 保存模型或者验证（这里每10个 epoch 保存一次模型）
        if epoch % 20 == 0:
            val_loss = validate(model, test_loader, criterion, device, writer, epoch)
            print(f"Validation Loss: {val_loss:.4f}")

            save_path = os.path.join(save_dir, f"model_epoch{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        # 最终测试
    test(model, test_loader, device, writer, "final_test")
    writer.close()


def validate(model, val_loader, criterion, device, writer, epoch):
    model.eval()
    val_loss = 0.0
    num_batches = 0

    criterion = {
        'recon': torch.nn.MSELoss().to(device),
        'gender': torch.nn.BCEWithLogitsLoss().to(device),
        'adv': torch.nn.BCEWithLogitsLoss().to(device),
        'ortho': OrthogonalLoss(device),
    }

    with torch.no_grad():
        for batch_idx, (mel, emb_org, f0_org, len_org, age, gender, accent) in enumerate(val_loader):

            mel = mel.to(device)  # [B, 1, 80, T]
            mel = mel.permute(0, 2, 1).unsqueeze(1)  # 转换为 [B, 1, 80, 192]
            gender = gender.float().to(device)

            # 前向传播
            recon, g_feat, n_feat, gender_pred, adv_pred, gender_pred_g, gender_pred_n = model(mel)
            # 计算各项损失
            loss_recon = F.l1_loss(recon, mel)
            loss_gender = criterion['gender'](gender_pred_g.squeeze(), gender)
            # 对抗损失（强制 n_feat 无法分类）
            loss_adv = criterion['adv'](gender_pred_n.squeeze(), 0.5 * torch.ones_like(gender))
            # 此处计算 ortho_loss 时，我们假设模型内部有 encoder 接口供提取特征
            g_feat = g_feat.to(device)
            n_feat = n_feat.to(device)
            loss_ortho = criterion['ortho'](g_feat, n_feat)

            ortho_weight = 1.0 if epoch < 100 else 2.0  # 后期加强解耦
            # 总损失（权重可根据实际需求调整）
            total = (
                    2.0 * loss_recon +  # 基础重建
                    1.0 * loss_gender +  # 正常性别分类
                    1.0 * loss_adv +  # 强对抗
                    ortho_weight * loss_ortho  # 强调解耦
            )
            val_loss += total.item()
            num_batches += 1

            # 每隔一定步长保存验证样本
            if batch_idx == 0:
                log_mel_images(writer, "Validation", mel, recon, g_feat, n_feat, epoch)
                save_audio_samples(writer, "Validation", mel, recon, g_feat, n_feat, len_org, device, batch_idx)

    avg_loss = val_loss / num_batches
    writer.add_scalar("Val/Loss", avg_loss, epoch)
    return avg_loss


def test(model, test_loader, device, writer, tag):
    model.eval()
    model.load_state_dict(torch.load("saved_models_mel_add_gate_small_hifigan/model_epoch180.pth", map_location=device))
    # model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    with torch.no_grad():
        for batch_idx, (mel, _, _, len_org, _, gender, _) in enumerate(test_loader):  # 正确解包 len_org
            mel = mel.to(device).permute(0, 2, 1).unsqueeze(1)
            gender = gender.float().to(device)
            alpha = 1.0
            recon, g_feat, n_feat, gender_pred, adv_pred, gender_pred_n, gender_pred_g = model(mel)

            # 生成音频样本
            if batch_idx < 5:  # 保存前5个样本
                save_audio_samples(writer, f"Test_{1}", mel, recon, g_feat, n_feat, len_org, device, batch_idx)
                log_mel_images(writer, f"Test_{1}", mel, recon, g_feat, n_feat, batch_idx)

    # 保存最终模型
    torch.save(model.state_dict(), f"model_final.pth")


#
# def log_mel_images(writer, tag, original_mel, reconstructed_mel, step):
#     # 可视化梅尔谱对比
#     fig, axes = plt.subplots(1, 2, figsize=(12, 4))
#     axes[0].imshow(original_mel[0, 0].cpu().numpy(), origin='lower')
#     axes[0].set_title('Original')
#     axes[1].imshow(reconstructed_mel[0, 0].cpu().numpy(), origin='lower')
#     axes[1].set_title('Reconstructed')
#     writer.add_figure(f"{tag}/Mel_Spectrogram", fig, step)


def log_mel_images(writer, tag, original_mel, reconstructed_mel, g_feat, n_feat, step):
    # 创建2行2列的子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 原始梅尔谱
    axes[0, 0].imshow(original_mel[0, 0].cpu().numpy(), aspect='auto', origin='lower')
    axes[0, 0].set_title("Original Mel")

    # 重构后的梅尔谱
    axes[0, 1].imshow(reconstructed_mel[0, 0].cpu().numpy(), aspect='auto', origin='lower')
    axes[0, 1].set_title("Reconstructed Mel")

    # g_feat 对应的图像（假设它也是类似的二维数据）
    axes[1, 0].imshow(g_feat[0, 0].cpu().numpy(), aspect='auto', origin='lower')
    axes[1, 0].set_title("g_feat")

    # n_feat 对应的图像（同样假设为二维数据）
    axes[1, 1].imshow(n_feat[0, 0].cpu().numpy(), aspect='auto', origin='lower')
    axes[1, 1].set_title("n_feat")

    plt.tight_layout()
    writer.add_figure(f"{tag}/Mel_Spectrograms", fig, step)


def save_audio_samples(writer, tag, original_mel, reconstructed_mel, g_feat, n_feat, lengths, device, step):
    # 将梅尔谱转换为音频
    save_dir = 'result_models_mel_add_gate_small_hifigan'  # 确保路径与错误信息中的一致
    os.makedirs(save_dir, exist_ok=True)  # 关键修复：创建目录

    valid_length = lengths[step].item() if isinstance(lengths, torch.Tensor) else lengths[0]
    mel_single = original_mel[step, :, :, :valid_length].squeeze()  # [80, valid_length]
    recon_single = reconstructed_mel[step, :, :, :valid_length].squeeze()
    g_feat_single = g_feat[step, :, :, :valid_length].squeeze()
    n_feat_single = n_feat[step, :, :, :valid_length].squeeze()
    # mel_single = original_mel[0].squeeze(0)
    # recon_single = reconstructed_mel[0].squeeze(0)
    # g_feat_single = g_feat[0].squeeze(0)
    # n_feat_single = n_feat[0].squeeze(0)
    # print("mel_single", mel_single.shape)
    # print("recon_single", recon_single.shape)
    # print("g_feat_single", g_feat_single.shape)
    # print("n_feat_single", n_feat_single.shape)
    # 构造保存路径，并打印路径信息
    file_path1 = os.path.join(save_dir, f"{step}.wav")
    file_path2 = os.path.join(save_dir, f"{step}_recon.wav")
    file_path3 = os.path.join(save_dir, f"{step}_g_feat.wav")
    file_path4 = os.path.join(save_dir, f"{step}_n_feat.wav")

    waveform1 = generate(mel_single, device)
    waveform1 = waveform1.detach().cpu().squeeze().numpy()  # 确保形状为[T]
    # waveform1 = wavegen(model1, c=mel_single)
    soundfile.write(file_path1, waveform1, samplerate=22050)
    waveform2 = generate(recon_single, device)
    waveform2 = waveform2.detach().cpu().squeeze().numpy()  # 确保形状为[T]
    # waveform2 = wavegen(model1, c=recon_single)
    soundfile.write(file_path2, waveform2, samplerate=22050)
    waveform3 = generate(g_feat_single, device)
    waveform3 = waveform3.detach().cpu().squeeze().numpy()  # 确保形状为[T]
    # waveform3 = wavegen(model1, c=g_feat_single)
    soundfile.write(file_path3, waveform3, samplerate=22050)
    waveform4 = generate(n_feat_single, device)
    waveform4 = waveform4.detach().cpu().squeeze().numpy()  # 确保形状为[T]
    # waveform4 = wavegen(model1, c=n_feat_single)
    soundfile.write(file_path4, waveform4, samplerate=22050)

    # 保存音频到TensorBoard
    writer.add_audio(f"{tag}/Original_Audio", waveform1, step, sample_rate=hparams.sample_rate)
    writer.add_audio(f"{tag}/Reconstructed_Audio", waveform2, step, sample_rate=hparams.sample_rate)
    writer.add_audio(f"{tag}/g_feat_Audio", waveform3, step, sample_rate=hparams.sample_rate)
    writer.add_audio(f"{tag}/n_feat_Audio", waveform4, step, sample_rate=hparams.sample_rate)


if __name__ == "__main__":
    train()
