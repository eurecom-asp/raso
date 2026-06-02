import torch
import numpy as np
import matplotlib.pyplot as plt
from model_mel_two_decoder_1 import VoiceConversionSystem
from data_loader_mel import get_data_loader
from config import config as hparams
from transformers import SpeechT5HifiGan
import os
import soundfile as sf


def test_model(checkpoint_path, output_dir="test_results"):
    # 设备配置
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # 创建保存目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "audios"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "spectrograms"), exist_ok=True)

    # 初始化模型并加载检查点
    model = VoiceConversionSystem().to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    print(f"成功加载模型检查点：{checkpoint_path}")

    # 初始化声码器
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    vocoder.eval()

    # 获取测试数据加载器
    test_loader = get_data_loader(hparams, mode='test')

    # 存储评估指标
    total_loss = 0.0
    mse_loss = torch.nn.MSELoss()

    with torch.no_grad():
        for batch_idx, (mel, _, _, _, _, gender, _) in enumerate(test_loader):
            # 数据预处理
            print("输入到模型之前的 Mel 谱形状:", mel.shape)  # 应为 [B, 1, 80, 192]x

            mel = mel.to(device).permute(0, 2, 1)  # [B, 1, 80, 192]
            # 检查形状是否正确
            print("输入到模型的 Mel 谱形状:", mel.shape)  # 应为 [B, 1, 80, 192]x
            # 模型推理
            recon_mel, _, _, _, _ = model(mel)

            # 计算重建损失
            loss = mse_loss(recon_mel, mel)
            total_loss += loss.item()

            # 处理每个样本
            for i in range(mel.size(0)):
                # 保存原始和重建的梅尔频谱图
                plot_spectrogram_comparison(
                    original=mel[i].squeeze().cpu().numpy(),
                    reconstructed=recon_mel[i].squeeze().cpu().numpy(),
                    save_path=os.path.join(output_dir, "spectrograms", f"sample_{batch_idx}_{i}.png")
                )

                # 生成并保存音频
                save_audio_sample(
                    original_mel=mel[i],
                    reconstructed_mel=recon_mel[i],
                    vocoder=vocoder,
                    save_dir=os.path.join(output_dir, "audios"),
                    sample_idx=f"{batch_idx}_{i}",
                    device=device
                )

    # 打印平均重建误差
    avg_loss = total_loss / len(test_loader)
    print(f"\n测试完成！平均MSE损失：{avg_loss:.4f}")
    print(f"结果保存在目录：{output_dir}")


def plot_spectrogram_comparison(original, reconstructed, save_path):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.imshow(original, aspect='auto', origin='lower')
    plt.title("Original Mel-Spectrogram")
    plt.colorbar()

    plt.subplot(2, 1, 2)
    plt.imshow(reconstructed, aspect='auto', origin='lower')
    plt.title("Reconstructed Mel-Spectrogram")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_audio_sample(original_mel, reconstructed_mel, vocoder, save_dir, sample_idx, device):
    def process_mel(mel):
        # 调整维度顺序适配声码器
        return mel.transpose(1, 2).unsqueeze(1)  # [1, 1, T, 80]

    # 处理原始音频
    orig_audio = vocoder(process_mel(original_mel)).squeeze().cpu().numpy()
    sf.write(os.path.join(save_dir, f"orig_{sample_idx}.wav"), orig_audio, hparams.sample_rate)

    # 处理重建音频
    recon_audio = vocoder(process_mel(reconstructed_mel)).squeeze().cpu().numpy()
    sf.write(os.path.join(save_dir, f"recon_{sample_idx}.wav"), recon_audio, hparams.sample_rate)


if __name__ == "__main__":
    # 使用示例
    test_model(
        checkpoint_path="saved_models_mel_two_de_1/model_epoch20.pth",  # 替换为实际模型路径
        output_dir="test_results"
    )