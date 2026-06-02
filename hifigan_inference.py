from __future__ import absolute_import, division, print_function, unicode_literals
import glob
import os
import argparse
import json
import torch
import numpy as np
from scipy.io.wavfile import write
from hifi_gan.env import AttrDict
from hifi_gan.models import Generator
from torch.utils.tensorboard import SummaryWriter  # 新增：导入 TensorBoard

h = None
device = None
MAX_WAV_VALUE = 32768.0

def load_checkpoint(filepath, device):
    """加载检查点文件"""
    print("Loading checkpoint from:", filepath)
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("成功加载检查点")
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    """自动查找目录中最新的检查点文件"""
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if not cp_list:
        return ''
    return sorted(cp_list, key=lambda x: int(x.split('_')[-1]))[-1]

def find_mel_files(root_dir):
    """递归查找所有梅尔谱文件（假设是 .npy 格式）"""
    mel_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(".npy"):
                full_path = os.path.join(dirpath, fname)
                mel_paths.append(full_path)
    return mel_paths

def load_mel(path):
    """加载梅尔谱文件（假设是 .npy 格式）"""
    mel = np.load(path)
    mel = torch.FloatTensor(mel).unsqueeze(0)  # [1, num_mels, mel_length]
    print(f"加载梅尔谱形状: {mel.shape}")
    return mel

def inference(a, device, writer):
    """推理过程并记录到 TensorBoard"""
    generator = Generator(h).to(device)

    # 加载检查点
    checkpoint = load_checkpoint(a.checkpoint_file1, device)
    generator.load_state_dict(checkpoint['generator'])
    generator.remove_weight_norm()
    generator.eval()

    # 创建输出目录
    os.makedirs(a.output_dir, exist_ok=True)

    # 查找所有梅尔谱文件
    mel_files = find_mel_files(a.input_mels_dir)
    if not mel_files:
        raise RuntimeError("在 {} 中未找到任何梅尔谱文件".format(a.input_mels_dir))

    step = 0  # 记录全局步骤数
    with torch.no_grad():
        for mel_path in mel_files:
            try:
                # 1. 加载梅尔谱
                # mel = load_mel(mel_path).to(device)
                # print("Mel Path:", mel_path)
                mel = load_mel("dataset/spmel-hifigan/p323/p323_277.npy").to(device)
                # 2. 生成音频
                generated_audio = generator(mel)
                audio_tensor = generated_audio.squeeze()  # [mel_length]
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # 转为 [1, 1, mel_length] 格式（符合 TensorBoard 要求）

                # 3. 记录到 TensorBoard
                base_name = os.path.splitext(os.path.basename(mel_path))[0]
                writer.add_audio(
                    f"Generated_Audio/{base_name}",
                    audio_tensor,  # 形状为 [batch, channels, length]，这里为 [1,1,T]
                    global_step=step,
                    sample_rate=h.sampling_rate
                )

                # 4. 转换为 numpy 并缩放（保存为 WAV 文件）
                audio = generated_audio.squeeze().cpu().numpy()
                audio = np.clip(audio, -1.0, 1.0)  # 防止超出范围
                audio = (audio * MAX_WAV_VALUE).astype('int16')

                # 5. 保存生成的音频文件
                output_path = os.path.join(a.output_dir, f"{base_name}_generated.wav")
                write(output_path, h.sampling_rate, audio)
                print(f"Generated: {output_path}")

                step += 1  # 更新步骤数

            except Exception as e:
                print(f"Error processing {mel_path}: {str(e)}")

def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mels_dir',
                        default='dataset/spmel-hifigan',
                        help='梅尔谱文件根目录')
    parser.add_argument('--output_dir',
                        default='dataset/generated_files',
                        help='生成音频的保存目录')
    parser.add_argument('--checkpoint_file1',
                        default='hifi_gan/VCTK_V3/generator_v3',
                        help='检查点文件路径')
    parser.add_argument('--checkpoint_file',
                        default='hifi_gan/VCTK_V3',
                        help='配置文件目录')
    a = parser.parse_args()

    # 加载配置文件
    config_path = os.path.join(a.checkpoint_file, 'config.json')
    try:
        with open(config_path) as f:
            config = json.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"配置文件未找到：{config_path}")

    global h
    h = AttrDict(config)

    # 设置设备
    torch.manual_seed(h.seed)
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 初始化 TensorBoard 日志记录器
    writer = SummaryWriter(log_dir="runs/inference")  # 日志保存路径

    try:
        inference(a, device, writer)  # 传递 writer 到推理函数
    finally:
        writer.close()  # 确保最后关闭 writer

if __name__ == '__main__':
    main()