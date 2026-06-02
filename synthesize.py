# coding: utf-8
"""
Synthesis waveform from trained WaveNet.

Modified from https://github.com/r9y9/wavenet_vocoder
"""

import torch
from tqdm import tqdm
import librosa
from config import config as hparams
from wavenet_vocoder import builder

torch.set_num_threads(4)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")


def build_model():
    
    model = getattr(builder, hparams.builder)(
        out_channels=hparams.out_channels,
        layers=hparams.layers,
        stacks=hparams.stacks,
        residual_channels=hparams.residual_channels,
        gate_channels=hparams.gate_channels,
        skip_out_channels=hparams.skip_out_channels,
        cin_channels=hparams.cin_channels,
        gin_channels=hparams.gin_channels,
        weight_normalization=hparams.weight_normalization,
        n_speakers=hparams.n_speakers,
        dropout=hparams.dropout,
        kernel_size=hparams.kernel_size,
        upsample_conditional_features=hparams.upsample_conditional_features,
        upsample_scales=hparams.upsample_scales,
        freq_axis_kernel_size=hparams.freq_axis_kernel_size,
        scalar_input=True,
        legacy=hparams.legacy,
    )
    return model


def wavegen(model, c=None, tqdm=tqdm):
    """Generate waveform samples by WaveNet."""
    # 如果 c 存在，获取其所在的 device，否则默认使用 CPU
    device = c.device if c is not None else torch.device("cpu")

    # 设置默认 CUDA 设备（仅在使用 GPU 时有效）
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # 将模型移动到该设备上，并进入评估模式
    model = model.to(device)
    model.eval()
    model.make_generation_fast_()

    # 获取 c 的形状参数，计算生成音频长度
    Tc = c.shape[0]
    upsample_factor = hparams.hop_size
    length = Tc * upsample_factor

    # 对 c 进行转置、类型转换，并转移到 device 上
    c = c.T.float().unsqueeze(0).to(device)
    # 创建初始输入张量，并转移到 device 上
    initial_input = torch.zeros(1, 1, 1).fill_(0.0).to(device)

    print("c所在设备:", c.device)
    print("initial_input所在设备:", initial_input.device)

    with torch.no_grad():
        y_hat = model.incremental_forward(
            initial_input, c=c, g=None, T=length, tqdm=tqdm,
            softmax=True, quantize=True, log_scale_min=hparams.log_scale_min
        )

    y_hat = y_hat.view(-1).cpu().data.numpy()
    return y_hat
