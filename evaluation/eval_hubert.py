import os
import torch
import torchaudio
import numpy as np
from sympy import pprint
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, det_curve
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2PreTrainedModel
import torch.nn as nn
from sklearn.metrics import DetCurveDisplay

from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# 设置全局随机种子
import random

random.seed(42)

# 设置NumPy随机种子
import numpy as np

np.random.seed(42)

# 设置PyTorch确定性模式
import torch

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 设置CUDA随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

model_name = "m3hrdadfi/hubert-base-persian-speech-gender-recognition"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name).to(device)
model.eval()

# ==== 提取标签映射 ====
id2label = model.config.id2label  # {'0': 'f', '1': 'm'}
label2id = {v.lower(): int(k) for k, v in id2label.items()}
print("模型标签映射：", label2id)

# 保证 female 是 1，male 是 0 （与模型一致）
gender_mapping = {
    "female": 1,
    "male": 0
}
female_index = 0
male_index = 1


# ===== 预测单个音频性别 =====
def predict_gender(file_path):
    waveform, sr = torchaudio.load(file_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

    inputs = feature_extractor(
        waveform.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        logits = model(inputs["input_values"].to(device)).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
        pred = int(np.argmax(probs))
        prob_male = probs[male_index]
        return pred, prob_male


# ===== 评估函数 =====
def evaluate(y_true, y_pred, name="System"):
    print("y_true:", len(y_true))
    print("y_pred:", len(y_pred))
    print(f"\n🎯 [{name}] 分类报告:")
    print(classification_report(
        y_true, y_pred,
        target_names=["female", "male"],
        labels=[0, 1],  # ✅ 添加这个
        digits=4))
    print("混淆矩阵:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))  # ✅ 一致性


from sklearn.metrics import roc_curve


# def plot_det_comparison(y_true1, y_score1, y_true2, y_score2, label1, label2, save_path="det_compare.png"):
#     fpr1, fnr1, _ = det_curve(y_true1, y_score1, pos_label=1)
#     fpr2, fnr2, _ = det_curve(y_true2, y_score2, pos_label=1)
#     eer1 = fpr1[np.nanargmin(np.abs(fpr1 - fnr1))]
#     eer2 = fpr2[np.nanargmin(np.abs(fpr2 - fnr2))]
#
#     plt.figure(figsize=(8, 6))
#     plt.plot(fpr1, fnr1, label=f"{label1} (EER={eer1*100:.2f}%)", linewidth=2)
#     plt.plot(fpr2, fnr2, label=f"{label2} (EER={eer2*100:.2f}%)", linewidth=2)
#     plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
#
#     plt.xscale("log")
#     plt.yscale("log")
#
#     plt.xlabel("False Positive Rate (FAR)")
#     plt.ylabel("False Negative Rate (FRR)")
#     plt.title("DET Curve Comparison Based on HuBERT")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(save_path)
#     print(f"✅ DET 对比图已保存: {save_path}")
#     plt.close()


def plot_eer_lines(y_true1, y_score1, y_true2, y_score2, label1, label2, save_path="eer_line_comparison.png"):
    from sklearn.metrics import det_curve

    # 获取 EER
    def compute_eer(y_true, y_score):
        fpr, fnr, _ = det_curve(y_true, y_score, pos_label=1)
        eer = fpr[np.nanargmin(np.abs(fpr - fnr))]
        return eer

    eer1 = compute_eer(y_true1, y_score1)
    eer2 = compute_eer(y_true2, y_score2)

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [eer1, eer1], label=f"{label1} (EER={eer1 * 100:.2f}%)", linestyle='--', linewidth=2)
    plt.plot([0, 1], [eer2, eer2], label=f"{label2} (EER={eer2 * 100:.2f}%)", linestyle='-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k:', alpha=0.4)

    plt.xlabel("False Positive Rate (FAR)")
    plt.ylabel("False Negative Rate (FRR)")
    plt.title("DET (EER) Line Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ 简化 EER 直线图已保存: {save_path}")
    plt.close()


from sklearn.metrics import DetCurveDisplay
from sklearn.metrics import det_curve, DetCurveDisplay
import matplotlib.pyplot as plt
import numpy as np


# === 手动计算 EER ===
def compute_eer(y_true, y_score):
    fpr, fnr, _ = det_curve(y_true, y_score, pos_label=1)
    eer = fpr[np.nanargmin(np.abs(fpr - fnr))]
    return fpr, fnr, eer


def plot_det_comparison(y_true1, y_score1, y_true2, y_score2,
                        label1="Original", label2="NAC",
                        save_path="det_compare.png"):
    # === 模型 1 ===
    fpr1, fnr1, eer1 = compute_eer(y_true1, y_score1)
    disp1 = DetCurveDisplay(fpr=fpr1, fnr=fnr1,
                            estimator_name=f"{label1} (EER={eer1 * 100:.2f}%)")

    # === 模型 2 ===
    fpr2, fnr2, eer2 = compute_eer(y_true2, y_score2)
    disp2 = DetCurveDisplay(fpr=fpr2, fnr=fnr2,
                            estimator_name=f"{label2} (EER={eer2 * 100:.2f}%)")

    # === 绘图 ===
    plt.figure(figsize=(8, 6))
    disp1.plot(linewidth=2)
    disp2.plot(linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)

    # # ✅ 控制坐标轴范围，避免线贴边
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)

    plt.xlabel("False Positive Rate (FAR)")
    plt.ylabel("False Negative Rate (FRR)")
    plt.title("DET Curve Comparison (with EER)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ DET 曲线对比图保存成功: {save_path}")
    plt.close()


# ===== 读取两种数据格式 =====
def classify_from_scp(scp_path, gender, root_prefix):
    wav_paths = []
    with open(scp_path, "r") as f:
        for line in f:
            _, rel_path = line.strip().split()
            full_path = os.path.join(root_prefix, rel_path)
            if os.path.exists(full_path):
                wav_paths.append(full_path)

    return predict_all(wav_paths, gender, f"SCP_{gender}")


def classify_from_folder(folder_path, gender):
    wav_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".wav")]
    return predict_all(wav_paths, gender, f"FOLDER_{gender}")


def predict_all(wav_paths, gender, task_name=""):
    true_label = 1 if gender == "male" else 0
    y_true, y_pred, y_score = [], [], []

    for wav in tqdm(wav_paths, desc=f"预测中：{task_name}"):
        pred, score = predict_gender(wav)
        if pred in [0, 1]:
            mapped_pred = 0 if pred == 1 else 1  # female→0, male→1
            y_pred.append(mapped_pred)
            y_true.append(true_label)
            y_score.append(score)

    return y_true, y_pred, y_score


# ===== 主程序入口 =====
if __name__ == "__main__":
    # 数据路径
    root_prefix = "/medias/speech/projects/panariel/anon_augmentation/umpteenth_fucking_pipeline"

    # 原始数据集 (scp)
    female_scp = f"{root_prefix}/data/libri_test_trials_f/wav.scp"
    male_scp = f"{root_prefix}/data/libri_test_trials_m/wav.scp"
    y_f1, y_p1, y_s1 = classify_from_scp(female_scp, "female", root_prefix)
    y_m1, y_p2, y_s2 = classify_from_scp(male_scp, "male", root_prefix)

    y_true_ori = y_f1 + y_m1
    y_pred_ori = y_p1 + y_p2
    y_score_ori = y_s1 + y_s2
    evaluate(y_true_ori, y_pred_ori, "原始数据集")
    _, _, eer_ori = compute_eer(y_true_ori, y_score_ori)

    # NAC 数据集 (folder)
    female_dir_nac = f"{root_prefix}/data/libri_test_trials_f_nac/wav"
    male_dir_nac = f"{root_prefix}/data/libri_test_trials_m_nac/wav"

    female_dir_speaker_sex = f"/home/quy/Voice-Privacy-Challenge-2024/data/libri_test_trials_f_speaker_sex"
    male_dir_speaker_sex = f"/home/quy/Voice-Privacy-Challenge-2024/data/libri_test_trials_m_speaker_sex"
    female_dir_star = f"/home/quy/Voice-Privacy-Challenge-2024/data/libri_test_trials_f_stargan/libri_test_trials_f/"
    male_dir_star = f"/home/quy/Voice-Privacy-Challenge-2024/data/libri_test_trials_m_stargan/libri_test_trials_m/"

    y_f3_nac, y_p3_nac, y_s3_nac = classify_from_folder(female_dir_nac, "female")
    y_m3_nac, y_p4_nac, y_s4_nac = classify_from_folder(male_dir_nac, "male")

    y_true_nac = y_f3_nac + y_m3_nac
    y_pred_nac = y_p3_nac + y_p4_nac
    y_score_nac = y_s3_nac + y_s4_nac
    _, _, eer_nac = compute_eer(y_true_nac, y_score_nac)

    y_f3_speaker_sex, y_p3_speaker_sex, y_s3_speaker_sex = classify_from_folder(female_dir_speaker_sex, "female")
    y_m3_speaker_sex, y_p4_speaker_sex, y_s4_speaker_sex = classify_from_folder(male_dir_speaker_sex, "male")

    y_true_speaker_sex = y_f3_speaker_sex + y_m3_speaker_sex
    y_pred_speaker_sex = y_p3_speaker_sex + y_p4_speaker_sex
    y_score_speaker_sex = y_s3_speaker_sex + y_s4_speaker_sex
    _, _, eer_speaker_sex = compute_eer(y_true_speaker_sex, y_score_speaker_sex)

    y_f3_speaker_star, y_p3_speaker_star, y_s3_speaker_star = classify_from_folder(female_dir_star, "female")
    y_m3_speaker_star, y_p4_speaker_star, y_s4_speaker_star = classify_from_folder(male_dir_star, "male")

    y_true_speaker_star = y_f3_speaker_star + y_m3_speaker_star
    y_pred_speaker_star = y_p3_speaker_star + y_p4_speaker_star
    y_score_speaker_star = y_s3_speaker_star + y_s4_speaker_star
    _, _, eer_star = compute_eer(y_true_speaker_star, y_score_speaker_star)

    print("eer_ori:",eer_ori)
    print("eer_nac:",eer_nac)
    print("eer_speaker_sex",eer_speaker_sex)
    print("eer_star:",eer_star)
    evaluate(y_true_nac, y_pred_nac, "NAC 数据集")

    # DET 对比图
    plot_det_comparison(y_true_ori, y_score_ori, y_true_nac, y_score_nac,
                        label1="Original", label2="NAC",
                        save_path="det_comparison_original_vs_nac.png")
    plot_eer_lines(
        y_true_ori, y_score_ori,
        y_true_nac, y_score_nac,
        label1="Original", label2="NAC",
        save_path="eer_line_comparison.png"
    )
