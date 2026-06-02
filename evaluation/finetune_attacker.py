import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from sklearn.model_selection import train_test_split
import torchaudio
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from collections import Counter
import torch.nn.functional as F
import matplotlib.pyplot as plt
import io
from PIL import Image
# 保存为CSV
import pandas as pd
import seaborn as sns

# ===== 配置参数 =====
class Config:
    model_name = "audeering/wav2vec2-large-robust-24-ft-age-gender"
    freeze_layers = 6
    batch_size = 16
    num_epochs = 60
    learning_rate = 1e-5
    train_ratio = 0.8
    val_ratio = 0.1
    seed = 42
    label_file = "/medias/speech/projects/panariel/anon_augmentation/umpteenth_fucking_pipeline/data/train-clean-360_nac/spk2gender"
    wav_dir = "/medias/speech/projects/panariel/anon_augmentation/umpteenth_fucking_pipeline/data/train-clean-360_nac/wav"
    grad_clip = 1.0
    best_acc = 0.0
    output_dir = "best_model_one_dataset"
    subset_ratio = 0.5  # ✅ 新增：仅使用 50% 的数据进行训练


class GenderDataset(Dataset):
    def __init__(self, file_list, gender_mapping, processor, is_train=True):
        self.processor = processor
        self.gender_mapping = gender_mapping
        self.file_list = file_list
        self.is_train = is_train
        self.max_length = 16000 * 5

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        waveform, sr = torchaudio.load(file_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

        if waveform.shape[1] < self.max_length:
            waveform = F.pad(waveform, (0, self.max_length - waveform.shape[1]))
        else:
            waveform = waveform[:, :self.max_length]

        spk_id = os.path.basename(file_path).split("-")[0]
        label = self.gender_mapping[spk_id]

        inputs = self.processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
        return {
            "input_values": inputs.input_values.squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
            "index": idx  # 添加索引信息
        }


import matplotlib.pyplot as plt
import numpy as np


def log_score_scatter(writer, step, scores, labels, title="Score Distribution (Dot Plot)"):
    scores = np.array(scores)
    labels = np.array(labels)

    fig, ax = plt.subplots(figsize=(6, 4))
    # 绘制 Female
    female_idx = np.where(labels == 0)[0]
    ax.scatter(scores[female_idx], np.random.normal(0, 0.01, size=len(female_idx)),
               alpha=0.5, label="Female", color='tab:blue', s=5)

    # 绘制 Male
    male_idx = np.where(labels == 1)[0]
    ax.scatter(scores[male_idx], np.random.normal(1, 0.01, size=len(male_idx)),
               alpha=0.5, label="Male", color='tab:orange', s=5)

    ax.set_title(title)
    ax.set_xlabel("Predicted Male Probability")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Female", "Male"])
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    # 写入 TensorBoard
    import io
    from PIL import Image
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image = np.array(image)
    writer.add_image("Validation/Score_Scatter", image, global_step=step, dataformats='HWC')
    buf.close()
    plt.close()


def freeze_layers(model, num_frozen):
    for layer in model.wav2vec2.encoder.layers[:num_frozen]:
        for param in layer.parameters():
            param.requires_grad = False


def log_score_distribution(writer, step, scores, labels, title="Probability Density"):
    scores = np.array(scores)
    labels = np.array(labels)

    plt.figure(figsize=(10, 6))

    # 分离男女样本数据
    female_scores = scores[labels == 0]
    male_scores = scores[labels == 1]

    # 使用Seaborn绘制KDE曲线
    sns.kdeplot(female_scores, label=f"Female (n={len(female_scores)})",
                color='tab:blue', linewidth=2, fill=True, alpha=0.2)
    sns.kdeplot(male_scores, label=f"Male (n={len(male_scores)})",
                color='tab:orange', linewidth=2, fill=True, alpha=0.2)

    # 设置图形属性
    plt.xlabel("Predicted Male Probability", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.title(f"{title}\n(Gaussian KDE Estimation)", fontsize=14)
    plt.xlim(-0.1, 1.1)
    plt.grid(True, linestyle='--', alpha=0.3)

    # 添加统计标注
    plt.axvline(0.5, color='red', linestyle=':', linewidth=1.5, alpha=0.8)
    plt.text(0.52, plt.ylim()[1] * 0.9, "Decision Boundary",
             color='red', rotation=90, va='center')

    # 添加分布参数
    for data, color, pos in [(female_scores, 'tab:blue', (0.05, 0.8)),
                             (male_scores, 'tab:orange', (0.05, 0.7))]:
        mean = np.mean(data)
        std = np.std(data)
        plt.text(pos[0], pos[1],
                 f"μ={mean:.2f}\nσ={std:.2f}",
                 color=color, transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))

    plt.legend(loc='upper left')

    # 保存到TensorBoard
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    image = Image.open(buf)
    writer.add_image(f"Test/{title}", np.array(image),
                     global_step=step, dataformats='HWC')
    plt.close()

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir="runs/gender_classifier_final")

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,
        hidden_dropout=0.2,
        attention_dropout=0.2,
        final_dropout=0.2
    )

    freeze_layers(model, config.freeze_layers)
    model.to(device)

    gender_mapping = {}
    with open(config.label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            spk_id, gender_str = parts[0], parts[1].lower()
            gender = 0 if gender_str == "f" else 1
            gender_mapping[spk_id] = gender

    all_files = []
    for fname in os.listdir(config.wav_dir):
        full_path = os.path.join(config.wav_dir, fname)
        if not os.path.isfile(full_path):
            continue
        spk_id = fname.split("-")[0]
        if spk_id in gender_mapping:
            all_files.append(full_path)

    # === 按 speaker ID 分组划分 ===
    spk2files = {}
    for f in all_files:
        spk_id = os.path.basename(f).split("-")[0]
        spk2files.setdefault(spk_id, []).append(f)

    all_spk_ids = sorted(spk2files.keys())
    train_ids, val_test_ids = train_test_split(all_spk_ids, test_size=0.15, random_state=config.seed)
    val_ids, test_ids = train_test_split(val_test_ids, test_size=1 / 3, random_state=config.seed)

    train_files = [f for spk in train_ids for f in spk2files[spk]]
    val_files = [f for spk in val_ids for f in spk2files[spk]]
    test_files = [f for spk in test_ids for f in spk2files[spk]]

    processor = Wav2Vec2Processor.from_pretrained(config.model_name)
    train_dataset = GenderDataset(train_files, gender_mapping, processor, is_train=True)
    val_dataset = GenderDataset(val_files, gender_mapping, processor, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size * 2, shuffle=False, num_workers=4)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate,
                            weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * config.num_epochs)

    from collections import Counter
    class_counts = Counter(gender_mapping[os.path.basename(f).split("-")[0]] for f in train_files)
    class_weights = torch.tensor([1 / class_counts[0], 1 / class_counts[1]]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_acc = 0.0
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        for batch in pbar:
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(inputs, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            correct += (torch.argmax(outputs.logits, dim=1) == labels).sum().item()
            pbar.set_postfix({"loss": loss.item()})

        val_loss = 0.0
        val_correct = 0
        val_probs = []
        val_labels = []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input_values"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(inputs, labels=labels)

                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                val_correct += (preds == labels).sum().item()

                probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
                val_probs.extend(probs)
                val_labels.extend(labels.cpu().numpy())

        val_acc = val_correct / len(val_dataset)
        train_acc = correct / len(train_dataset)
        writer.add_scalars("Loss", {"Train": total_loss / len(train_loader), "Val": val_loss / len(val_loader)}, epoch)
        writer.add_scalars("Accuracy", {"Train": train_acc, "Val": val_acc}, epoch)

        # === DET curve ===
        from sklearn.metrics import det_curve
        fpr, fnr, _ = det_curve(val_labels, val_probs)
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(fpr, fnr, label="DET")
        ax.plot(fpr[eer_idx], fnr[eer_idx], "ro", label=f"EER={eer:.4f}")
        ax.set_title("DET Curve")
        ax.set_xlabel("FPR")
        ax.set_ylabel("FNR")
        ax.grid(True)
        ax.legend()
        writer.add_figure("DET", fig, global_step=epoch)
        plt.close(fig)

        # === Score distribution ===
        fig, ax = plt.subplots()
        ax.hist([val_probs[i] for i in range(len(val_labels)) if val_labels[i] == 0], bins=30, alpha=0.5,
                label="female")
        ax.hist([val_probs[i] for i in range(len(val_labels)) if val_labels[i] == 1], bins=30, alpha=0.5, label="male")
        ax.set_title("Score Distribution")
        ax.set_xlabel("Male probability")
        ax.legend()
        writer.add_figure("Score_Distribution", fig, global_step=epoch)
        plt.close(fig)

        log_score_scatter(writer, epoch, val_labels, val_probs)  # ✅ 添加这个绘制点阵图

        # === Save best ===
        if val_acc > best_acc:
            best_acc = val_acc
            model.save_pretrained(os.path.join(config.output_dir, "best_model"))
            processor.save_pretrained(os.path.join(config.output_dir, "best_model"))
            print(f"\n✅ Best model saved at epoch {epoch + 1}, Val Acc: {val_acc:.4f}")


def test(config):
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir="runs/gender_classifier_test")

    # 加载最佳模型
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        os.path.join(config.output_dir, "best_model")
    ).to(device)
    processor = Wav2Vec2Processor.from_pretrained(os.path.join(config.output_dir, "best_model"))

    # 加载标签映射
    gender_mapping = {}
    with open(config.label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            spk_id, gender_str = parts[0], parts[1].lower()
            gender = 0 if gender_str == "f" else 1
            gender_mapping[spk_id] = gender

    # 构建测试集（与训练代码相同的划分逻辑）
    all_files = []
    for fname in os.listdir(config.wav_dir):
        full_path = os.path.join(config.wav_dir, fname)
        if not os.path.isfile(full_path):
            continue
        spk_id = fname.split("-")[0]
        if spk_id in gender_mapping:
            all_files.append(full_path)

    spk2files = {}
    for f in all_files:
        spk_id = os.path.basename(f).split("-")[0]
        spk2files.setdefault(spk_id, []).append(f)

    all_spk_ids = sorted(spk2files.keys())
    train_ids, val_test_ids = train_test_split(all_spk_ids, test_size=0.15, random_state=config.seed)
    val_ids, test_ids = train_test_split(val_test_ids, test_size=1 / 3, random_state=config.seed)
    test_files = [f for spk in test_ids for f in spk2files[spk]]

    # 创建测试数据集
    test_dataset = GenderDataset(test_files, gender_mapping, processor, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size * 2, shuffle=False, num_workers=4)

    # 测试过程
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []
    file_paths = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].cpu().numpy()
            indices = batch["index"].cpu().numpy()  # 获取索引
            outputs = model(inputs)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_probs.extend(probs)
            all_labels.extend(labels)
            all_preds.extend(preds)
            file_paths.extend([test_dataset.file_list[i] for i in indices])  # 使用索引获取正确路径

    # 计算指标
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        roc_auc_score,
        classification_report
    )

    print("\n" + "=" * 40)
    print("Test Results")
    print("=" * 40)
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(f"F1 Score: {f1_score(all_labels, all_preds):.4f}")
    print(f"AUC: {roc_auc_score(all_labels, all_probs):.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Female", "Male"]))

    log_score_scatter(writer, 0, all_probs, all_labels, title="Test Score Scatter")
    log_score_distribution(writer, 0, all_probs, all_labels, title="Test Score Distribution")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    writer.add_image("Test/Confusion_Matrix", np.array(image), dataformats="HWC")
    plt.close()

    # 保存预测结果
    results = []
    for path, prob, pred, label in zip(file_paths, all_probs, all_preds, all_labels):
        results.append({
            "file": os.path.basename(path),
            "true_label": "Female" if label == 0 else "Male",
            "pred_label": "Female" if pred == 0 else "Male",
            "male_prob": float(prob),
            "correct": int(pred == label)
        })


    df = pd.DataFrame(results)
    df.to_csv(os.path.join(config.output_dir, "test_results.csv"), index=False)
    print(f"\nSaved test results to {config.output_dir}/test_results.csv")

    # 绘制DET曲线
    from sklearn.metrics import det_curve
    fpr, fnr, _ = det_curve(all_labels, all_probs)
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    plt.figure()
    plt.plot(fpr, fnr, label=f"DET (EER={eer:.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("False Positive Rate")
    plt.ylabel("False Negative Rate")
    plt.title("DET Curve")
    plt.grid(True)
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    writer.add_image("Test/DET_Curve", np.array(image), dataformats="HWC")
    plt.close()

    writer.close()
    print("Test completed!")

# if __name__ == "__main__":
#     config = Config()
#     # 训练模型
#     train(config)
#
#     # 运行测试
#     test(config)


if __name__ == "__main__":
    config = Config()
    assert os.path.exists(config.label_file), f"标签文件 {config.label_file} 不存在"
    assert os.path.exists(config.wav_dir), f"音频目录 {config.wav_dir} 不存在"
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    # train(config)
    # 运行测试
    test(config)
