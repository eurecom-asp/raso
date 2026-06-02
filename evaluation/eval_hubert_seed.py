import os
import random
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, det_curve
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


# 全局配置
class Config:
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    root_prefix = "/medias/speech/projects/panariel/anon_augmentation/umpteenth_fucking_pipeline"
    datasets = {
        "original": {
            "female": "data/libri_test_trials_f/wav.scp",
            "male": "data/libri_test_trials_m/wav.scp"
        },
        "NAC": {
            "female": "/medias/speech/projects/panariel/anon_augmentation/umpteenth_fucking_pipeline/data/libri_test_trials_f_nac/wav",
            "male": "/medias/speech/projects/panariel/anon_augmentation/umpteenth_fucking_pipeline/data/libri_test_trials_m_nac/wav"
        },
        "speaker_sex": {
            "female": "/home/quy/Voice-Privacy-Challenge-2024/data/libri_test_trials_f_speaker_sex",
            "male": "/home/quy/Voice-Privacy-Challenge-2024/data/libri_test_trials_m_speaker_sex"
        },
        "stargan": {
            "female": "/home/quy/Voice-Privacy-Challenge-2024/data/libri_test_trials_f_stargan/libri_test_trials_f/",
            "male": "/home/quy/Voice-Privacy-Challenge-2024/data/libri_test_trials_m_stargan/libri_test_trials_m/"
        }
    }


# 初始化模型
def init_model():
    model_name = "m3hrdadfi/hubert-base-persian-speech-gender-recognition"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(model_name).to(Config.device)
    model.eval()
    return feature_extractor, model


# 预测函数
def predict_gender(file_path, feature_extractor, model):
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
        logits = model(inputs["input_values"].to(Config.device)).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
    return int(np.argmax(probs)), probs[1]


# 数据集处理
class DataProcessor:
    @staticmethod
    def process_scp(scp_path):
        paths = []
        with open(os.path.join(Config.root_prefix, scp_path)) as f:
            for line in f:
                _, rel_path = line.strip().split()
                full_path = os.path.join(Config.root_prefix, rel_path)
                if os.path.exists(full_path):
                    paths.append(full_path)
        return paths

    @staticmethod
    def process_folder(folder_path):
        print("fold_path", folder_path)
        return [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                if f.endswith(".wav")]


# 实验运行器
class ExperimentRunner:
    def __init__(self, feature_extractor, model):
        self.feature_extractor = feature_extractor
        self.model = model
        self.results = []

    def run_dataset(self, dataset_type, gender, paths):
        y_true, y_pred, y_scores = [], [], []
        true_label = 0 if gender == "male" else 1

        for path in tqdm(paths, desc=f"Processing {dataset_type}-{gender}"):
            pred, score = predict_gender(path, self.feature_extractor, self.model)
            y_true.append(true_label)
            y_pred.append(pred)
            y_scores.append(score)

        return y_true, y_pred, y_scores

    def evaluate_dataset(self, dataset_type):
        metrics = {}
        for gender in ["female", "male"]:
            if dataset_type == "original":
                paths = DataProcessor.process_scp(Config.datasets[dataset_type][gender])
            else:
                paths = DataProcessor.process_folder(Config.datasets[dataset_type][gender])

            t, p, s = self.run_dataset(dataset_type, gender, paths)
            metrics[gender] = (t, p, s)

        # 合并结果
        y_true = metrics["female"][0] + metrics["male"][0]
        y_pred = metrics["female"][1] + metrics["male"][1]
        y_scores = metrics["female"][2] + metrics["male"][2]

        return self._compute_metrics(y_true, y_pred, y_scores)

    def _compute_metrics(self, y_true, y_pred, y_scores):
        fpr, fnr, _ = det_curve(y_true, y_scores, pos_label=1)
        eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

        report = classification_report(
            y_true, y_pred,
            target_names=["female", "male"],
            output_dict=True,
            digits=4
        )

        return {
            "eer": eer,
            "accuracy": report["accuracy"],
            "precision": report["macro avg"]["precision"],
            "recall": report["macro avg"]["recall"],
            "f1": report["macro avg"]["f1-score"],
            "cm": confusion_matrix(y_true, y_pred).tolist()
        }


# 主程序
def main():
    # 初始化模型
    feature_extractor, model = init_model()

    # 运行10次实验
    all_results = {dataset: [] for dataset in Config.datasets}

    for exp_id in range(50):
        # 生成随机种子
        seed = random.randint(0, 2 ** 32 - 1)
        print(f"\n🚀 实验 {exp_id + 1}/50 - 随机种子: {seed}")

        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 初始化实验运行器
        runner = ExperimentRunner(feature_extractor, model)

        # 运行所有数据集
        dataset_results = {}
        for dataset in Config.datasets:
            metrics = runner.evaluate_dataset(dataset)
            dataset_results[dataset] = metrics
            print(f"\n📊 {dataset} 结果:")
            print(f"EER: {metrics['eer'] * 100:.2f}%")
            print(f"准确率: {metrics['accuracy'] * 100:.2f}%")

        # 保存结果
        for dataset in Config.datasets:
            all_results[dataset].append({
                "seed": seed,
                **dataset_results[dataset]
            })

    # 打印最终报告
    print("\n🔬 最终实验结果汇总:")
    for dataset in Config.datasets:
        eers = [res["eer"] * 100 for res in all_results[dataset]]
        accs = [res["accuracy"] * 100 for res in all_results[dataset]]

        print(f"\n📈 {dataset.upper()} 统计:")
        print(f"平均 EER: {np.mean(eers):.2f}% ± {np.std(eers):.2f}%")
        print(f"平均准确率: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
        print(f"最佳 EER: {np.min(eers):.2f}%")
        print(f"最佳准确率: {np.max(accs):.2f}%")

    # 保存详细结果
    print("\n💾 详细结果已保存到内存，可通过all_results变量访问")


if __name__ == "__main__":
    # 禁用确定性算法
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    main()