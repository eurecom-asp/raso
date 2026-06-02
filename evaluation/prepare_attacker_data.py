import os
import pandas as pd
from pydub import AudioSegment
from pydub.silence import split_on_silence
from collections import defaultdict
import random

# ------------------------ 配置 ------------------------
scp_file = "/medias/speech/projects/panariel/anon_augmentation/umpteenth_fucking_pipeline/data/train-clean-360/wav.scp"
spk2gender_file = "/medias/speech/projects/panariel/anon_augmentation/umpteenth_fucking_pipeline/data/train-clean-360_nac/spk2gender"

converted_wav_root = "/home/quy/autovc-master/voice-gender-classfier-train/converted_data"
output_dir = "./Data/voicePrivacy"

target_duration_ms = 5000
min_duration_ms = 1000
random.seed(42)

# ------------------------ 第0步：读取 spk2gender ------------------------
spk2gender = {}
with open(spk2gender_file, "r") as f:
    for line in f:
        spk, gender = line.strip().split()
        spk2gender[spk] = 1 if gender == 'f' else 0

# ------------------------ 第1步：从 .scp 中提取出每个 .wav 的真实路径 ------------------------
spk2wav = defaultdict(list)

with open(scp_file, "r") as f:
    entries = f.read().split("|")
    for entry in entries:
        parts = entry.strip().split()
        for p in parts:
            if p.endswith(".flac"):
                wav_path = p.replace(".flac", ".wav")
                path_parts = wav_path.split("/")
                if len(path_parts) < 5:
                    continue
                spk_id = path_parts[3]
                if spk_id not in spk2gender:
                    continue
                abs_path = os.path.join(converted_wav_root, *path_parts[1:])  # 跳过 "corpora"
                if os.path.exists(abs_path):  # 确保文件真的存在
                    spk2wav[spk_id].append(abs_path)

# ------------------------ 第2~6步：合并、切割、保存 ------------------------
data_list = []

for spk, wav_paths in spk2wav.items():
    out_path = os.path.join(output_dir, f"p{spk}")
    os.makedirs(out_path, exist_ok=True)

    # 合并所有 .wav
    full_audio = AudioSegment.empty()
    for wav in sorted(wav_paths):
        try:
            audio = AudioSegment.from_wav(wav)
            full_audio += audio
        except Exception as e:
            print(f"[WARN] Failed to load {wav}: {e}")

    # 切割静音
    chunks = split_on_silence(full_audio,
                              min_silence_len=100,
                              silence_thresh=full_audio.dBFS - 16,
                              keep_silence=100)

    print(f"[INFO] Speaker {spk} -> {len(chunks)} chunks")

    # 合并到 5s 左右一段
    tmp_chunk = AudioSegment.empty()
    final_chunks = []
    for c in chunks:
        tmp_chunk += c
        if len(tmp_chunk) >= target_duration_ms:
            final_chunks.append(tmp_chunk)
            tmp_chunk = AudioSegment.empty()
    if len(tmp_chunk) >= min_duration_ms:
        final_chunks.append(tmp_chunk)

    label = spk2gender[spk]
    for i, chunk in enumerate(final_chunks, 1):
        if len(chunk) < min_duration_ms:
            continue
        chunk = chunk.set_frame_rate(24000).set_channels(1)
        out_file = os.path.join(out_path, f"{i}.wav")
        chunk.export(out_file, format="wav")
        data_list.append({"Path": out_file, "Label": label})
        print(f"[SAVE] {out_file}")

# ------------------------ 第7~9步：DataFrame + Train/Val 划分 ------------------------
df = pd.DataFrame(data_list)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

split_idx = int(len(df) * 0.1)
val_df = df[:split_idx]
train_df = df[split_idx:]


def save_list(df, filename):
    with open(filename, "w") as f:
        for _, row in df.iterrows():
            f.write(f"{row['Path']}|{row['Label']}\n")
    print(f"[DONE] Saved {len(df)} lines to {filename}")


save_list(train_df, os.path.join(output_dir, "train_privacy_list.txt"))
save_list(val_df, os.path.join(output_dir, "val_privacy_list.txt"))
