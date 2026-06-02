import os
import torch
import pickle
import numpy as np

from functools import partial
from numpy.random import uniform
from multiprocessing import Process, Manager

from torch.utils import data
from torch.utils.data.sampler import Sampler

import os
import pickle
import numpy as np
from multiprocessing import Process, Manager
from torch.utils import data


class UtteranceDataset(data.Dataset): # 类名更改为 UtteranceDataset，更明确表示是 Utterance 数据集
    """Dataset class for the Utterances dataset."""

    def __init__(self, root_dir, feat_dir, mode):
        """Initialize and preprocess the Utterances dataset."""
        self.root_dir = root_dir
        self.feature_directory = feat_dir # 变量名 feat_dir 更名为 feature_directory，更清晰
        self.mode = mode
        self.process_step = 20 # 变量名 step 更名为 process_step，更明确表示是多进程处理的步长
        self.split_index = 0  # 变量名 split 更名为 split_index, 更明确表示是切分索引，虽然当前未使用

        # 根据 mode 选择使用的 pickle 文件
        if mode == 'train':
            metadata_file_name = os.path.join(self.root_dir, "train_hifigan.pkl") # 变量名 metaname 更名为 metadata_file_name
            print(metadata_file_name)
        elif mode == 'test':
            metadata_file_name = os.path.join(self.root_dir, "test_hifigan.pkl") # 变量名 metaname 更名为 metadata_file_name, 并明确指出是 new_test.pkl
        else:
            raise ValueError("mode 必须为 'train' 或 'test'")
        metadata = pickle.load(open(metadata_file_name, "rb")) # 变量名 meta 更名为 metadata

        manager = Manager()
        metadata = manager.list(metadata) # 变量名 meta 更名为 metadata
        processed_dataset = manager.list(len(metadata) * [None])  # 变量名 dataset 更名为 processed_dataset，更明确表示是处理后的数据集，并使用更具描述性的变量名
        processes = []
        for i in range(0, len(metadata), self.process_step): # 变量名 meta 更名为 metadata, step 更名为 process_step
            p = Process(target=self._load_data, args=(metadata[i:i + self.process_step], processed_dataset, i, mode)) # 方法名 load_data 更名为 _load_data (以下划线开头表示是类内部方法), 变量名 meta 更名为 metadata, dataset 更名为 processed_dataset, step 更名为 process_step
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        # 训练 & 测试模式选择
        if mode == 'train':
            self.train_dataset = list(processed_dataset) # 变量名 dataset 更名为 processed_dataset
            self.num_utterances = len(self.train_dataset) # 变量名 num_tokens 更名为 num_utterances，更明确表示是 utterance 的数量
        elif mode == 'test':
            self.test_dataset = list(processed_dataset) # 变量名 dataset 更名为 processed_dataset
            self.num_utterances = len(self.test_dataset) # 变量名 num_tokens 更名为 num_utterances

        print(f'Finished loading {mode} dataset...')

    import traceback

    def _load_data(self, sub_metadata, processed_dataset, index_offset, mode): # 方法名 load_data 更名为 _load_data (以下划线开头表示是类内部方法), 变量名 submeta 更名为 sub_metadata, dataset 更名为 processed_dataset, idx_offset 更名为 index_offset
        for k, segment_metadata in enumerate(sub_metadata): # 变量名 sbmt 更名为 segment_metadata，更清晰表达是元数据中的一个片段
            try:
                # ==== 检查元数据字段是否完整 ====
                if len(segment_metadata) < 6:  # 假设每个样本需要至少6个字段 # 变量名 sbmt 更名为 segment_metadata
                    print(f"错误：样本字段不足，跳过处理。样本内容: {segment_metadata}") # 变量名 sbmt 更名为 segment_metadata
                    processed_dataset[index_offset + k] = None # 变量名 dataset 更名为 processed_dataset, idx_offset 更名为 index_offset
                    continue

                utterance_features = [None] * 6 # 变量名 uttrs 更名为 utterance_features，更清晰表达是 utterance 的特征
                utterance_features[0] = segment_metadata[0]  # 说话人 ID # 变量名 sbmt 更名为 segment_metadata, uttrs 更名为 utterance_features
                utterance_features[1] = segment_metadata[1]  # One-hot 编码 # 变量名 sbmt 更名为 segment_metadata, uttrs 更名为 utterance_features
                utterance_features[3] = segment_metadata[3]  # 年龄 # 变量名 sbmt 更名为 segment_metadata, uttrs 更名为 utterance_features
                utterance_features[4] = segment_metadata[4]  # 性别 # 变量名 sbmt 更名为 segment_metadata, uttrs 更名为 utterance_features
                utterance_features[5] = segment_metadata[5]  # 口音 # 变量名 sbmt 更名为 segment_metadata, uttrs 更名为 utterance_features
                # ==== 检查特征文件是否存在 ====
                spectrogram_path = os.path.join(self.root_dir, segment_metadata[2]) # 变量名 sp_path 更名为 spectrogram_path，更清晰表达是频谱路径, sbmt 更名为 segment_metadata

                new_filename = segment_metadata[2].replace("_raw.npy", ".npy")
                f0_path = os.path.join(self.feature_directory, new_filename)
                # f0_path = os.path.join(self.feature_directory, segment_metadata[2]) # 变量名 f0_path 保持不变，因为 f0 是标准缩写，feat_dir 更名为 feature_directory, sbmt 更名为 segment_metadata
                if not os.path.exists(spectrogram_path): # 变量名 sp_path 更名为 spectrogram_path
                    raise FileNotFoundError(f"频谱文件不存在: {spectrogram_path}") # 变量名 sp_path 更名为 spectrogram_path

                if not os.path.exists(f0_path):
                    raise FileNotFoundError(f"基频文件不存在: {f0_path}")
                # ==== 加载特征数据 ====
                spectrogram_feature = np.load(spectrogram_path) # 变量名 sp_tmp 更名为 spectrogram_feature, sp_path 更名为 spectrogram_path
                f0_feature = np.load(f0_path) # 变量名 f0_tmp 更名为 f0_feature

                # # ==== 根据模式切片 ====
                # if mode == 'train':
                #     spectrogram_feature = spectrogram_feature[self.split_index:, :] # 变量名 sp_tmp 更名为 spectrogram_feature, split 更名为 split_index
                #     f0_feature = f0_feature[self.split_index:] # 变量名 f0_tmp 更名为 f0_feature, split 更名为 split_index
                # else:
                #     spectrogram_feature = spectrogram_feature[:self.split_index, :] # 变量名 sp_tmp 更名为 spectrogram_feature, split 更名为 split_index
                #     f0_feature = f0_feature[:self.split_index] # 变量名 f0_tmp 更名为 f0_feature, split 更名为 split_index
                # ==== 根据模式切片 ====

                spectrogram_feature = spectrogram_feature[self.split_index:, :] # 变量名 sp_tmp 更名为 spectrogram_feature, split 更名为 split_index
                f0_feature = f0_feature[self.split_index:] # 变量名 f0_tmp 更名为 f0_feature, split 更名为 split_index

                utterance_features[2] = (spectrogram_feature, f0_feature) # 变量名 uttrs 更名为 utterance_features, sp_tmp 更名为 spectrogram_feature, f0_tmp 更名为 f0_feature
                processed_dataset[index_offset + k] = utterance_features # 变量名 dataset 更名为 processed_dataset, idx_offset 更名为 index_offset, uttrs 更名为 utterance_features

            except Exception as e:
                # ==== 打印错误信息 ====
                # print(f"处理样本失败: {segment_metadata}") # 变量名 sbmt 更名为 segment_metadata
                processed_dataset[index_offset + k] = None  # 标记为无效样本 # 变量名 dataset 更名为 processed_dataset, idx_offset 更名为 index_offset

    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        utterance_data = dataset[index] # 变量名 list_uttrs 更名为 utterance_data，更清晰表达是 utterance 的数据
        speaker_id = utterance_data[0] # 变量名 spk_id_org 更名为 speaker_id, list_uttrs 更名为 utterance_data
        speaker_embedding = utterance_data[1] # 变量名 emb_org 更名为 speaker_embedding, list_uttrs 更名为 utterance_data
        mel_spectrogram, f0 = utterance_data[2] # 变量名 melsp 更名为 mel_spectrogram, f0_org 更名为 f0, list_uttrs 更名为 utterance_data
        age = utterance_data[3]
        gender = utterance_data[4]
        accent = utterance_data[5]
        return mel_spectrogram, speaker_embedding, f0, age, gender, accent # 变量名 melsp 更名为 mel_spectrogram, emb_org 更名为 speaker_embedding, f0_org 更名为 f0

    def __len__(self):
        return self.num_utterances # 变量名 num_tokens 更名为 num_utterances

class BatchCollator(object):
    def __init__(self, hparams):
        self.min_sequence_length = hparams.min_len_seq
        self.max_sequence_length = hparams.max_len_seq
        self.max_padding_length = hparams.max_len_pad
        # 确保填充长度足够
        assert self.max_padding_length >= self.max_sequence_length, "max_len_pad必须≥max_len_seq"

    def __call__(self, batch):
        processed_batch = []
        for utterance_features in batch:
            mel, emb, f0, age, gender, accent = utterance_features

            # 检查并转置梅尔频谱
            if mel.shape[0] == 80:  # 原为(80, T)
                mel = mel.T  # 转置为(T, 80)

            # 裁剪逻辑保持不变
            if len(mel) < self.min_sequence_length:
                crop_length = len(mel)
            else:
                crop_length = np.random.randint(self.min_sequence_length,
                                              min(self.max_sequence_length, len(mel)) + 1)
            start = np.random.randint(0, len(mel) - crop_length) if len(mel) > crop_length else 0

            mel_cropped = mel[start:start + crop_length, :]
            f0_cropped = f0[start:start + crop_length]

            # 填充到固定长度
            mel_padded = np.pad(mel_cropped, ((0, self.max_padding_length - crop_length), (0, 0)), 'constant')
            f0_padded = np.pad(f0_cropped[:, np.newaxis], ((0, self.max_padding_length - crop_length), (0, 0)),
                             'constant', constant_values=-1e10)

            processed_batch.append((mel_padded, emb, f0_padded, crop_length, age, gender, accent))

        # 确保堆叠时形状一致
        mels, embs, f0s, lengths, ages, genders, accents = zip(*processed_batch)
        return (
            torch.FloatTensor(np.stack(mels)),
            torch.FloatTensor(np.stack(embs)),
            torch.FloatTensor(np.stack(f0s)),
            torch.LongTensor(np.stack(lengths)),
            torch.LongTensor(ages),
            torch.LongTensor(genders),
            torch.LongTensor(accents)
        )

class RepeatedSampler(Sampler): # 类名 MultiSampler 更名为 RepeatedSampler，更清晰表达是重复采样的 Sampler
    """Samples elements more than once in a single pass through the data."""

    def __init__(self, num_samples, n_repeats, shuffle=False):
        self.num_samples = num_samples
        self.num_repeats = n_repeats # 变量名 n_repeats 更名为 num_repeats，更清晰表达是重复次数
        self.shuffle = shuffle
        self.sample_indices_array = self._generate_sample_indices()  # 变量名 sample_idx_array 更名为 sample_indices_array，更清晰，方法名 gen_sample_array 更名为 _generate_sample_indices (以下划线开头表示是类内部方法) # ✅ 确保初始化

    def _generate_sample_indices(self): # 方法名 gen_sample_array 更名为 _generate_sample_indices (以下划线开头表示是类内部方法)
        sample_indices_array = torch.arange(self.num_samples, dtype=torch.int64).repeat(self.num_repeats) # 变量名 sample_idx_array 更名为 sample_indices_array, n_repeats 更名为 num_repeats
        if self.shuffle:
            sample_indices_array = sample_indices_array[torch.randperm(len(sample_indices_array))] # 变量名 sample_idx_array 更名为 sample_indices_array
        return sample_indices_array # 变量名 sample_idx_array 更名为 sample_indices_array

    def __iter__(self):
        return iter(self.sample_indices_array)  # 直接返回已经生成的索引数组的迭代器 # 变量名 sample_idx_array 更名为 sample_indices_array

    def __len__(self):
        return len(self.sample_indices_array)  # ✅ 确保 sample_idx_array 始终存在 # 变量名 sample_idx_array 更名为 sample_indices_array


def get_data_loader(hparams, mode): # 函数名 get_loader 更名为 get_data_loader，更清晰表达是获取数据加载器
    """Build and return a data loader."""

    dataset = UtteranceDataset(hparams.root_dir_mel, hparams.feat_dir, mode) # 类名 Utterances 更名为 UtteranceDataset

    batch_collator = BatchCollator(hparams) # 类名 MyCollator 更名为 BatchCollator

    sampler = RepeatedSampler(len(dataset), hparams.samplier, shuffle=hparams.shuffle) # 类名 MultiSampler 更名为 RepeatedSampler
    print("len", len(dataset))
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    if mode =='test':
        hparams.batch_size = 20

        # 使用默认采样器（替代原有的 RepeatedSampler）
        sampler = data.SequentialSampler(dataset)  # 顺序采样，确保不重复

        data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=hparams.batch_size,
            sampler=sampler,
            num_workers=0,
            drop_last=False,  # 重要！确保不丢弃最后一个批次
            pin_memory=True,
            collate_fn=BatchCollator(hparams)
        )
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=hparams.batch_size,
                                      sampler=sampler,
                                      num_workers=0,
                                      drop_last=True,
                                      pin_memory=True,
                                      worker_init_fn=worker_init_fn,
                                      collate_fn=batch_collator) # 变量名 my_collator 更名为 batch_collator
    return data_loader