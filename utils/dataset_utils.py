import torch
import torchaudio
import librosa
import csv
from pathlib import Path
from utils.logger import get_logger


logger = get_logger('asr.train')


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, data, sample_rate, mode='librosa'):
        self.sample_rate = sample_rate
        self.data = data
        self.mode = mode

    def __getitem__(self, idx):
        filename, ids = self.data(idx)
        if self.mode == 'librosa':
            audio, _ = librosa.load(filename, self.sample_rate)
            audio = torch.Tensor(audio)
        elif self.mode == 'torchaudio':
            audio, sr = torchaudio.load(filename)
            if sr != self.sample_rate:
                audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio)
            audio.squeeze_()
        return audio, torch.Tensor(ids)


def get_collate_fn(max_len_src=None, max_len_tgt=None):
    def collate_fn(batch):
        if max_len_src is None:
            max_len_src = max(audio.size(0) for audio, _ in batch)
        if max_len_tgt is None:
            max_len_tgt = max(ids.size(0) for _, ids in batch)
        batch_size = len(batch)
        features = torch.zeros((batch_size, max_len_src), dtype=torch.float)
        targets = torch.zeros((batch_size, max_len_tgt), dtype=torch.int32)
        feature_lengths = torch.zeros((batch_size,), dtype=torch.int32)
        target_lengths = torch.zeros((batch_size,), dtype=torch.int32)
        for i, (feats, ids) in enumerate(batch):
            features[i, :feats.size(0)] = feats
            targets[i, :ids.size(0)] = ids
            feature_lengths[i] = feats.size(0)
            target_lengths[i] = ids.size(0)
        return features, feature_lengths, targets, target_lengths
    return collate_fn


def load_dataset(data_path, subset='train'):
    filename = str(Path(data_path) / f'{subset}.tsv')
    logger.info(f'Reading data from {filename}')
    dataset = []
    with open(filename, 'r') as fid:
        reader = csv.reader(fid, dialect='excel-tab')
        for row in reader:
            dataset.append(row)
    return dataset
