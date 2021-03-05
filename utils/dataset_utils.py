import torch
import torchaudio
import librosa
import csv
import ast
import warnings
from multiprocessing import cpu_count
from pathlib import Path
from utils.logger import get_logger
from utils.ipa_encoder import EOS_ID

logger = get_logger('asr.train')
warnings.filterwarnings('ignore')


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, data, sample_rate, mode='torchaudio'):
        self.sample_rate = sample_rate
        self.data = data
        self.mode = mode

    def __getitem__(self, idx):
        filename, ids = self.data[idx]
        if self.mode == 'librosa':
            audio, _ = librosa.load(filename, self.sample_rate)
            audio = torch.Tensor(audio)
        elif self.mode == 'torchaudio':
            audio, sr = torchaudio.load(filename)
            if sr != self.sample_rate:
                audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio)
            audio.squeeze_()
        return audio, torch.Tensor(ast.literal_eval(ids))

    def __len__(self):
        return len(self.data)


def get_collate_fn(max_len_src=None, max_len_tgt=None):
    def collate_fn(batch):
        nonlocal max_len_src, max_len_tgt
        feature_lengths = torch.Tensor([audio.size(0) for audio, _ in batch])
        if max_len_src is None:
            max_len_src = feature_lengths.max().item()
        target_lengths = torch.Tensor([ids.size(0) for _, ids in batch])
        if max_len_tgt is None:
            max_len_tgt = target_lengths.max().item()
        features = torch.nn.utils.rnn.pad_sequence([audio for audio, _ in batch], batch_first=True)
        targets = torch.nn.utils.rnn.pad_sequence([targets for _, targets in batch], batch_first=True,
                                                  padding_value=EOS_ID).to(torch.int64)
        return features, feature_lengths, targets, target_lengths
    return collate_fn


def get_loader(data, sample_rate, batch_size, shuffle,
               max_len_src, max_len_tgt):
    dataset = SpeechDataset(data, sample_rate)
    loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle,
                                         collate_fn=get_collate_fn(max_len_src, max_len_tgt),
                                         num_workers=cpu_count())
    return loader


def load_dataset(data_path, subset='train'):
    filename = str(Path(data_path) / f'{subset}.tsv')
    logger.info(f'Reading data from {filename}')
    dataset = []
    with open(filename, 'r') as fid:
        reader = csv.reader(fid, dialect='excel-tab')
        for row in reader:
            dataset.append(row)
    return dataset
