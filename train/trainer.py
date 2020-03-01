import torch
from pathlib import Path

from utils.config_utils import read_model_config
from model.asr_model import ASRTransformerModel
from utils.dataset_utils import SpeechDataset, get_collate_fn, load_dataset


def train(model_params, train_params, model_dir, dataset):
    # TODO
    pass
