import json
from shutil import copy
from pathlib import Path
from collections import namedtuple
from utils.mapping_utils import load_binf2phone


ModelParams = namedtuple('ModelParams',
                         ['n_mfcc',
                          'remove_zeroth_coef',
                          'sample_rate',
                          'n_fft',
                          'hop_length',
                          'use_deltas',
                          'normalize_features',
                          'embedding_dim',
                          'n_convolutions',
                          'kernel_size',
                          'stride',
                          'n_head',
                          'num_encoder_layers',
                          'num_decoder_layers',
                          'dim_feedforward',
                          'dropout',
                          'positional_encoding',
                          'max_src_len',
                          'max_tgt_len',
                          'binf_targets'])
TrainParams = namedtuple('TrainParams',
                         ['epochs',
                          'batch_size',
                          'learning_rate',
                          'weight_decay',
                          'clip_grad_thresh',
                          'val_step',
                          'val_batch_size'
                          ])
MODEL_CONFIG_NAME = 'model_config.json'
TRAIN_CONFIG_NAME = 'train_config.json'


def read_config_json(model_path, config_name):
    config_path = Path(model_path) / config_name
    if not config_path.exists():
        # copy default config to the model directory
        defaut_config_path = Path(__file__).parents[1] / 'config' / config_name
        if not Path(model_path).exists():
            Path(model_path).mkdir(parents=True)
        copy(str(defaut_config_path), str(model_path))
    with open(config_path) as fid:
        params_dict = json.load(fid)
    return params_dict


def read_model_config(model_path):
    params_dict = read_config_json(model_path, MODEL_CONFIG_NAME)
    return ModelParams(**params_dict)


def read_train_config(model_path):
    params_dict = read_config_json(model_path, TRAIN_CONFIG_NAME)
    return TrainParams(**params_dict)


def read_binf_mapping(model_path, vocab=None):
    mapping_name = 'binf_map.csv'
    mapping_path = Path(model_path) / mapping_name
    if not mapping_path.exists():
        default_mapping_path = Path(__file__).parents[1] / 'config' / mapping_name
        if not Path(model_path).exists():
            Path(model_path).mkdir(parents=True)
        copy(str(default_mapping_path), str(model_path))
    mapping = load_binf2phone(mapping_path, vocab)
    return mapping
