import json
from shutil import copy
from pathlib import Path
from collections import namedtuple


ModelParams = namedtuple('ModelParams',
                         ['n_mfcc',
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
                          'dim_feedforward',
                          'dropout',
                          'positional_encoding'])
MODEL_CONFIG_NAME = 'model_config.json'


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
