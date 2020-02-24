from collections import namedtuple

ModelParams = namedtuple('ModelParams',
                         ['n_mfcc',
                          'sample_rate',
                          'n_fft',
                          'hop_length',
                          'use_deltas',
                          'normalize_features',
                          'n_head',
                          'num_encoder_layers',
                          'dim_feedforward',
                          'dropout'
                         ])
