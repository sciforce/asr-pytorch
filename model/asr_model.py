import torch
from model.layers import MFCCLayer


class ASRTransformerModel(torch.nn.Module):
    def __init__(self, params):
        self.parms = params
        self.mfcc = MFCCLayer(params.sample_rate, params.n_mfcc,
                              params.n_fft, params.hop_length,
                              params.use_deltas, params.normalize_features)
        num_features = n_mfcc
        if use_deltas:
            num_features *= 3
        self.transformer = torch.nn.Transformer(d_model=num_features,
                                                n_head=params.n_head,
                                                num_encoder_layers=params.num_encoder_layers,
                                                num_decoder_layers=params.num_decoder_layers,
                                                dim_feedforward=params.dim_feedforward,
                                                dropout=params.dropout)

    def forward(self, x, x_lengths, targets, target_lengths):
        features, feat_lengths = self.mfcc(x, x_lengths)
