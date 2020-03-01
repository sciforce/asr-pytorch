import torch
import math
import torchaudio
import librosa


class MFCCLayer(torch.nn.Module):
    def __init__(self, sample_rate, n_mfcc, n_fft, hop_length, use_deltas,
                 normalize_features):
        super(MFCCLayer, self).__init__()
        self.use_deltas = use_deltas
        self.normalize_features = normalize_features
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.mfcc = torchaudio.transforms.MFCC(sample_rate, n_mfcc,
                                               melkwargs={'n_fft': n_fft,
                                                          'hop_length': hop_length})
        if self.use_deltas:
            self.deltas = torchaudio.transforms.ComputeDeltas()
        num_features = n_mfcc
        if use_deltas:
            num_features *= 3
        self.norm = torch.nn.BatchNorm1d(num_features)

    def forward(self, signals, lengths):
        # trim signals tensor for memory saving purposes if we split batch into smaller batches
        signals = signals[:, :lengths.max()]
        mel_features = self.mfcc(signals)
        device = lengths.device
        lengths_frames = librosa.samples_to_frames(lengths.cpu().numpy(),
                                                   hop_length=self.hop_length,
                                                   n_fft=self.n_fft)
        lengths_frames = torch.Tensor(lengths_frames).to(device).int()
        if self.params.use_deltas:
            delta = self.deltas(mel_features)
            delta2 = self.deltas(delta)
            mel_features = torch.cat((mel_features, delta, delta2), dim=-2)
        if self.params.normalize_features:
            mel_features = self.norm(mel_features)
        return mel_features, lengths_frames


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class InputsEncoder(torch.nn.Module):
    def __init__(self, ninputs, conv_channels, kernel_size, stride,
                 n_convolutions, dropout):
        super(InputsEncoder, self).__init__()
        self.dropout = dropout

        self.convolutions = torch.nn.ModuleList()
        self.convolutions.append(
            torch.nn.Sequential(
                ConvNorm(ninputs, conv_channels,
                         kernel_size=kernel_size, stride=stride,
                         padding=int((kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                torch.nn.BatchNorm1d(conv_channels))
        )
        for _ in range(1, n_convolutions):
            self.convolutions.append(
                torch.nn.Sequential(
                    ConvNorm(conv_channels,
                             conv_channels,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=int((kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    torch.nn.BatchNorm1d(conv_channels))
            )

    def forward(self, x, input_lengths, max_length=None):
        for conv in self.convolutions:
            x = torch.nn.functional.dropout(torch.nn.functional.relu(conv(x)),
                                            self.dropout_prob, self.training)
        x.transpose_(1, 2)   # batch x input_channels x time -> batch x time x input_channels

        conv_length = input_lengths
        total_length = max_length
        for _ in range(self.params.n_convolutions):
            conv_length = torch.div((conv_length + 2 * int((self.params.kernel_size - 1) / 2)
                                    - (self.params.kernel_size - 1) - 1), self.params.stride) + 1
            if total_length is not None:
                total_length = (total_length + 2 * int((self.params.kernel_size - 1) / 2)
                                - (self.params.kernel_size - 1) - 1) // self.params.stride + 1
        x.transpose_(1, 2)    # batch x time x input_channels -> batch x input_channels x time
        return x, conv_length, total_length

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)
