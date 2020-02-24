import torch
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
