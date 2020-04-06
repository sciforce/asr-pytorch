import torch
from model.layers import MFCCLayer, PositionalEncoding, InputsEncoder, LinearNorm
from utils.model_utils import get_mask_from_lengths


class ASRTransformerModel(torch.nn.Module):
    def __init__(self, params, n_outputs):
        """
        Args:
        params: ModelParams named tuple
        n_outputs: vocabulary size or binary features count
        """
        super(ASRTransformerModel, self).__init__()
        self.params = params
        self.n_outputs = n_outputs
        self.mfcc = MFCCLayer(params.sample_rate, params.n_mfcc,
                              params.n_fft, params.hop_length,
                              params.use_deltas, params.normalize_features)
        num_features = params.n_mfcc
        if params.use_deltas:
            num_features *= 3
        if self.params.positional_encoding:
            self.positional_encoder = PositionalEncoding(params.embedding_dim, params.dropout,
                                                         params.max_src_len)
            self.positional_encoder_tgt = PositionalEncoding(params.embedding_dim, params.dropout,
                                                             params.max_tgt_len)
        if params.n_convolutions > 0:
            self.inputs_encoder = InputsEncoder(num_features, params.embedding_dim,
                                                params.kernel_size, params.stride,
                                                params.n_convolutions, params.dropout)
            num_features = params.embedding_dim
        transformer_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=num_features,
                                                                     nhead=params.n_head,
                                                                     dim_feedforward=params.dim_feedforward,
                                                                     dropout=params.dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(transformer_encoder_layer,
                                                               params.num_encoder_layers)
        transformer_decoder_layer = torch.nn.TransformerDecoderLayer(d_model=num_features,
                                                                     nhead=params.n_head,
                                                                     dim_feedforward=params.dim_feedforward,
                                                                     dropout=params.dropout)
        self.transformer_decoder = torch.nn.TransformerDecoder(transformer_decoder_layer,
                                                               params.num_decoder_layers)
        self.embedding_layer = torch.nn.Embedding(n_outputs, num_features)
        self.projection_layer = LinearNorm(num_features, n_outputs)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _run_encoder(self, x, x_lengths):
        features, feat_lengths = self.mfcc(x, x_lengths)
        max_len_src = self.params.max_src_len
        if self.params.n_convolutions > 0:
            features, feat_lengths, max_len_src = self.inputs_encoder(features, feat_lengths, max_len_src)
        max_len_src = min(max_len_src, features.size(-1))
        # Features: batch x n_features x time -> time x batch x n_features
        features.transpose_(1, 2).transpose_(0, 1)
        src_key_padding_mask = get_mask_from_lengths(feat_lengths, max_len=max_len_src)
        src_mask = self._generate_square_subsequent_mask(features.size(0)).to(x.device)
        if self.params.positional_encoding:
            features = self.positional_encoder(features)
        memory = self.transformer_encoder(features,
                                          mask=src_mask,
                                          src_key_padding_mask=src_key_padding_mask)
        return memory, src_mask, src_key_padding_mask

    def _decoder_step(self, memory, targets, target_lengths, memory_mask, memory_key_padding_mask):
        """
        Args:
        memory: Transformer encoder's output, tensor of size [time x batch x n_features]
        targets: decoder input, tensor of size [max_target_sequence_length x batch x target_dim]
        target_lengths: lengths of target sequences, tensor of size [batch]
        memory_mask: mask for self-attention of size [time x time]
        memory_key_padding_mask: mask for sequences of different lengths of size [batch x time]
        Returns:
        outputs: tensor of size [max_target_sequence_length x batch x target_dim]
        """
        max_len_tgt = min(self.params.max_tgt_len, targets.size(0))
        if self.params.positional_encoding:
            targets = self.positional_encoder_tgt(targets)
        tgt_key_padding_mask = get_mask_from_lengths(target_lengths, max_len=max_len_tgt)
        tgt_mask = self._generate_square_subsequent_mask(targets.size(0)).to(targets.device)
        output = self.transformer_decoder(targets, memory, tgt_mask=tgt_mask,
                                          tgt_key_padding_mask=tgt_key_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask)
        # Output: time x batch x target_dim -> batch x time x target_dim
        output.transpose_(0, 1)
        output = self.projection_layer(output)
        return output

    def forward(self, x, x_lengths, targets, target_lengths):
        """
        Args:
        x: input waveforms, tensor of size [batch x max_samples_count]
        x_lengths: number of samples in the inputs, tensor of size [batch]
        targets: target embeddings, tensor of size [batch x max_target_sequence_length]
        target_lengths: lengths of target sequences, tensor of size [batch]
        Returns:
        outputs: tensor of size [batch x max_target_sequence_length x target_dim]
        """
        targets = self.embedding_layer(targets)
        # Targets: batch x time x target_dim -> time x batch x target_dim
        targets.transpose_(0, 1)
        memory, memory_mask, memory_key_padding_mask = self._run_encoder(x, x_lengths)
        output = self._decoder_step(memory, targets, target_lengths, memory_mask,
                                    memory_key_padding_mask)
        return output

    def inference(self, x, x_lengths, partial_targets, partial_target_lengths, eos=None):
        """
        Perform decoding of the input sequence
        Args:
        x: input waveforms, tensor of size [batch x max_samples_count]
        x_lengths: number of samples in the inputs, tensor of size [batch]
        partial targets: begginings of output sequence, tensor of size [batch x max_partial_target_sequence_length]
        partial_target_lengths: lengths of partial target sequences, tensor of size [batch]
        eos: end-of-sequence id
        """
        # TODO: implement beam search decoder
        # TODO: implement binary features decoder

        partial_targets = self.embedding_layer(partial_targets)

        # Partial targets: batch x time x target_dim -> time x batch x target_dim
        partial_targets.transpose_(0, 1)
        max_len = partial_target_lengths.max()
        memory, memory_mask, memory_key_padding_mask = self._run_encoder(x, x_lengths)
        for _ in range(max_len, self.params.max_tgt_len):
            output = self._decoder_step(memory, partial_targets, partial_target_lengths,
                                        memory_mask, memory_key_padding_mask)
            output = torch.argmax(output, dim=-1)
            if eos is not None and (output[-1, ...] == eos).all():
                break
            out = self.embedding_layer(output)
            partial_targets = torch.cat((partial_targets[:1, ...], out.transpose(0, 1).detach()), dim=0).detach()
        return output
