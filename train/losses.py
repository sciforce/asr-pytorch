import torch
from utils.model_utils import get_mask_from_lengths


def get_loss(outputs, targets, lengths, max_len=None):
    mask = get_mask_from_lengths(lengths, max_len)
    # Remove SOS character from the beginning of target sequence
    loss = torch.nn.functional.cross_entropy(outputs[:, :-1, :].transpose(1, 2),
                                             targets[:, 1:], reduction='none')
    loss = loss.masked_fill(mask[:, 1:], 0).sum(dim=-1) / (lengths - 1)
    return loss.mean()


def get_binf_loss(outputs, targets, lengths, max_len=None):
    mask = get_mask_from_lengths(lengths, max_len).unsqueeze(-1).repeat(1, 1, targets.size(-1))
    # Remove SOS vector from the beginning of target sequence
    loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs[:, :-1, :],
                                                                targets[:, 1:, :],
                                                                reduction='none')
    loss = loss.masked_fill(mask[:, 1:, :], 0).sum(dim=-1).sum(dim=-1) / (lengths - 1)
    return loss.mean()
