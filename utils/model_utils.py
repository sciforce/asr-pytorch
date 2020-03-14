import torch


def get_mask_from_lengths(lengths, max_len=None, inv=True):
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, dtype=torch.int32, device=lengths.device)
    if inv:
        mask = ids.unsqueeze(0).expand(lengths.size(0), -1) >= lengths.unsqueeze(1)
    else:
        mask = ids.unsqueeze(0).expand(lengths.size(0), -1) < lengths.unsqueeze(1)
    return mask
