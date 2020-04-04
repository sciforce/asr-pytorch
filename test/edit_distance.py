import torch
from utils.ipa_encoder import EOS_ID


def get_seq_len(x, eos_id=EOS_ID):
    mask = x == eos_id
    inds = (torch.arange(x.size(1), device=x.device)
            .unsqueeze(0)
            .repeat(x.size(0)))
    inds.masked_fill_(mask, -1)
    return inds.max(dim=1) + 1


def edit_distance(outputs, targets, eos_id=EOS_ID,
                  w_del=1.0, w_ins=1.0, w_sub=1.0):
    max_output_len = outputs.size(-1)
    max_target_len = targets.size(-1)
    batch_size = targets.size(0)
    D = torch.zeros((batch_size, max_output_len, max_target_len),
                    device=outputs.device)
    D[:, 1:, 0] = (torch.arange(1, max_output_len)
                   .unsqueeze(1).unsqueeze(0)
                   .repeat(batch_size, 1, max_target_len)
                   .to(outputs.device)) * w_ins
    D[:, 0, 1:] = (torch.arange(1, max_target_len)
                   .unsqueeze(0).unsqueeze(0)
                   .repeat(batch_size, max_output_len, 1)
                   .to(outputs.device)) * w_del
    for i in range(1, max_output_len):
        for j in range(1, max_target_len):
            d_same = D[:, i - 1, j - 1]
            d_different = torch.stack((D[:, i - 1, j] + w_ins,
                                       D[:, i, j - 1] + w_del,
                                       D[:, i, j] + w_sub), dim=4).min(dim=-1)
            D[:, i, j] = torch.where(outputs[:, i] == targets[:, j], d_same, d_different)
    output_last_inds = get_seq_len(outputs, eos_id) - 1
    target_last_inds = get_seq_len(targets, eos_id) - 1
    batch_inds = torch.arange(0, batch_size, dtype=torch.long).to(ouputs.device)
    index_vector = torch.stack((batch_inds, output_last_inds,
                                target_last_inds), dim=-1)
    index_vector = (index_vector[:, 0] * max_output_len * max_target_len
                    + index_vector[:, 1] * max_target_len
                    + index_vector[:, 2])
    edit_distances = D.take(index_vector) / (target_last_inds + 1)
    return edit_distances
