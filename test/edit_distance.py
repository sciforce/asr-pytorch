import torch
from utils.ipa_encoder import SOS_ID, EOS_ID


def get_seq_len(x, eos_id=EOS_ID):
    mask = x != eos_id
    inds = (torch.arange(x.size(1), device=x.device)
            .unsqueeze(0)
            .repeat(x.size(0), 1))
    inds.masked_fill_(mask, x.size(1) + 1)
    return inds.min(dim=1).values


def edit_distance(outputs, targets, eos_id=EOS_ID,
                  w_del=1.0, w_ins=1.0, w_sub=1.0):
    max_output_len = outputs.size(-1)
    max_target_len = targets.size(-1)
    batch_size = targets.size(0)
    D = torch.zeros((batch_size, max_output_len, max_target_len),
                    device=outputs.device)
    D[:, 0, 0] = torch.where(outputs[:, 0] == targets[:, 0], torch.full((batch_size,), 0., device=outputs.device),
                             torch.full((batch_size,), w_sub, device=outputs.device))
    D[:, 1:, 0] = (torch.arange(1, max_output_len)
                   .unsqueeze(0)
                   .repeat(batch_size, 1)
                   .to(outputs.device)) * w_ins
    D[:, 0, 1:] = (torch.arange(1, max_target_len)
                   .unsqueeze(0)
                   .repeat(batch_size, 1)
                   .to(outputs.device)) * w_del
    for i in range(1, max_output_len):
        for j in range(1, max_target_len):
            d_same = D[:, i - 1, j - 1]
            d_different = torch.stack((D[:, i - 1, j].unsqueeze(-1) + w_ins,
                                       D[:, i, j - 1].unsqueeze(-1) + w_del,
                                       D[:, i - 1, j - 1].unsqueeze(-1) + w_sub), dim=-1).min(dim=-1).values
            D[:, i, j] = torch.where(outputs[:, i] == targets[:, j], d_same, d_different.squeeze(-1))
    output_last_inds = get_seq_len(outputs, eos_id) - 1
    target_last_inds = get_seq_len(targets, eos_id) - 1
    batch_inds = torch.arange(0, batch_size, dtype=torch.long).to(outputs.device)
    index_vector = torch.stack((batch_inds, output_last_inds,
                                target_last_inds), dim=-1)
    index_vector = (index_vector[:, 0] * max_output_len * max_target_len
                    + index_vector[:, 1] * max_target_len
                    + index_vector[:, 2])
    edit_distances = D.take(index_vector) / (target_last_inds + 1)
    return edit_distances


if __name__ == '__main__':
    x = 'Hell, ward.'
    y = 'Hello, world.'
    vocab = {'SOS': SOS_ID, 'EOS': EOS_ID}
    for c in x + y:
        if c not in vocab:
            vocab[c] = len(vocab)
    x = torch.Tensor([vocab[c] for c in x] + [EOS_ID, EOS_ID]).unsqueeze(0)
    y = torch.Tensor([vocab[c] for c in y] + [EOS_ID, EOS_ID, EOS_ID]).unsqueeze(0)
    d = edit_distance(x, y)
    print(f'edit distance is {d}')
