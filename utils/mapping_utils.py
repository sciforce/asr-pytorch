import torch
import numpy as np
import pandas as pd
from utils.ipa_encoder import SOS, EOS, SOS_ID, EOS_ID
from utils.ipa_utils import IPAError


def load_binf2phone(filename, vocab=None):
    binf2phone = pd.read_csv(filename, index_col=0)
    binf2phone.insert(SOS_ID, SOS, 0)
    binf2phone.insert(EOS_ID, EOS, 0)

    bottom_df = pd.DataFrame(np.zeros([2, binf2phone.shape[1]]),
                             columns=binf2phone.columns, index=[SOS, EOS])
    binf2phone = pd.concat((binf2phone, bottom_df))
    binf2phone.loc[binf2phone.index==SOS, SOS] = 1
    binf2phone.loc[binf2phone.index==EOS, EOS] = 1
    if vocab is not None:
        # Leave only phonemes from the vocabluary
        binf2phone = binf2phone[vocab.keys()]
    return torch.Tensor(binf2phone.to_numpy())


def ipa2binf(ipa, binf2phone, try_merge_diphtongs=False):
    binf = np.empty((len(ipa), len(binf2phone.index)), np.float32)
    for k, phone in enumerate(ipa):
        if phone in binf2phone.columns:
            binf_vec = binf2phone[phone].values
        elif len(phone) > 1 and try_merge_diphtongs:
            try:
                binf_vec = np.zeros((len(binf2phone.index)), np.int)
                for p in phone:
                    binf_vec = np.logical_or(binf_vec, binf2phone[p].values).astype(np.float32)
            except KeyError:
                raise IPAError(phone)
        else:
            raise IPAError(phone)
        binf[k, :] = binf_vec
    return binf
