import json
from pathlib import Path
from utils.logger import get_logger
from utils.ipa_utils import get_ipa


SOS = '<sos>'
EOS = '<eos>'
SOS_ID = 0
EOS_ID = 1


class IPAEncoder:
    def __init__(self, data_dir, logger=None):
        self.vocab_path = Path(data_dir) / 'vocab.json'
        if logger is not None:
            self.logger = logger
        else:
            self.logger = get_logger('root')
        self.vocab = {SOS: SOS_ID, EOS: EOS_ID}
        self.reverse_vocab = None
        self.load_vocab()

    def load_vocab(self):
        if Path(self.vocab_path).exists():
            self.logger.info(f'Reading vocabulary from {self.vocab_path}')
            with open(self.vocab_path) as fid:
                self.vocab = json.load(fid)
        else:
            self.logger.debug(f'No vocabulary found at path {self.vocab_path}')

    def save_vocab(self):
        self.logger.info(f'Saving vocabulary to {self.vocab_path}')
        with open(self.vocab_path, 'w', encoding='utf8') as fid:
            json.dump(self.vocab, fid, ensure_ascii=False)

    def encode(self, s, lang='en', plain_text=False,
               skip_lang_tags=False, **kwargs):
        if plain_text:
            ipa = list(s)
        else:
            ipa = get_ipa(s, lang, **kwargs)
        if skip_lang_tags:
            ipa = [SOS] + ipa + [EOS]
        else:
            ipa = [SOS, f'<{lang}>'] + ipa + [EOS]
        res = list()
        for phone in ipa:
            if len(phone) > 0:
                if phone not in self.vocab:
                    self.vocab[phone] = len(self.vocab)
                res.append(self.vocab[phone])
        return res

    def get_reverse_vocab(self):
        if self.reverse_vocab is None and self.vocab is not None:
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        return self.reverse_vocab

    def decode(self, ids):
        res = []
        reverse_vocab = self.get_reverse_vocab()
        for phone_id in ids:
            res.append(reverse_vocab[phone_id])
            if phone_id == EOS_ID:
                break
        return res
