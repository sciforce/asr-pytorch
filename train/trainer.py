import torch
from pathlib import Path
from tqdm.auto import tqdm

from utils.config_utils import read_model_config, read_train_config
from model.asr_model import ASRTransformerModel
from utils.dataset_utils import SpeechDataset, get_collate_fn, load_dataset
from utils.logger import get_logger
from utils.ipa_encoder import IPAEncoder
from utils.model_utils import get_mask_from_lengths

logger = get_logger('asr.train')


def save_checkpoint(model, optimizer, global_step,
                    checkpoint_path):
    logger.debug(f'Saving checkpoint at step {global_step}')
    torch.save({'global_step': global_step,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
                }, checkpoint_path)


def load_checkpoint(model, optimizer, checkpoint_path):
    logger.debug(f'Loading checkpoint from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    global_step = checkpoint['global_step']
    return model, optimizer, global_step


# TODO: add loss for binary features
def get_loss(outputs, targets, lengths, max_len=None):
    mask = get_mask_from_lengths(lengths, max_len)
    # Remove SOS character from the beginning of target sequence
    loss = torch.nn.functional.cross_entropy(outputs[:, :-1, :].transpose(1, 2),
                                             targets[:, 1:], reduction='none')
    loss = loss.masked_fill(mask[:, 1:], 0).sum(dim=-1) / (lengths - 1)
    return loss.mean()


def validate(model, val_loader, device):
    model.eval()
    losses = []
    for batch in val_loader:
        # TODO: Calculate edit distance instead of crossentropy at validation
        x, x_lengths, targets, target_lengths = [x.to(device) for x in batch]
        outputs = model(x, x_lengths, targets, target_lengths)
        loss = get_loss(outputs, targets, target_lengths)
        losses.append(loss.item())
    loss = torch.Tensor(losses).mean()
    model.train()
    return loss


def get_model(model_params, n_outputs, device):
    model = ASRTransformerModel(model_params, n_outputs).to(device)
    return model


def get_loader(data, sample_rate, batch_size, shuffle,
               max_len_src, max_len_tgt):
    dataset = SpeechDataset(data, sample_rate)
    loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle,
                                         collate_fn=get_collate_fn(max_len_src, max_len_tgt))
    return loader


def do_train(train_data, val_data, device, n_outputs,
             checkpoint_dir, start_checkpoint):
    model_params = read_model_config(checkpoint_dir)
    train_params = read_train_config(checkpoint_dir)
    train_loader = get_loader(train_data, model_params.sample_rate, train_params.batch_size,
                              True, model_params.max_src_len, model_params.max_tgt_len)
    val_loader = get_loader(val_data, model_params.sample_rate, train_params.val_batch_size,
                            False, model_params.max_src_len, model_params.max_tgt_len)
    model = get_model(model_params, n_outputs, device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=train_params.learning_rate,
                                 weight_decay=train_params.weight_decay)
    global_step, epoch = 0, 0

    if start_checkpoint is not None:
        model, optimizer, global_step = load_checkpoint(model, optimizer,
                                                        start_checkpoint)
        global_step += 1
        epoch = max(0, int(global_step // len(train_loader)))

    val_loss = None
    for _ in range(train_params.epochs):
        logger.debug(f'Epoch {epoch}')
        pbar = tqdm(train_loader)
        for batch in pbar:
            optimizer.zero_grad()
            x, x_lengths, targets, target_lengths = [x.to(device) for x in batch]
            outputs = model(x, x_lengths, targets, target_lengths)
            loss = get_loss(outputs, targets, target_lengths, model_params.max_tgt_len)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=train_params.clip_grad_thresh)
            optimizer.step()
            if global_step % train_params.val_step == 0 and global_step > 0:
                val_loss = validate(model, val_loader, device)
                save_checkpoint(model, optimizer, global_step,
                                str(Path(checkpoint_dir) / f'checkpoint_{global_step}'))
            pbar.set_description('Step {}: loss {:2.4f}, val_loss {:2.4f}'
                                 .format(global_step, loss.item(), val_loss or 0.0))
            global_step += 1
        epoch += 1


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to directory with CSV files.')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to directory with checkpoints.')
    parser.add_argument('--start_checkpoint', type=str, default=None,
                        help='Checkpoint to start training from.')
    args = parser.parse_args()
    train_data = load_dataset(Path(args.data_dir), subset='train')
    val_data = load_dataset(Path(args.data_dir), subset='dev')
    if torch.cuda.is_available():
        device = 'cuda'
        logger.debug('Using CUDA')
    else:
        device = 'cpu'

    encoder = IPAEncoder(args.data_dir)
    do_train(train_data, val_data, device, len(encoder.vocab),
             args.model_dir, args.start_checkpoint)
