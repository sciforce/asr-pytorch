import torch
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
from torch.utils.tensorboard.writer import SummaryWriter

from utils.config_utils import read_model_config, read_train_config, read_binf_mapping
from model.asr_model import ASRTransformerModel
from utils.dataset_utils import get_loader, load_dataset
from utils.logger import get_logger
from utils.ipa_encoder import IPAEncoder
from train.losses import get_binf_loss, get_loss
from model.layers import IPA2BinfMapper

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
    load_optimizer_state = True
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except RuntimeError:
        model.load_state_dict({k: v for k, v in checkpoint['state_dict'].items()
                               if 'embedding_layer' not in k and 'projection_layer' not in k},
                              strict=False)
        logger.debug('Skipped embedding and projection layers due to vocabulary size mismatch.')
        load_optimizer_state = False
    if load_optimizer_state:
        optimizer.load_state_dict(checkpoint['optimizer'])
    global_step = checkpoint['global_step']
    return model, optimizer, global_step


def validate(model, val_loader, device, binf_mapper=None):
    model.eval()
    losses = []
    loss_fn = get_loss
    if binf_mapper is not None:
        loss_fn = get_binf_loss
    for batch in val_loader:
        # TODO: Calculate edit distance instead of crossentropy at validation
        x, x_lengths, targets, target_lengths = [x.to(device) for x in batch]
        if binf_mapper is not None:
            targets = binf_mapper(targets)
        outputs = model(x, x_lengths, targets, target_lengths)
        loss = loss_fn(outputs, targets, target_lengths)
        losses.append(loss.item())
    loss = torch.Tensor(losses).mean()
    model.train()
    return loss


def get_model(model_params, n_outputs, device, binf_map):
    model = ASRTransformerModel(model_params, n_outputs, binf_map).to(device)
    return model


def create_summary_writer(model_dir):
    dir_name = str(Path(model_dir) / f'logs-{datetime.now().strftime("%Y%m%d-%H%M%S")}')
    Path(dir_name).mkdir()
    return SummaryWriter(dir_name)


def do_train(train_data, val_data, device, vocab,
             checkpoint_dir, start_checkpoint):
    model_params = read_model_config(checkpoint_dir)
    train_params = read_train_config(checkpoint_dir)
    summary_writer = create_summary_writer(checkpoint_dir)
    n_outputs = len(vocab)
    binf_map = None
    if model_params.binf_targets:
        binf_map = read_binf_mapping(checkpoint_dir, vocab).to(device)
        n_outputs = binf_map.size(0)
    train_loader = get_loader(train_data, model_params.sample_rate, train_params.batch_size,
                              True, model_params.max_src_len, model_params.max_tgt_len)
    val_loader = get_loader(val_data, model_params.sample_rate, train_params.val_batch_size,
                            False, model_params.max_src_len, model_params.max_tgt_len)
    model = get_model(model_params, n_outputs, device, binf_map)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=train_params.learning_rate,
                                 weight_decay=train_params.weight_decay)
    global_step, epoch = 0, 0

    if start_checkpoint is not None:
        model, optimizer, global_step = load_checkpoint(model, optimizer,
                                                        start_checkpoint)
        global_step += 1
        epoch = max(0, int(global_step // len(train_loader)))

    binf_mapper = None
    loss_fn = get_loss
    if model_params.binf_targets:
        binf_mapper = IPA2BinfMapper(binf_map)
        loss_fn = get_binf_loss
    val_loss = None

    for _ in range(train_params.epochs):
        logger.debug(f'Epoch {epoch}')
        pbar = tqdm(train_loader)
        for batch in pbar:
            optimizer.zero_grad()
            x, x_lengths, targets, target_lengths = [x.to(device) for x in batch]
            if binf_mapper is not None:
                targets = binf_mapper(targets)
            outputs = model(x, x_lengths, targets, target_lengths)
            loss = loss_fn(outputs, targets, target_lengths)
            if binf_mapper is not None:
                emb_loss = 0.01 * torch.norm(model.embedding_layer.linear_layer.weight, p=1) / binf_map.size(0)
                loss += emb_loss
            loss.backward()
            summary_writer.add_scalar('train/loss', loss.item(), global_step=global_step)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=train_params.clip_grad_thresh)
            optimizer.step()
            if global_step % train_params.val_step == 0 and global_step > 0:
                val_loss = validate(model, val_loader, device, binf_mapper=binf_mapper)
                summary_writer.add_scalar('val/loss', val_loss.item(), global_step=global_step)
                save_checkpoint(model, optimizer, global_step,
                                str(Path(checkpoint_dir) / f'checkpoint_{global_step}'))
            pbar.set_description('Step {}: loss {:2.4f}, val_loss {:2.4f}'
                                 .format(global_step, loss.item(), val_loss or 0.0))
            global_step += 1
        epoch += 1
    save_checkpoint(model, optimizer, global_step,
                    str(Path(checkpoint_dir) / f'checkpoint_{global_step}'))


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
    do_train(train_data, val_data, device, encoder.vocab,
             args.model_dir, args.start_checkpoint)
