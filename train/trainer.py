import torch
from pathlib import Path
from tqdm.auto import tqdm

from utils.config_utils import read_model_config, read_train_config
from model.asr_model import ASRTransformerModel
from utils.dataset_utils import SpeechDataset, get_collate_fn, load_dataset
from utils.logger import get_logger

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
    optimizer.load_state_dict(checkpoint['state_dict'])
    global_step = checkpoint['global_step']
    return model, optimizer, global_step


def validate(model, val_loader, device, loss_fn):
    model.eval()
    losses = []
    for batch in val_loader:
        # TODO: Calculate edit distance instead of crossentropy at validation
        x, x_lengths, targets, target_lengths = [x.to(device) for x in batch]
        outputs = model(x, x_lengths, targets, target_lengths)
        losses.append(loss_fn(outputs, targets).item())
    loss = torch.cat(losses).mean()
    model.train()
    return loss


def get_model(model_params, n_outputs, device):
    model = ASRTransformerModel(model_params, n_outputs).to(device)
    return model


def do_train(train_loader, val_loader, device, model_params, train_params, n_outputs,
             checkpoint_dir, start_checkpoint):
    model = get_model(model_params, n_outputs, device)
    model.train()
    # TODO: add loss for binary features
    loss_fn = torch.nn.CrossEntropyLoss()

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
            loss = loss_fn(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=train_params.clip_grad_thresh)
            optimizer.step()
            if global_step % train_params.val_step == 0 and global_step > 0:
                val_loss = validate(model, val_loader, device, loss_fn)
                save_checkpoint(model, optimizer, global_step,
                                str(Path(checkpoint_dir) / f'checkpoint_{global_step}'))
            pbar.set_description('Step {}: loss {:2.4f}, val_loss {:2.4f}'
                                 .format(global_step, loss.item(), val_loss))
            global_step += 1
        epoch += 1
