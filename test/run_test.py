import torch
from pathlib import Path
from tqdm.auto import tqdm
from test.edit_distance import edit_distance
from model.asr_model import ASRTransformerModel
from utils.logger import get_logger
from utils.ipa_encoder import SOS_ID, EOS_ID, IPAEncoder
from utils.config_utils import read_model_config
from utils.dataset_utils import get_loader, load_dataset

logger = get_logger('asr.train')


def run_test(test_data, device, n_outputs,
             checkpoint_path, test_batch_size, max_batches_count=None,
             encoder=None, beam_size=3):
    checkpoint_dir = Path(checkpoint_path).parents[0]
    model_params = read_model_config(checkpoint_dir)
    test_loader = get_loader(test_data, model_params.sample_rate, test_batch_size,
                             False, model_params.max_src_len, model_params.max_tgt_len)
    model = ASRTransformerModel(model_params, n_outputs).to(device)
    logger.debug(f'Loading checkpoint from {checkpoint_path}')
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    model.eval()
    pbar = tqdm(test_loader)
    distances = []
    with torch.no_grad():
        for i, batch in enumerate(pbar):
            if max_batches_count is not None and i >= max_batches_count:
                break
            x, x_lengths, targets, _ = [x.to(device) for x in batch]
            partial_targets = torch.full((x.size(0), 1), SOS_ID, device=device, dtype=torch.long)
            partial_target_lengths = torch.ones((x.size(0),), device=device, dtype=torch.long)
            if beam_size > 0:
                outputs = model.inference_beam_search(x, x_lengths, partial_targets, partial_target_lengths,
                                                    eos=EOS_ID, beam_size=beam_size)
                distances.append(edit_distance(outputs[:, 0, :], targets[:, 1:], EOS_ID).detach())
                if encoder is not None:
                    for i in range(x.size(0)):
                        for beam in range(beam_size):
                            ipa_outputs = encoder.decode(outputs[i, beam, ...].detach().cpu().numpy())
                            logger.debug(f'Beam {beam}: {ipa_outputs}')
                        ipa_targets = encoder.decode(targets[i, 1:].detach().cpu().numpy())
                        logger.debug(f'Targets: {ipa_targets}\n\n')
            else:
                outputs = model.inference(x, x_lengths, partial_targets, partial_target_lengths,
                                        eos=EOS_ID)
                distances.append(edit_distance(outputs[:, :], targets[:, 1:], EOS_ID).detach())
                if encoder is not None:
                    for i in range(x.size(0)):
                        ipa_outputs = encoder.decode(outputs[i, ...].detach().cpu().numpy())
                        ipa_targets = encoder.decode(targets[i, 1:].detach().cpu().numpy())
                        logger.debug(f'Outputs: {ipa_outputs}\nTargets: {ipa_targets}\n')
        distances = torch.cat(distances)
        return distances.mean().cpu().numpy()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to directory with CSV files.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint to test.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Test batch size')
    parser.add_argument('--subset', default='test', choices=['train', 'dev', 'test'],
                        help='Dataset on which to run test.')
    parser.add_argument('--max_batches_count', type=int, default=None,
                        help='Maximal batches count to limit the test.')
    parser.add_argument('--beam_size', type=int, default=3,
                        help='Beam size.')
    parser.add_argument('--verbose', action='store_true',
                        help='Output predictions and targets.')
    args = parser.parse_args()
    test_data = load_dataset(Path(args.data_dir), subset=args.subset)
    if torch.cuda.is_available():
        device = 'cuda'
        logger.debug('Using CUDA')
    else:
        device = 'cpu'

    encoder = IPAEncoder(args.data_dir)
    PER = run_test(test_data, device, len(encoder.vocab),
                   args.checkpoint, args.batch_size, args.max_batches_count,
                   encoder if args.verbose else None, beam_size=args.beam_size)
    logger.info(f'Average PER is {PER}. {len(test_data)} samples tested.')
