import csv
import json
from argparse import ArgumentParser
from tqdm.auto import tqdm
from pathlib import Path
from utils.ipa_encoder import IPAEncoder
from utils.logger import get_logger


logger = get_logger('asr.train')


def collect_data(directory, langs_list=None, subset='train', max_num_files=None):
    """Traverses directory collecting input and target files.

    Args:
    directory: base path to extracted audio and transcripts.
    langs_list: optional list of languages to be processed. All languages are processed by default.
    subset: 'train', 'test' or 'dev'
    Returns:
    list of (media_filepath, label, language) tuples
    """
    data_files = list()
    langs = [d for d in Path(directory).iterdir()
             if d.is_dir()]
    if langs_list is not None:
        langs = [d for d in langs if d.name in langs_list]

    for lang_dir in langs:
        lang = lang_dir.name[:2]
        logger.info(f'Parsing language {lang}')
        transcript_path = lang_dir / f'{subset}.tsv'
        with open(transcript_path, 'r') as transcript_file:
            transcript_reader = csv.reader(transcript_file, dialect='excel-tab')
            # skip header
            _ = next(transcript_reader)
            for transcript_line in transcript_reader:
                _, media_name, label = transcript_line[:3]
                if '.mp3' not in media_name:
                    media_name += '.mp3'
                filename = lang_dir / 'clips' / media_name

                data_files.append((str(filename), label, lang))
                if max_num_files is not None and len(data_files) >= max_num_files:
                    return data_files
    return data_files


def encode_data(data_files, encoder, skip_lang_tags=False, **ipa_kwargs):
    """Encodes targets
    Args:
    data_files: result of `collect_data` call
    encoder: instance of IPAEncoder
    ipa_kwargs: arguments for ipa processing
    Returns:
    list of (media_filepath, encoded_ipa)
    """
    logger.info('Encoding data')
    encoded_data = list()
    for filename, label, lang in tqdm(data_files):
        ids = encoder.encode(label, lang,
                             skip_lang_tags=skip_lang_tags, **ipa_kwargs)
        encoded_data.append((filename, ids))
    return encoded_data


def serialize_encoded_data(encoded_data, output_path, subset='train'):
    filename = str(Path(output_path) / f'{subset}.tsv')
    logger.info(f'Serializing data to {filename}')
    with open(filename, 'w') as fid:
        writer = csv.writer(fid, dialect='excel-tab')
        for row in encoded_data:
            writer.writerow(row)


def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to Common Voice directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to directory where to store preprocessed dataset')
    parser.add_argument('--langs', nargs='*', type=str, default=None,
                        help='A list of languages to prepare. ISO-639 names should be used. All languages are processed by default')
    parser.add_argument('--subset', choices=['train', 'test', 'dev'],
                        help='Data subset to prepare')
    parser.add_argument('--not_remove_semi_stress', action='store_false', dest='remove_semi_stress',
                        help='Disable removing semistress symbol.')
    parser.add_argument('--not_split_diphthongs', action='store_false', dest='split_all_diphthongs',
                        help='Disable splitting diphthongs.')
    parser.add_argument('--not_split_stress_gemination', action='store_false', dest='split_stress_gemination',
                        help='Keep stress and gemination symbols sticked to a phone symbol.')
    parser.add_argument('--not_remove_lang_markers', action='store_false', dest='remove_lang_markers',
                        help='Keep language markers returned by eSpeak-ng.')
    parser.add_argument('--remove_all_stress', action='store_true',
                        help='Remove all stress marks')
    parser.add_argument('--skip_lang_tags', action='store_true',
                        help='Skip language tags.')
    parser.add_argument('--max_samples_count', type=int, default=None,
                        help='Maximal number of audio files to use. Use all by default.')
    parser.add_argument('--plain_text', action='store_true',
                        help='Use characteres as targets instead of IPA.')
    args = parser.parse_args()

    encoder = IPAEncoder(args.output_dir, logger)
    data_files = collect_data(args.dataset_dir, args.langs, args.subset, args.max_samples_count)
    encoded_data = encode_data(data_files, encoder,
                               skip_lang_tags=args.skip_lang_tags,
                               plain_text=args.plain_text,
                               remove_semi_stress=args.remove_semi_stress,
                               split_all_diphthongs=args.split_all_diphthongs,
                               split_stress_gemination=args.split_stress_gemination,
                               remove_lang_markers=args.remove_lang_markers,
                               remove_all_stress=args.remove_all_stress)
    serialize_encoded_data(encoded_data, args.output_dir, args.subset)
    encoder.save_vocab()
    with open(Path(args.output_dir) / 'args.json', 'w') as fid:
        json.dump(vars(args), fid)


if __name__ == '__main__':
    main()
