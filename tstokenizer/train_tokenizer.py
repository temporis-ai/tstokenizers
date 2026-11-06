'''
python -m tstokenizer.train_tokenizer \
  --dataset-path=path \
  --output-path=path \
  --vocab-size=size \
  --add-evt-token
'''

from datasets import load_from_disk
from tokenizers import Tokenizer, decoders, models, trainers
from transformers import PreTrainedTokenizerFast
from tstokenizer.codebook import ALPHABET
import argparse

def train_tokenizer(dataset_path, output_path, vocab_size, add_evt_token=False):
    dataset = load_from_disk(dataset_path)
    def iterator():
        for batch in dataset.iter(batch_size=1000):
            yield batch['text']

    tokenizer = Tokenizer(models.BPE())
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=ALPHABET,
        special_tokens=['[PAD]', '[BOS]'],
    )
    tokenizer.train_from_iterator(iterator(), trainer, len(dataset))
    tokenizer.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token='[PAD]',
        bos_token='[BOS]',
        clean_up_tokenization_spaces=False,
    )
    if add_evt_token:
        tokenizer.add_special_tokens({'additional_special_tokens': ['[EVT]']})
    tokenizer.save_pretrained(output_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-path',
        type=str,
    )
    parser.add_argument(
        '--output-path',
        type=str,
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
    )
    parser.add_argument(
        '--add-evt-token',
        action='store_true',
    )
    return parser.parse_args()

def main():
    args = parse_args()
    train_tokenizer(args.dataset_path, args.output_path, args.vocab_size, args.add_evt_token)

if __name__ == '__main__':
    main()
