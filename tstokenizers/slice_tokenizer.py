'''
python slice_tokenizer.py \
  --model_path=path \
  --output_path=path \
  --new_vocab_size=size
'''

from tokenizers import models
from transformers import AutoTokenizer
import argparse
import json

def slice_tokenizer(model_path, output_path, new_vocab_size):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    old_vocab_size = tokenizer.vocab_size
    assert new_vocab_size < old_vocab_size, 'New_vocab_size must be less than old_vocab_size'
    model_state = json.loads(tokenizer.backend_tokenizer.model.__getstate__())
    new_vocab = {k: v for k, v in model_state['vocab'].items() if v < new_vocab_size}
    new_merges = [tuple(item) for item in model_state['merges'][:new_vocab_size-old_vocab_size]]
    tokenizer.backend_tokenizer.model = models.BPE(new_vocab, new_merges)
    tokenizer.save_pretrained(output_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
    )
    parser.add_argument(
        '--output_path',
        type=str,
    )
    parser.add_argument(
        '--new_vocab_size',
        type=int,
    )
    return parser.parse_args()

def main():
    args = parse_args()
    slice_tokenizer(args.model_path, args.output_path, args.new_vocab_size)

if __name__ == '__main__':
    main()
