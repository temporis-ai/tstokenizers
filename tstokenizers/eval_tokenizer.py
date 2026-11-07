'''
python eval_tokenizer.py \
  --model_path=path \
  --dataset_path=path \
  --num_proc=number
'''

from datasets import load_from_disk
from transformers import AutoTokenizer
import argparse

def eval_tokenizer(model_path, dataset_path, num_proc):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    def add_lengths(example):
        text = example['text']
        example['text_length'] = len(text)
        example['input_ids_length'] = len(tokenizer(text)['input_ids'])
        return example

    dataset = load_from_disk(dataset_path)
    dataset = dataset.map(add_lengths, num_proc=num_proc)
    return sum(dataset['input_ids_length']) / sum(dataset['text_length'])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
    )
    parser.add_argument(
        '--num_proc',
        type=int,
    )
    return parser.parse_args()

def main():
    args = parse_args()
    tokens_per_char = eval_tokenizer(args.model_path, args.dataset_path, args.num_proc)
    print(f'{tokens_per_char=}')

if __name__ == '__main__':
    main()
