import os
import json
import torch

from entropy import (
    calculate_transient_entropy_unigram,
    calculate_transient_entropy_bigram
)

results = sorted([
    f for f in os.listdir('results') if 'grams' in f
])
assert sorted(list(set(results))) == sorted(results)

sequence_length = 64
batch_size = 4
bos_token_id = 0
eos_token_id = 1
pad_token_id = 2

def get_entropy(name, table):
    if 'unigram' in name:
        return calculate_transient_entropy_unigram(
            torch.tensor(table),
            sequence_length,
            batch_size,
            bos_token_id,
            eos_token_id,
            pad_token_id
        )
    elif 'bigram' in name:
        return calculate_transient_entropy_bigram(
            torch.tensor(table),
            sequence_length,
            batch_size,
            bos_token_id,
            eos_token_id,
            pad_token_id
        )
    else:
        raise ValueError('Unknown n-gram type')

name_to_entropy = {}

for result in results:
    try:
        with open(f'results/{result}') as f:
            table = json.load(f)['table']
            name = result.replace('.json', '')
            entropy = get_entropy(name, table)
            name_to_entropy[name] = entropy
            print(f'{name}: {entropy}')
    except Exception as e:
        print(e)
        print(f'Failed to load {result}')
        
with open('results/transient_entropies.json', 'w+', encoding='utf-8') as f:
    json.dump(name_to_entropy, f, ensure_ascii=False, indent=4)