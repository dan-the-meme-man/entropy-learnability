from itertools import product

from unigram_tables import create_uniform_unigram_table, create_normal_unigram_table
from bigram_tables import create_normal_bigram_table, create_uneven_bigram_table
from entropy import calculate_entropy_unigram, calculate_entropy_bigram

hparams = {
    'name_table_and_entropy_functions': [
        ('uu', create_uniform_unigram_table, calculate_entropy_unigram),
        ('nu', create_normal_unigram_table, calculate_entropy_unigram),
        ('nb', create_normal_bigram_table, calculate_entropy_bigram),
        ('ub', create_uneven_bigram_table, calculate_entropy_bigram)
    ],
    'vocab_size': [
        10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10_000
    ],
    'softmax': [True, False]
}

for table_and_entropy_functions, vocab_size, softmax in product(*hparams.values()):
    name, table_function, entropy_function = table_and_entropy_functions
    
    if name == 'uu' and softmax:
        continue
    if name == 'ub' and (vocab_size > 1000 or softmax):
        continue
    
    t = table_function(vocab_size, softmax)
    msg = f'Name: {name}, Vocab size: {vocab_size}'
    msg += f', Softmax: {softmax}, Entropy: {entropy_function(t)}'
    print(msg)
